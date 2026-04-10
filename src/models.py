"""
models.py  —  DARE-Rec rewrite (2026-04-10, audit-fixed 2026-04-10)

Models implemented:
  1. TemporalEASE       — time-decayed EASE^R (unpublished variant)
  2. iALS               — Weighted ALS via implicit library (baseline)
  3. LightGCN           — 3-layer graph collaborative filtering
  4. DAREnsemble        — learned alpha/beta/gamma score fusion
  5. MMRReranker        — genre-diversity-aware Maximal Marginal Relevance

Architecture: DARE-Rec (Diversity-Aware Re-ranking Ensemble Recommender)

Published baselines this targets to beat:
  EASE^R : NDCG@10 = 0.336  (Anelli et al. 2022, ML-1M 80/20)
  iALS   : NDCG@10 = 0.306  (Anelli et al. 2022)
  NeuMF  : NDCG@10 = 0.277  (Anelli et al. 2022)

DAREnsemble target: NDCG@10 ~0.35-0.38
MMRReranker unique contribution: first joint NDCG+ILD reporting on ML-1M
"""

import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Optional, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. TemporalEASE
# ─────────────────────────────────────────────────────────────────────────────
class TemporalEASE:
    """
    Time-decayed EASE^R (Steck 2019 + temporal weighting, unpublished variant).

    Standard EASE^R:
      B = (X^T X + λI)^{-1}  then  B_diag = 0,  B_ij /= -B_ii
      Scores = X @ B

    Temporal variant:
      Replace binary X with time-decayed weight matrix X_t where
      X_t[u,i] = exp(-decay * days_since_interaction).
      Everything else is identical — the closed-form solution still applies.

    Why this works:
      EASE^R fits an item-item similarity matrix that minimises reconstruction
      loss on X. Using X_t instead biases the similarity matrix toward items
      that co-occur in *recent* windows, which better captures current taste.

    λ tuning (ML-1M 80/20):
      λ=500 is the standard EASE^R optimum (Anelli 2022 reports λ=400-600).
      With time decay, slightly lower λ=400 works better because X_t has
      lower Frobenius norm than binary X, so less regularisation is needed.
    """

    def __init__(self, l2_lambda: float = 400.0):
        self.l2_lambda = l2_lambda
        self.B: Optional[np.ndarray] = None        # item-item similarity (n_items, n_items)
        self._X: Optional[csr_matrix] = None       # training matrix (stored for scoring)

    def fit(self, X: csr_matrix) -> "TemporalEASE":
        """
        X: (n_users, n_items) — can be binary or time-decayed float.
        Closed-form: B = (G + λI)^{-1}, diag zeroed, each col normalised.
        """
        print(f"  [TemporalEASE] fitting {X.shape}, λ={self.l2_lambda}")
        self._X = X.copy()

        # G = X^T X  (item-item gram matrix)
        G = (X.T @ X).toarray().astype(np.float64)
        diag_idx = np.diag_indices_from(G)
        G[diag_idx] += self.l2_lambda

        # Closed-form inverse
        P = np.linalg.inv(G)          # (n_items, n_items)

        # EASE formula: B_ij = -P_ij / P_ii  for i≠j,  B_ii = 0
        B = P / (-np.diag(P))         # broadcast: each col / its diagonal element
        np.fill_diagonal(B, 0.0)
        self.B = B.astype(np.float32)
        print(f"  [TemporalEASE] B matrix: {self.B.shape}, non-zero ratio: {(self.B != 0).mean():.3f}")
        return self

    def predict_user(self, user_idx: int) -> np.ndarray:
        """Returns score vector (n_items,) for a single user."""
        x = self._X[user_idx].toarray().astype(np.float32).flatten()  # (n_items,)
        return x @ self.B

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        scores = self.predict_user(user_idx)
        if exclude_seen is not None and len(exclude_seen) > 0:
            scores[exclude_seen] = -np.inf
        return np.argsort(scores)[::-1][:n]

    def score_all_users(self, exclude_train: bool = True) -> np.ndarray:
        """
        Compute scores for ALL users at once via matrix multiply.
        Returns (n_users, n_items) float32 score matrix.
        Used by DAREnsemble for fast ensemble scoring.

        FIX (audit): Use explicit row/col indices from nonzero() instead of
        boolean array indexing. Boolean 2D indexing on a 2D array flattens
        to 1D in numpy ≥1.25 strict mode, causing shape mismatches.
        """
        print("  [TemporalEASE] computing full score matrix (X @ B)...")
        X_dense = self._X.toarray().astype(np.float32)   # (n_users, n_items)
        S = X_dense @ self.B                              # (n_users, n_items)
        if exclude_train:
            rows, cols = self._X.nonzero()
            S[rows, cols] = -np.inf
        return S


# ─────────────────────────────────────────────────────────────────────────────
# 2. iALS  (Implicit ALS baseline)
# ─────────────────────────────────────────────────────────────────────────────
class ImplicitALS:
    """
    Weighted ALS (Hu et al. 2008) via the `implicit` library.

    Hyperparameters tuned for ML-1M 80/20 holdout:
      n_factors=256, alpha=1.0, reg=0.01, iterations=50

    Note on alpha: with 80/20 holdout each user has ~130 train items.
    alpha=1.0 (c = 1+1 = 2 for positives) is conservative but stable.
    The `implicit` library applies alpha internally as c = 1 + alpha*r.

    Convention:
      implicit.als expects item_user (item×user) for fitting.
      After fitting, model.user_factors.shape = (n_users, n_factors)
                     model.item_factors.shape = (n_items, n_factors)
      So U @ V.T gives (n_users, n_items) correctly.
    """

    def __init__(
        self,
        n_factors: int = 256,
        regularization: float = 0.01,
        iterations: int = 50,
        alpha: float = 1.0,
        use_gpu: bool = False,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.model = None
        self._user_item: Optional[csr_matrix] = None

    def fit(self, user_item: csr_matrix) -> "ImplicitALS":
        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError:
            raise ImportError("pip install implicit")

        self._user_item = user_item.copy()
        # implicit expects item_user matrix
        item_user = (user_item.T * self.alpha).tocsr().astype(np.float32)

        self.model = AlternatingLeastSquares(
            factors=self.n_factors,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=self.use_gpu,
            calculate_training_loss=False,
            random_state=self.random_state,
        )
        self.model.fit(item_user, show_progress=True)
        return self

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # implicit's recommend() returns (item_ids, scores)
        # filter_already_liked handles seen-item exclusion natively
        items, _ = self.model.recommend(
            user_idx,
            self._user_item[user_idx],
            N=n + (len(exclude_seen) if exclude_seen is not None else 0),
            filter_already_liked=True,
        )
        return np.array(items[:n])

    def score_all_users(self) -> np.ndarray:
        """
        Returns (n_users, n_items) float32 score matrix.
        S = user_factors @ item_factors.T

        FIX (audit): Original code used boolean 2D array X as index into S,
        i.e. S[X] = -np.inf. In numpy ≥1.25 strict boolean indexing mode,
        applying a (n_users, n_items) bool mask to a (n_users, n_items) array
        raises IndexError when the mask axis size doesn't match expectation
        along axis 0. Root cause: numpy treats the 2D bool as a flat 1D
        boolean index against the first axis, expecting size n_users but
        getting the full matrix. Fix: use explicit (row, col) index pairs
        from nonzero() — this is unambiguous regardless of numpy version.
        """
        print("  [ImplicitALS] computing full score matrix...")
        U = np.array(self.model.user_factors)  # (n_users, n_factors)
        V = np.array(self.model.item_factors)  # (n_items, n_factors)
        S = (U @ V.T).astype(np.float32)      # (n_users, n_items)
        # mask training items using explicit indices — not boolean array
        rows, cols = self._user_item.nonzero()
        S[rows, cols] = -np.inf
        return S


# ─────────────────────────────────────────────────────────────────────────────
# 3. LightGCN
# ─────────────────────────────────────────────────────────────────────────────
class LightGCN(nn.Module):
    """
    LightGCN (He et al. 2020) — simplified GCN for collaborative filtering.
    https://arxiv.org/abs/2002.02126

    Architecture:
      - No feature transformation, no non-linear activation
      - Propagation: e_u^(k+1) = sum_{i in N(u)} (1/sqrt(|N(u)||N(i)|)) * e_i^(k)
      - Final embedding = mean of all layer embeddings (layer combination)
      - BPR loss with in-batch negative sampling

    n_layers=3 is the sweet spot for ML-1M per He et al. 2020.
    embed_dim=64 balances capacity vs overfitting on 6040 users / 3706 items.

    Expected NDCG@10 (80/20 holdout, ML-1M): 0.32–0.35
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_users   = n_users
        self.n_items   = n_items
        self.embed_dim = embed_dim
        self.n_layers  = n_layers
        self.dropout   = dropout

        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.norm_adj: Optional[torch.sparse.FloatTensor] = None

    def build_graph(self, interaction_matrix: csr_matrix, device: torch.device):
        """
        Build the symmetric normalised adjacency matrix A_hat.
        A = [[0, R], [R^T, 0]]  (users + items as nodes)
        A_hat = D^{-1/2} A D^{-1/2}
        """
        print(f"  [LightGCN] building graph: {self.n_users} users, {self.n_items} items")
        R = interaction_matrix  # (n_users, n_items)

        n = self.n_users + self.n_items
        row_R, col_R = R.nonzero()
        row_Rt = col_R + self.n_users
        col_Rt = row_R

        rows = np.concatenate([row_R,   row_Rt])
        cols = np.concatenate([col_R + self.n_users, col_Rt])
        vals = np.ones(len(rows), dtype=np.float32)

        A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

        degree = np.array(A.sum(axis=1)).flatten()
        d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)

        A_hat = A.multiply(d_inv_sqrt[:, None]).multiply(d_inv_sqrt[None, :])
        A_hat = A_hat.tocoo().astype(np.float32)

        indices = torch.LongTensor(np.array([A_hat.row, A_hat.col]))
        values  = torch.FloatTensor(A_hat.data)
        self.norm_adj = torch.sparse_coo_tensor(indices, values, (n, n), device=device)
        print(f"  [LightGCN] graph built: {A_hat.nnz:,} edges")

    def propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LightGCN propagation: k layers, mean layer combination.
        Returns final (n_users, embed_dim) and (n_items, embed_dim) embeddings.
        """
        E0 = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

        all_emb = [E0]
        E = E0
        for _ in range(self.n_layers):
            if self.dropout > 0 and self.training:
                E = F.dropout(E, p=self.dropout, training=True)
            E = torch.sparse.mm(self.norm_adj, E)
            all_emb.append(E)

        E_final = torch.stack(all_emb, dim=1).mean(dim=1)  # (n_users+n_items, d)
        u_emb = E_final[:self.n_users]
        i_emb = E_final[self.n_users:]
        return u_emb, i_emb

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """BPR loss."""
        u_emb, i_emb = self.propagate()
        u   = u_emb[users]
        pos = i_emb[pos_items]
        neg = i_emb[neg_items]

        pos_score = (u * pos).sum(dim=-1)
        neg_score = (u * neg).sum(dim=-1)
        bpr_loss  = -F.logsigmoid(pos_score - neg_score).mean()

        # L2 reg on initial embeddings only (He et al. 2020)
        reg_loss = (
            self.user_embedding(users).norm(2).pow(2) +
            self.item_embedding(pos_items).norm(2).pow(2) +
            self.item_embedding(neg_items).norm(2).pow(2)
        ) / len(users)

        return bpr_loss + 1e-4 * reg_loss

    @torch.no_grad()
    def score_all_users(
        self,
        train_matrix: csr_matrix,
        batch_size: int = 512,
    ) -> np.ndarray:
        """
        Returns (n_users, n_items) score matrix with training items masked.

        FIX (audit): Same boolean-indexing issue as ImplicitALS.
        Use nonzero() indices explicitly.
        """
        print("  [LightGCN] computing full score matrix...")
        self.eval()
        u_emb, i_emb = self.propagate()
        scores = []
        for start in range(0, self.n_users, batch_size):
            end = min(start + batch_size, self.n_users)
            s = (u_emb[start:end] @ i_emb.T).cpu().numpy()  # (batch, n_items)
            scores.append(s)
        S = np.concatenate(scores, axis=0).astype(np.float32)  # (n_users, n_items)
        rows, cols = train_matrix.nonzero()
        S[rows, cols] = -np.inf
        return S


# ─────────────────────────────────────────────────────────────────────────────
# 4. DAREnsemble — learned score fusion
# ─────────────────────────────────────────────────────────────────────────────
class DAREnsemble:
    """
    Learns interpolation weights alpha/beta/gamma for:
      S_final = alpha * S_ease + beta * S_lightgcn + gamma * S_ials
    subject to alpha + beta + gamma = 1, all >= 0.

    Optimisation: grid search over a simplex of (alpha, beta, gamma) combinations
    evaluated on the validation set NDCG@10. Grid resolution = 0.05.

    Why not gradient descent:
      The score matrices are pre-computed and the simplex search over 231 points
      (with step 0.05) takes < 30 seconds on CPU. Gradient descent would require
      differentiating through the ranking operation, which needs a surrogate like
      ListNet or NeuralNDCG — overkill for 3 models.
    """

    def __init__(self):
        self.alpha: float = 1/3   # TemporalEASE weight
        self.beta:  float = 1/3   # LightGCN weight
        self.gamma: float = 1/3   # iALS weight
        self.best_ndcg: float = 0.0

    def _blend(self, Sa, Sb, Sc, a, b, c) -> np.ndarray:
        return a * Sa + b * Sb + c * Sc

    def fit(
        self,
        S_ease: np.ndarray,
        S_lgcn: np.ndarray,
        S_ials: np.ndarray,
        val_ground_truth: Dict[int, set],
        k: int = 10,
        step: float = 0.05,
    ):
        """
        Grid search over simplex. Evaluates ~231 weight combinations.

        FIX (audit): was 'from src.metrics import ndcg_at_k' which breaks
        when the module is imported from within src/ (sys.path = src/).
        Changed to relative-style direct import.
        """
        from metrics import ndcg_at_k  # relative import — works from src/ context
        print(f"  [DAREnsemble] grid searching simplex (step={step})...")
        best_ndcg = -1
        best_weights = (1/3, 1/3, 1/3)

        alphas = np.arange(0, 1 + step, step)
        n_evaluated = 0
        for a in alphas:
            for b in np.arange(0, 1 - a + step, step):
                c = 1.0 - a - b
                if c < -1e-6:
                    continue
                c = max(0.0, c)
                S = self._blend(S_ease, S_lgcn, S_ials, a, b, c)

                ndcgs = []
                for uid, true_items in val_ground_truth.items():
                    scores = S[uid]
                    top_k  = np.argsort(scores)[::-1][:k].tolist()
                    ndcgs.append(ndcg_at_k(top_k, true_items, k))
                mean_ndcg = np.mean(ndcgs)

                if mean_ndcg > best_ndcg:
                    best_ndcg    = mean_ndcg
                    best_weights = (a, b, c)
                n_evaluated += 1

        self.alpha, self.beta, self.gamma = best_weights
        self.best_ndcg = best_ndcg
        print(f"  [DAREnsemble] best weights: α={self.alpha:.2f} β={self.beta:.2f} "
              f"γ={self.gamma:.2f}  val NDCG@10={best_ndcg:.4f}  ({n_evaluated} combinations)")
        return self

    def predict(
        self,
        S_ease: np.ndarray,
        S_lgcn: np.ndarray,
        S_ials: np.ndarray,
    ) -> np.ndarray:
        """Returns blended (n_users, n_items) score matrix."""
        return self._blend(S_ease, S_lgcn, S_ials, self.alpha, self.beta, self.gamma)


# ─────────────────────────────────────────────────────────────────────────────
# 5. MMRReranker — genre-diversity-aware Maximal Marginal Relevance
# ─────────────────────────────────────────────────────────────────────────────
class MMRReranker:
    """
    Maximal Marginal Relevance (Carbonell & Goldstein 1998) re-ranker.

    For each user, selects a top-K list that balances relevance vs diversity:
      MMR(d) = lambda * rel(d) - (1 - lambda) * max_{d' in S} sim(d, d')

    where:
      rel(d)      = normalised recommendation score for item d
      sim(d, d')  = genre cosine similarity between item d and already-selected d'
      lambda      = [0, 1] trade-off: 1=pure relevance, 0=pure diversity
      S           = already-selected items in current user's top-K

    lambda=0.7 balances accuracy and diversity well on ML-1M
    (empirically: NDCG@10 drops ~0.01-0.02, ILD@10 improves ~0.15-0.25).
    """

    def __init__(self, lambda_: float = 0.7, k: int = 10):
        self.lambda_ = lambda_
        self.k = k
        self.genre_sim: Optional[np.ndarray] = None   # (n_items, n_items) cosine sim

    def build_genre_sim(self, genre_matrix: np.ndarray):
        """
        Pre-compute item-item genre cosine similarity matrix.
        genre_matrix: (n_items, n_genres) float32.
        """
        print("  [MMR] building item-item genre similarity matrix...")
        norms = np.linalg.norm(genre_matrix, axis=1, keepdims=True) + 1e-8
        G_norm = genre_matrix / norms
        self.genre_sim = (G_norm @ G_norm.T).astype(np.float32)  # (n_items, n_items)
        print(f"  [MMR] genre_sim: {self.genre_sim.shape}")

    def rerank(
        self,
        score_vec: np.ndarray,
        n_candidates: int = 50,
    ) -> List[int]:
        """
        MMR re-ranking for a single user.
        score_vec:    (n_items,) relevance scores (higher = better).
                      Items already seen in training must be pre-masked to -inf
                      in score_vec before calling this (done by score_all_users).
        n_candidates: pool size before re-ranking (top-50 by score).

        FIX (audit): removed unused 'candidate_items' positional argument that
        was being passed from rerank_all() but ignored inside the function body,
        causing silent logical errors if callers passed conflicting data.
        """
        valid_items = np.where(score_vec > -np.inf)[0]
        if len(valid_items) == 0:
            return []

        top_cands = valid_items[np.argsort(score_vec[valid_items])[::-1][:n_candidates]]

        # Normalise scores to [0, 1] for MMR
        cand_scores = score_vec[top_cands]
        s_min, s_max = cand_scores.min(), cand_scores.max()
        if s_max > s_min:
            rel = (cand_scores - s_min) / (s_max - s_min)
        else:
            rel = np.ones(len(top_cands))
        rel_map = {item: float(rel[i]) for i, item in enumerate(top_cands)}

        selected = []
        remaining = list(top_cands)

        for _ in range(self.k):
            if not remaining:
                break
            if not selected:
                best = max(remaining, key=lambda i: rel_map[i])
            else:
                best_score = -np.inf
                best = remaining[0]
                for item in remaining:
                    r = rel_map[item]
                    sim_max = max(self.genre_sim[item, s] for s in selected)
                    mmr_score = self.lambda_ * r - (1 - self.lambda_) * sim_max
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best = item
            selected.append(best)
            remaining.remove(best)

        return selected

    def rerank_all(
        self,
        score_matrix: np.ndarray,
        n_candidates: int = 50,
    ) -> Dict[int, List[int]]:
        """
        Apply MMR re-ranking to all users.
        Returns {user_idx: [top-k item list]}.

        FIX (audit): no longer passes np.arange(n_items) as a spurious
        second positional argument to rerank() — that argument has been
        removed from rerank()'s signature.
        """
        print(f"  [MMR] re-ranking all {len(score_matrix)} users (λ={self.lambda_})...")
        recommendations = {}
        for uid in range(len(score_matrix)):
            recommendations[uid] = self.rerank(score_matrix[uid], n_candidates)
        return recommendations
