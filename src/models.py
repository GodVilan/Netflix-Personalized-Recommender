"""
models.py  —  DARE-Rec (2026-04-10, audit-fixed, ials-fix, emb-fix)

Models implemented:
  1. TemporalEASE       — time-decayed EASE^R (unpublished variant)
  2. iALS               — Weighted ALS via implicit library (baseline)
  3. LightGCN           — 3-layer graph collaborative filtering
  4. DAREnsemble        — learned alpha/beta/gamma score fusion
  5. MMRReranker        — genre-diversity-aware Maximal Marginal Relevance

Architecture: DARE-Rec (Diversity-Aware Re-ranking Ensemble Recommender)

LightGCN forward() update (2026-04-10):
  forward() now accepts neg_items of shape (B,) or (B, n_neg).
  When neg_items is 2-D, BPR loss is averaged over all n_neg negatives
  per sample (vectorised via einsum — no extra forward passes).
  This enables n_neg=8 training from trainer.py without changing the
  model–trainer interface: trainer calls model(u, pos, negs) as before.

Published baselines this targets to beat:
  EASE^R : NDCG@10 = 0.336  (Anelli et al. 2022, ML-1M 80/20)
  iALS   : NDCG@10 = 0.306  (Anelli et al. 2022)
  NeuMF  : NDCG@10 = 0.277  (Anelli et al. 2022)
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


# ───────────────────────────────────────────────────────────────────
# 1. TemporalEASE
# ───────────────────────────────────────────────────────────────────
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
    """

    def __init__(self, l2_lambda: float = 400.0):
        self.l2_lambda = l2_lambda
        self.B: Optional[np.ndarray] = None
        self._X: Optional[csr_matrix] = None

    def fit(self, X: csr_matrix) -> "TemporalEASE":
        print(f"  [TemporalEASE] fitting {X.shape}, λ={self.l2_lambda}")
        self._X = X.copy()
        G = (X.T @ X).toarray().astype(np.float64)
        diag_idx = np.diag_indices_from(G)
        G[diag_idx] += self.l2_lambda
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        np.fill_diagonal(B, 0.0)
        self.B = B.astype(np.float32)
        print(f"  [TemporalEASE] B matrix: {self.B.shape}, non-zero ratio: {(self.B != 0).mean():.3f}")
        return self

    def predict_user(self, user_idx: int) -> np.ndarray:
        x = self._X[user_idx].toarray().astype(np.float32).flatten()
        return x @ self.B

    def recommend(self, user_idx: int, n: int = 10,
                  exclude_seen: Optional[np.ndarray] = None) -> np.ndarray:
        scores = self.predict_user(user_idx)
        if exclude_seen is not None and len(exclude_seen) > 0:
            scores[exclude_seen] = -np.inf
        return np.argsort(scores)[::-1][:n]

    def score_all_users(self, exclude_train: bool = True) -> np.ndarray:
        print("  [TemporalEASE] computing full score matrix (X @ B)...")
        X_dense = self._X.toarray().astype(np.float32)
        S = X_dense @ self.B
        if exclude_train:
            rows, cols = self._X.nonzero()
            S[rows, cols] = -np.inf
        return S


# ───────────────────────────────────────────────────────────────────
# 2. iALS
# ───────────────────────────────────────────────────────────────────
class ImplicitALS:
    """
    Weighted ALS (Hu et al. 2008) via the `implicit` library.
    Hyperparameters tuned for ML-1M 80/20 holdout.
    """

    def __init__(self, n_factors: int = 256, regularization: float = 0.01,
                 iterations: int = 50, alpha: float = 1.0,
                 use_gpu: bool = False, random_state: int = 42):
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.model = None
        self._user_item: Optional[csr_matrix] = None
        self.n_users: Optional[int] = None
        self.n_items: Optional[int] = None

    def fit(self, user_item: csr_matrix) -> "ImplicitALS":
        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError:
            raise ImportError("pip install implicit")

        self._user_item = user_item.tocsr().astype(np.float32).copy()
        self.n_users, self.n_items = self._user_item.shape
        weighted = (self._user_item * self.alpha).tocsr().astype(np.float32)

        self.model = AlternatingLeastSquares(
            factors=self.n_factors, regularization=self.regularization,
            iterations=self.iterations, use_gpu=self.use_gpu,
            calculate_training_loss=False, random_state=self.random_state,
        )
        self.model.fit(weighted, show_progress=True)

        uf = np.asarray(self.model.user_factors)
        vf = np.asarray(self.model.item_factors)
        if uf.shape[0] != self.n_users or vf.shape[0] != self.n_items:
            raise ValueError(
                f"[ImplicitALS] Factor shape mismatch: "
                f"user_factors={uf.shape}, item_factors={vf.shape}"
            )
        return self

    def recommend(self, user_idx: int, n: int = 10,
                  exclude_seen: Optional[np.ndarray] = None) -> np.ndarray:
        try:
            items, _ = self.model.recommend(
                user_idx, self._user_item[user_idx], N=n,
                filter_already_liked_items=True,
            )
        except TypeError:
            items, _ = self.model.recommend(
                user_idx, self._user_item[user_idx], N=n,
                filter_already_liked=True,
            )
        return np.array(items[:n])

    def score_all_users(self) -> np.ndarray:
        print("  [ImplicitALS] computing full score matrix...")
        U = np.asarray(self.model.user_factors, dtype=np.float32)
        V = np.asarray(self.model.item_factors, dtype=np.float32)
        if U.shape[0] != self.n_users or V.shape[0] != self.n_items:
            raise ValueError(f"[ImplicitALS] score_all_users shape mismatch")
        S = (U @ V.T).astype(np.float32)
        rows, cols = self._user_item.nonzero()
        S[rows, cols] = -np.inf
        return S


# ───────────────────────────────────────────────────────────────────
# 3. LightGCN
# ───────────────────────────────────────────────────────────────────
class LightGCN(nn.Module):
    """
    LightGCN (He et al. 2020) — simplified GCN for collaborative filtering.
    https://arxiv.org/abs/2002.02126

    forward() accepts neg_items of shape (B,) or (B, n_neg).
    When 2-D, BPR loss is averaged over all n_neg negatives (vectorised).
    """

    def __init__(self, n_users: int, n_items: int,
                 embed_dim: int = 64, n_layers: int = 3, dropout: float = 0.1):
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
        print(f"  [LightGCN] building graph: {self.n_users} users, {self.n_items} items")
        R = interaction_matrix
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
        self.norm_adj = torch.sparse_coo_tensor(
            indices, values, (n, n), device=device
        ).coalesce()
        print(f"  [LightGCN] graph built: {A_hat.nnz:,} edges")

    def propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LightGCN propagation: k layers, mean layer combination.
        Requires norm_adj to be set via build_graph().
        Returns (n_users, d) and (n_items, d) final embeddings.
        """
        assert self.norm_adj is not None, \
            "norm_adj is None — call build_graph() before propagate()"
        E0 = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight,
        ], dim=0)
        all_emb = [E0]
        E = E0
        for _ in range(self.n_layers):
            if self.dropout > 0 and self.training:
                E = F.dropout(E, p=self.dropout, training=True)
            E = torch.sparse.mm(self.norm_adj, E)
            all_emb.append(E)
        E_final = torch.stack(all_emb, dim=1).mean(dim=1)
        return E_final[:self.n_users], E_final[self.n_users:]

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """
        BPR loss.
        neg_items: (B,) or (B, n_neg).
        When (B, n_neg), loss is averaged over all n_neg negatives.
        """
        u_emb, i_emb = self.propagate()

        u   = u_emb[users]      # (B, d)
        pos = i_emb[pos_items]  # (B, d)
        pos_score = (u * pos).sum(dim=-1)  # (B,)

        if neg_items.dim() == 1:
            # Original path: single negative per sample
            neg = i_emb[neg_items]            # (B, d)
            neg_score = (u * neg).sum(dim=-1) # (B,)
            bpr_loss = -F.logsigmoid(pos_score - neg_score).mean()
            reg_loss = (
                self.user_embedding(users).norm(2).pow(2) +
                self.item_embedding(pos_items).norm(2).pow(2) +
                self.item_embedding(neg_items).norm(2).pow(2)
            ) / len(users)
        else:
            # n_neg path: neg_items is (B, n_neg)
            # neg: (B, n_neg, d)
            neg = i_emb[neg_items.view(-1)].view(
                neg_items.size(0), neg_items.size(1), -1
            )
            # neg_score: (B, n_neg)
            neg_score = torch.einsum("bd,bnd->bn", u, neg)
            bpr_loss = -F.logsigmoid(
                pos_score.unsqueeze(1) - neg_score
            ).mean()
            reg_loss = (
                self.user_embedding(users).norm(2).pow(2) +
                self.item_embedding(pos_items).norm(2).pow(2) +
                self.item_embedding(neg_items.view(-1)).norm(2).pow(2)
            ) / len(users)

        return bpr_loss + 1e-4 * reg_loss

    @torch.no_grad()
    def score_all_users(
        self,
        train_matrix: csr_matrix,
        batch_size: int = 512,
    ) -> np.ndarray:
        print("  [LightGCN] computing full score matrix...")
        self.eval()
        u_emb, i_emb = self.propagate()
        scores = []
        for start in range(0, self.n_users, batch_size):
            end = min(start + batch_size, self.n_users)
            s = (u_emb[start:end] @ i_emb.T).cpu().numpy()
            scores.append(s)
        S = np.concatenate(scores, axis=0).astype(np.float32)
        rows, cols = train_matrix.nonzero()
        S[rows, cols] = -np.inf
        return S


# ───────────────────────────────────────────────────────────────────
# 4. DAREnsemble
# ───────────────────────────────────────────────────────────────────
class DAREnsemble:
    """
    Learns interpolation weights alpha/beta/gamma for:
      S_final = alpha * S_ease + beta * S_lightgcn + gamma * S_ials
    subject to alpha + beta + gamma = 1, all >= 0.
    Grid search over simplex (step=0.05, 231 combinations).
    """

    def __init__(self):
        self.alpha: float = 1/3
        self.beta:  float = 1/3
        self.gamma: float = 1/3
        self.best_ndcg: float = 0.0

    def _blend(self, Sa, Sb, Sc, a, b, c):
        return a * Sa + b * Sb + c * Sc

    def fit(self, S_ease, S_lgcn, S_ials, val_ground_truth, k=10, step=0.05):
        from metrics import ndcg_at_k
        print(f"  [DAREnsemble] grid searching simplex (step={step})...")
        best_ndcg, best_weights = -1, (1/3, 1/3, 1/3)
        alphas = np.arange(0, 1 + step, step)
        n_evaluated = 0
        for a in alphas:
            for b in np.arange(0, 1 - a + step, step):
                c = max(0.0, 1.0 - a - b)
                if c < -1e-6:
                    continue
                S = self._blend(S_ease, S_lgcn, S_ials, a, b, c)
                ndcgs = [
                    ndcg_at_k(np.argsort(S[uid])[::-1][:k].tolist(), true_items, k)
                    for uid, true_items in val_ground_truth.items()
                ]
                mean_ndcg = np.mean(ndcgs)
                if mean_ndcg > best_ndcg:
                    best_ndcg, best_weights = mean_ndcg, (a, b, c)
                n_evaluated += 1
        self.alpha, self.beta, self.gamma = best_weights
        self.best_ndcg = best_ndcg
        print(f"  [DAREnsemble] best weights: α={self.alpha:.2f} β={self.beta:.2f} "
              f"γ={self.gamma:.2f}  val NDCG@10={best_ndcg:.4f}  ({n_evaluated} combinations)")
        return self

    def predict(self, S_ease, S_lgcn, S_ials):
        return self._blend(S_ease, S_lgcn, S_ials, self.alpha, self.beta, self.gamma)


# ───────────────────────────────────────────────────────────────────
# 5. MMRReranker
# ───────────────────────────────────────────────────────────────────
class MMRReranker:
    """
    Maximal Marginal Relevance (Carbonell & Goldstein 1998).
    MMR(d) = lambda * rel(d) - (1-lambda) * max_{d' in S} sim(d, d')
    lambda=0.7: balances accuracy and diversity on ML-1M.
    """

    def __init__(self, lambda_: float = 0.7, k: int = 10):
        self.lambda_ = lambda_
        self.k = k
        self.genre_sim: Optional[np.ndarray] = None

    def build_genre_sim(self, genre_matrix: np.ndarray):
        print("  [MMR] building item-item genre similarity matrix...")
        norms = np.linalg.norm(genre_matrix, axis=1, keepdims=True) + 1e-8
        G_norm = genre_matrix / norms
        self.genre_sim = (G_norm @ G_norm.T).astype(np.float32)
        print(f"  [MMR] genre_sim: {self.genre_sim.shape}")

    def rerank(self, score_vec: np.ndarray, n_candidates: int = 50) -> List[int]:
        valid_items = np.where(score_vec > -np.inf)[0]
        if len(valid_items) == 0:
            return []
        top_cands = valid_items[np.argsort(score_vec[valid_items])[::-1][:n_candidates]]
        cand_scores = score_vec[top_cands]
        s_min, s_max = cand_scores.min(), cand_scores.max()
        rel = (cand_scores - s_min) / (s_max - s_min + 1e-8)
        rel_map = {item: float(rel[i]) for i, item in enumerate(top_cands)}
        selected, remaining = [], list(top_cands)
        for _ in range(self.k):
            if not remaining:
                break
            if not selected:
                best = max(remaining, key=lambda i: rel_map[i])
            else:
                best_score, best = -np.inf, remaining[0]
                for item in remaining:
                    sim_max = max(self.genre_sim[item, s] for s in selected)
                    mmr_score = self.lambda_ * rel_map[item] - (1 - self.lambda_) * sim_max
                    if mmr_score > best_score:
                        best_score, best = mmr_score, item
            selected.append(best)
            remaining.remove(best)
        return selected

    def rerank_all(self, score_matrix: np.ndarray,
                   n_candidates: int = 50) -> Dict[int, List[int]]:
        print(f"  [MMR] re-ranking all {len(score_matrix)} users (λ={self.lambda_})...")
        return {uid: self.rerank(score_matrix[uid], n_candidates)
                for uid in range(len(score_matrix))}
