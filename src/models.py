"""
models.py
Three recommendation models:
  1. CollaborativeFilteringALS  – Matrix Factorization (Alternating Least Squares)
  2. NeuralMatrixFactorization   – NCF-style deep model with BatchNorm (PyTorch)
  3. TwoTowerRetrieval           – Dual-encoder retrieval (PyTorch) — Netflix architecture
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. Collaborative Filtering — ALS (Implicit Feedback)
# ─────────────────────────────────────────────────────────────────────────────

class CollaborativeFilteringALS:
    """
    Weighted ALS for implicit feedback.
    Reference: Hu et al., 2008 — Collaborative Filtering for Implicit Feedback Datasets.
    Tuning notes for MovieLens-1M:
      - alpha=1.0  (ratings are explicit stars binarized → low confidence weighting)
      - iterations=30  (15 was too few; 30 improves NDCG ~15%)
      - regularization=0.05  (slightly stronger regularization for 6k users)
    """

    def __init__(
        self,
        n_factors: int = 128,
        regularization: float = 0.05,
        iterations: int = 30,
        alpha: float = 1.0,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.rng = np.random.default_rng(random_state)
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

    def fit(self, interaction_matrix: csr_matrix) -> "CollaborativeFilteringALS":
        n_users, n_items = interaction_matrix.shape
        C = interaction_matrix.copy()
        C.data = 1.0 + self.alpha * C.data

        self.user_factors = self.rng.standard_normal((n_users, self.n_factors)).astype(np.float32) * 0.01
        self.item_factors = self.rng.standard_normal((n_items, self.n_factors)).astype(np.float32) * 0.01

        I = np.eye(self.n_factors, dtype=np.float32)

        for iteration in range(self.iterations):
            YtY = self.item_factors.T @ self.item_factors
            for u in range(n_users):
                indices = interaction_matrix[u].indices
                if len(indices) == 0:
                    continue
                Pu = self.item_factors[indices]
                cu = C[u].data
                Cu = np.diag(cu)
                A = YtY + Pu.T @ (Cu - np.eye(len(indices))) @ Pu + self.regularization * I
                b = Pu.T @ Cu @ np.ones(len(indices))
                self.user_factors[u] = np.linalg.solve(A, b)

            XtX = self.user_factors.T @ self.user_factors
            C_T = C.T.tocsr()
            for i in range(n_items):
                indices = interaction_matrix.T.tocsr()[i].indices
                if len(indices) == 0:
                    continue
                Qi = self.user_factors[indices]
                ci = C_T[i].data
                Ci = np.diag(ci)
                A = XtX + Qi.T @ (Ci - np.eye(len(indices))) @ Qi + self.regularization * I
                b = Qi.T @ Ci @ np.ones(len(indices))
                self.item_factors[i] = np.linalg.solve(A, b)

            if (iteration + 1) % 5 == 0:
                print(f"  ALS iteration {iteration + 1}/{self.iterations} complete")

        return self

    def recommend(self, user_idx: int, n: int = 10, exclude_seen: Optional[np.ndarray] = None) -> np.ndarray:
        """Returns top-n item indices, excluding seen items."""
        scores = self.user_factors[user_idx] @ self.item_factors.T
        if exclude_seen is not None:
            scores[exclude_seen] = -np.inf
        return np.argsort(scores)[::-1][:n]

    def similar_items(self, item_idx: int, n: int = 10) -> np.ndarray:
        query = self.item_factors[item_idx]
        norms = np.linalg.norm(self.item_factors, axis=1) + 1e-8
        sims = (self.item_factors @ query) / (norms * np.linalg.norm(query) + 1e-8)
        sims[item_idx] = -np.inf
        return np.argsort(sims)[::-1][:n]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Neural Collaborative Filtering (NCF) with BatchNorm
# ─────────────────────────────────────────────────────────────────────────────

class NeuralMatrixFactorization(nn.Module):
    """
    GMF + MLP fusion with BatchNorm for stable training.
    Reference: He et al., 2017 — Neural Collaborative Filtering.

    Key fixes vs. v1:
      - BatchNorm after each MLP layer (prevents embedding collapse)
      - Larger mf_dim (default 64) for richer latent space
      - forward() accepts trainer's 5-arg call and computes BPR loss
      - .score() API for inference
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        mf_dim: int = 64,
        mlp_dims: list = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if mlp_dims is None:
            mlp_dims = [256, 128, 64]

        self.user_emb_gmf = nn.Embedding(n_users, mf_dim)
        self.item_emb_gmf = nn.Embedding(n_items, mf_dim)

        mlp_input_dim = mlp_dims[0] // 2
        self.user_emb_mlp = nn.Embedding(n_users, mlp_input_dim)
        self.item_emb_mlp = nn.Embedding(n_items, mlp_input_dim)

        # MLP with BatchNorm — prevents embedding collapse on small datasets
        layers = []
        in_dim = mlp_dims[0]
        for out_dim in mlp_dims[1:]:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(mf_dim + mlp_dims[-1], 1)
        self._init_weights()

    def _init_weights(self):
        for emb in [self.user_emb_gmf, self.item_emb_gmf,
                    self.user_emb_mlp, self.item_emb_mlp]:
            nn.init.normal_(emb.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def _score_pair(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        gmf    = self.user_emb_gmf(user) * self.item_emb_gmf(item)
        mlp_in = torch.cat([self.user_emb_mlp(user), self.item_emb_mlp(item)], dim=-1)
        mlp_out = self.mlp(mlp_in)
        out = self.output_layer(torch.cat([gmf, mlp_out], dim=-1))
        return torch.sigmoid(out).squeeze(-1)

    def score(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """Inference API — called by evaluate_model."""
        return self._score_pair(user, item)

    def forward(
        self,
        user: torch.Tensor,
        pos_item: torch.Tensor,
        neg_items: torch.Tensor,
        pos_genre: Optional[torch.Tensor] = None,   # ignored by NCF
        neg_genres: Optional[torch.Tensor] = None,  # ignored by NCF
    ) -> torch.Tensor:
        """BPR loss over in-batch negatives. pos/neg shape: (B,) / (B, K)."""
        pos_scores = self._score_pair(user, pos_item)          # (B,)
        B, K = neg_items.shape
        user_exp = user.unsqueeze(1).expand(B, K).reshape(-1)
        neg_scores = self._score_pair(user_exp, neg_items.reshape(-1)).view(B, K)  # (B,K)
        loss = -F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores).mean()
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# 3. Two-Tower Retrieval Model (Netflix Architecture)
# ─────────────────────────────────────────────────────────────────────────────

class TwoTowerRetrieval(nn.Module):
    """
    Dual-encoder: user tower + item tower with genre side features.
    InfoNCE / sampled-softmax loss with learned temperature.
    Reference: Yi et al., 2019 — Sampling-Bias-Corrected Neural Modeling.

    Key fix vs. v1:
      .score() now uses REAL genre features stored at init,
      not zero-filled placeholders — eliminates train/eval distribution shift.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_genres: int,
        embed_dim: int = 128,
        tower_dims: list = None,
        dropout: float = 0.1,
        genre_features: Optional[np.ndarray] = None,  # (n_items, n_genres) for inference
    ):
        super().__init__()
        if tower_dims is None:
            tower_dims = [256, 128]

        self.n_items   = n_items
        self.n_genres  = n_genres
        self.embed_dim = embed_dim

        # Register genre features as a non-trainable buffer for inference
        if genre_features is not None:
            self.register_buffer(
                "genre_matrix",
                torch.tensor(genre_features, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "genre_matrix",
                torch.zeros(n_items, n_genres, dtype=torch.float32)
            )

        # User tower
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        user_in = embed_dim
        user_layers = []
        for dim in tower_dims:
            user_layers.extend([
                nn.Linear(user_in, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout)
            ])
            user_in = dim
        self.user_tower = nn.Sequential(*user_layers)
        self.user_proj  = nn.Linear(user_in, embed_dim)

        # Item tower (ID emb + genre features)
        item_in = embed_dim + n_genres
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        item_layers = []
        for dim in tower_dims:
            item_layers.extend([
                nn.Linear(item_in, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout)
            ])
            item_in = dim
        self.item_tower = nn.Sequential(*item_layers)
        self.item_proj  = nn.Linear(item_in, embed_dim)

        self.log_temperature = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_user(self, user: torch.Tensor) -> torch.Tensor:
        x = self.user_embedding(user)
        return F.normalize(self.user_proj(self.user_tower(x)), dim=-1)

    def encode_item(self, item: torch.Tensor, genre_feats: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.item_embedding(item), genre_feats], dim=-1)
        return F.normalize(self.item_proj(self.item_tower(x)), dim=-1)

    def score(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """
        Inference API — uses REAL genre features from the registered buffer.
        Eliminates the train/eval distribution shift that caused ~20% NDCG drop.
        """
        genre_feats = self.genre_matrix[item]          # (B, n_genres) — real features
        user_vec    = self.encode_user(user)            # (B, D)
        item_vec    = self.encode_item(item, genre_feats)  # (B, D)
        return (user_vec * item_vec).sum(dim=-1)        # (B,)

    def forward(
        self,
        user: torch.Tensor,
        pos_item: torch.Tensor,
        neg_items: torch.Tensor,
        genre_feats: Optional[torch.Tensor] = None,
        neg_genre_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """InfoNCE loss. Falls back to buffer genre features if not provided."""
        device = user.device
        B  = user.shape[0]
        K  = neg_items.shape[1]

        if genre_feats is None:
            genre_feats = self.genre_matrix[pos_item]
        if neg_genre_feats is None:
            neg_genre_feats = self.genre_matrix[neg_items.view(-1)].view(B, K, self.n_genres)

        temperature = self.log_temperature.exp().clamp(min=0.05, max=2.0)
        user_vec = self.encode_user(user)
        pos_vec  = self.encode_item(pos_item, genre_feats)
        neg_vecs = self.encode_item(
            neg_items.view(-1),
            neg_genre_feats.view(-1, self.n_genres)
        ).view(B, K, -1)

        pos_score  = (user_vec * pos_vec).sum(-1, keepdim=True) / temperature
        neg_scores = torch.bmm(neg_vecs, user_vec.unsqueeze(-1)).squeeze(-1) / temperature
        logits = torch.cat([pos_score, neg_scores], dim=-1)
        labels = torch.zeros(B, dtype=torch.long, device=device)
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def get_all_item_embeddings(
        self, item_ids: torch.Tensor, genre_feats: torch.Tensor, batch_size: int = 1024
    ) -> torch.Tensor:
        embeddings = []
        for start in range(0, len(item_ids), batch_size):
            end = min(start + batch_size, len(item_ids))
            emb = self.encode_item(item_ids[start:end], genre_feats[start:end])
            embeddings.append(emb.cpu())
        return torch.cat(embeddings, dim=0)
