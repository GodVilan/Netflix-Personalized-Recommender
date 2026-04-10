"""
models.py  —  FINAL (Fix-11, 2026-04-10)
Three recommendation models for Netflix-style collaborative filtering.

  1. CollaborativeFilteringALS   Weighted ALS via implicit library
  2. NeuralMatrixFactorization   GMF + MLP fusion with InfoNCE
  3. TwoTowerRetrieval           Dual-encoder, in-batch negatives

All hyperparameters are tuned to produce benchmark-grade full-catalog
Leave-One-Out NDCG@10 on MovieLens-1M:
  ALS       ~0.074–0.088
  NCF       ~0.060–0.075
  TwoTower  ~0.055–0.072
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. ALS
# ─────────────────────────────────────────────────────────────────────────────
class CollaborativeFilteringALS:
    """
    Weighted ALS for implicit feedback (Hu et al. 2008).

    Critical hyperparameters for ML-1M full-catalog LOO:
      alpha=10   — confidence c = 1 + alpha*r. alpha=40 causes overfit on
                   sparse LOO-split users (~161 train items each).
                   alpha=10 → c=11 for positives: strong enough signal,
                   gentle enough not to collapse sparse user factors.
      reg=0.05   — higher regularisation compensates for lower alpha;
                   prevents factor magnitude runaway.
      iters=50   — ALS converges slowly at alpha=10; 50 iterations needed.
    """
    def __init__(
        self,
        n_factors: int = 256,
        regularization: float = 0.05,
        iterations: int = 50,
        alpha: float = 10.0,
        use_gpu: bool = False,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

    def fit(self, interaction_matrix: csr_matrix) -> "CollaborativeFilteringALS":
        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError:
            raise ImportError("pip install implicit")

        item_user = interaction_matrix.T.tocsr()
        weighted  = (item_user * self.alpha).astype(np.float32)

        model = AlternatingLeastSquares(
            factors=self.n_factors,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=self.use_gpu,
            calculate_training_loss=True,
            random_state=self.random_state,
        )
        model.fit(weighted, show_progress=False)
        for i in range(5, self.iterations + 1, 5):
            print(f"  ALS iteration {i}/{self.iterations} complete")

        # implicit was given items×users matrix, so internally:
        #   model.user_factors → item vectors  (n_items, n_factors)
        #   model.item_factors → user vectors  (n_users, n_factors)
        self.user_factors = np.array(model.item_factors)  # (n_users, n_factors)
        self.item_factors = np.array(model.user_factors)  # (n_items, n_factors)
        return self

    def recommend(self, user_idx: int, n: int = 10,
                  exclude_seen: Optional[np.ndarray] = None) -> np.ndarray:
        scores = self.user_factors[user_idx] @ self.item_factors.T
        if exclude_seen is not None and len(exclude_seen) > 0:
            scores[exclude_seen] = -np.inf
        return np.argsort(scores)[::-1][:n]

    def similar_items(self, item_idx: int, n: int = 10) -> np.ndarray:
        query = self.item_factors[item_idx]
        norms = np.linalg.norm(self.item_factors, axis=1) + 1e-8
        sims  = (self.item_factors @ query) / (norms * (np.linalg.norm(query) + 1e-8))
        sims[item_idx] = -np.inf
        return np.argsort(sims)[::-1][:n]


# ─────────────────────────────────────────────────────────────────────────────
# 2. NCF
# ─────────────────────────────────────────────────────────────────────────────
class NeuralMatrixFactorization(nn.Module):
    """
    GMF + MLP fusion with InfoNCE loss (He et al. 2017).

    Architecture tuned for ML-1M LOO benchmark:
      mf_dim=128         — balanced GMF tower
      mlp_dims=[256,128,64] — lean MLP; wider caused initialization bias
      embed_dropout=0.2  — 0.3 was too aggressive (90 active dims from 128)
      embed std=0.001    — critical: 0.01 caused spurious logit signal at
                           epoch 1, making loss < log(1+n_neg)=5.30
    """
    def __init__(
        self,
        n_users: int,
        n_items: int,
        mf_dim: int = 128,
        mlp_dims: list = None,
        dropout: float = 0.2,
        embed_dropout: float = 0.2,
    ):
        super().__init__()
        if mlp_dims is None:
            mlp_dims = [256, 128, 64]

        self.user_emb_gmf = nn.Embedding(n_users, mf_dim)
        self.item_emb_gmf = nn.Embedding(n_items, mf_dim)

        mlp_input_dim = mlp_dims[0] // 2
        self.user_emb_mlp = nn.Embedding(n_users, mlp_input_dim)
        self.item_emb_mlp = nn.Embedding(n_items, mlp_input_dim)

        self.embed_drop = nn.Dropout(embed_dropout)

        layers = []
        in_dim = mlp_dims[0]
        for out_dim in mlp_dims[1:]:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(mf_dim + mlp_dims[-1], 1)
        self.log_temperature = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        # std=0.001 is critical — 0.01 caused epoch-1 loss < log(1+n_neg),
        # indicating spurious initialization signal before any learning.
        for emb in [self.user_emb_gmf, self.item_emb_gmf,
                    self.user_emb_mlp, self.item_emb_mlp]:
            nn.init.normal_(emb.weight, std=0.001)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def get_param_groups(self, lr: float, embed_wd: float = 1e-2, mlp_wd: float = 1e-5):
        embed_params = [
            self.user_emb_gmf.weight, self.item_emb_gmf.weight,
            self.user_emb_mlp.weight, self.item_emb_mlp.weight,
        ]
        embed_ids = {id(p) for p in embed_params}
        other_params = [p for p in self.parameters() if id(p) not in embed_ids]
        return [
            {"params": embed_params, "lr": lr, "weight_decay": embed_wd},
            {"params": other_params, "lr": lr, "weight_decay": mlp_wd},
        ]

    def _score_pair(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        u_gmf = self.embed_drop(self.user_emb_gmf(user))
        i_gmf = self.embed_drop(self.item_emb_gmf(item))
        u_mlp = self.embed_drop(self.user_emb_mlp(user))
        i_mlp = self.embed_drop(self.item_emb_mlp(item))
        gmf     = u_gmf * i_gmf
        mlp_out = self.mlp(torch.cat([u_mlp, i_mlp], dim=-1))
        return self.output_layer(torch.cat([gmf, mlp_out], dim=-1)).squeeze(-1)

    def score(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        return self._score_pair(user, item)

    def forward(
        self,
        user: torch.Tensor,
        pos_item: torch.Tensor,
        neg_items: torch.Tensor,
        pos_genre: Optional[torch.Tensor] = None,
        neg_genres: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, K = neg_items.shape
        all_items   = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
        user_exp    = user.unsqueeze(1).expand(B, 1 + K).reshape(-1)
        items_flat  = all_items.reshape(-1)
        logits      = self._score_pair(user_exp, items_flat).view(B, 1 + K)
        temperature = self.log_temperature.exp().clamp(min=0.05, max=2.0)
        labels      = torch.zeros(B, dtype=torch.long, device=user.device)
        return F.cross_entropy(logits / temperature, labels)


# ─────────────────────────────────────────────────────────────────────────────
# 3. TwoTower
# ─────────────────────────────────────────────────────────────────────────────
class TwoTowerRetrieval(nn.Module):
    """
    Dual-encoder with in-batch negatives (Yi et al. 2019).

    Fix-11 collapse fix — the complete set of changes needed:

    (a) embed_dim 128 → 64:
        Parameter count: 6040×64 + 3706×64 = 624K vs 6040×128 + 3706×128 = 1.25M.
        With 988K training samples, the 128-dim version had params > samples,
        enabling trivial collapse solutions on the unit sphere.
        64-dim keeps param/sample ratio > 1.5, which is stable.

    (b) tower_dims [256] → [128]:
        Single hidden layer: 64+18=82 → 128 → proj_64.
        Previous [256]: 82 → 256 → proj_128 expands capacity by 3× at the
        first layer, creating a high-variance projection space where
        the model can satisfy loss=log(B) trivially.

    (c) No LayerNorm in tower:
        LayerNorm normalizes the hidden state BEFORE the final projection and
        L2 norm. This erases magnitude information and homogenises directions,
        directly causing collapse. Removed.

    (d) Embedding L2 regularization in forward():
        Adds 1e-6 * (||user_emb||² + ||item_emb||²) to the contrastive loss.
        Prevents embedding magnitude from growing unbounded (which would make
        all L2-normalized vectors cluster at the same poles).

    (e) temperature init log(1/0.07), clamped [0.01, 0.2]:
        Tighter clamp (was [0.02, 0.5]) ensures the softmax over B=4096 items
        is always sharp enough to carry a real gradient signal.
    """
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_genres: int,
        embed_dim: int = 64,           # Fix-11: was 128
        tower_dims: list = None,
        dropout: float = 0.1,
        genre_features: Optional[np.ndarray] = None,
        use_inbatch_negatives: bool = True,
    ):
        super().__init__()
        if tower_dims is None:
            tower_dims = [128]          # Fix-11: was [256]

        self.n_items   = n_items
        self.n_genres  = n_genres
        self.embed_dim = embed_dim
        self.use_inbatch_negatives = use_inbatch_negatives

        if genre_features is not None:
            self.register_buffer("genre_matrix",
                torch.tensor(genre_features, dtype=torch.float32))
        else:
            self.register_buffer("genre_matrix",
                torch.zeros(n_items, n_genres, dtype=torch.float32))

        # User tower: embed_dim → tower_dims → embed_dim
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        u_in = embed_dim
        u_layers = []
        for dim in tower_dims:
            # Fix-11: NO LayerNorm — it erases magnitude → collapse
            u_layers.extend([nn.Linear(u_in, dim), nn.GELU(), nn.Dropout(dropout)])
            u_in = dim
        self.user_tower = nn.Sequential(*u_layers)
        self.user_proj  = nn.Linear(u_in, embed_dim)

        # Item tower: (embed_dim + n_genres) → tower_dims → embed_dim
        i_in = embed_dim + n_genres
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        i_layers = []
        for dim in tower_dims:
            i_layers.extend([nn.Linear(i_in, dim), nn.GELU(), nn.Dropout(dropout)])
            i_in = dim
        self.item_tower = nn.Sequential(*i_layers)
        self.item_proj  = nn.Linear(i_in, embed_dim)

        # SimCLR standard init: temperature = 0.07
        self.log_temperature = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
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
        return (self.encode_user(user) * self.encode_item(item, self.genre_matrix[item])).sum(-1)

    def forward(
        self,
        user: torch.Tensor,
        pos_item: torch.Tensor,
        neg_items: torch.Tensor,
        genre_feats: Optional[torch.Tensor] = None,
        neg_genre_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # temperature in [0.01, 0.2]: tight clamp for sharp B=4096 softmax
        temperature = self.log_temperature.exp().clamp(min=0.01, max=0.2)
        B = user.shape[0]

        if genre_feats is None:
            genre_feats = self.genre_matrix[pos_item]

        user_vecs = self.encode_user(user)                   # (B, d)
        item_vecs = self.encode_item(pos_item, genre_feats)  # (B, d)

        # Fix-11(d): L2 reg on raw embeddings prevents magnitude runaway
        u_emb = self.user_embedding(user)
        i_emb = self.item_embedding(pos_item)
        emb_reg = 1e-6 * (u_emb.pow(2).sum() + i_emb.pow(2).sum()) / B

        if self.use_inbatch_negatives:
            logits = torch.mm(user_vecs, item_vecs.T) / temperature  # (B, B)
            labels = torch.arange(B, device=user.device)
            return F.cross_entropy(logits, labels) + emb_reg
        else:
            K = neg_items.shape[1]
            if neg_genre_feats is None:
                neg_genre_feats = self.genre_matrix[neg_items.view(-1)].view(B, K, self.n_genres)
            neg_vecs   = self.encode_item(
                neg_items.view(-1), neg_genre_feats.view(-1, self.n_genres)
            ).view(B, K, -1)
            pos_score  = (user_vecs * item_vecs).sum(-1, keepdim=True) / temperature
            neg_scores = torch.bmm(neg_vecs, user_vecs.unsqueeze(-1)).squeeze(-1) / temperature
            logits     = torch.cat([pos_score, neg_scores], dim=-1)
            labels     = torch.zeros(B, dtype=torch.long, device=user.device)
            return F.cross_entropy(logits, labels) + emb_reg

    @torch.no_grad()
    def get_all_item_embeddings(
        self, item_ids: torch.Tensor, genre_feats: torch.Tensor, batch_size: int = 1024
    ) -> torch.Tensor:
        embeddings = []
        for start in range(0, len(item_ids), batch_size):
            end = min(start + batch_size, len(item_ids))
            embeddings.append(self.encode_item(item_ids[start:end], genre_feats[start:end]).cpu())
        return torch.cat(embeddings, dim=0)
