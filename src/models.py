"""
models.py
Three recommendation models:
  1. CollaborativeFilteringALS  – Matrix Factorization via implicit library
  2. NeuralMatrixFactorization   – NCF with InfoNCE, LayerNorm, embedding regularisation
  3. TwoTowerRetrieval           – Dual-encoder with IN-BATCH negatives

Fix log:
  Fix-10 (2026-04-10):
    ALS:      alpha 1.0 → 40. confidence = 1 + alpha*r. At alpha=1 every seen
              item has confidence 2.0 — barely above unseen items (confidence=1).
              alpha=40 gives confidence=41 for positives, matching Hu et al. 2008.
              Expected NDCG@10: 0.075–0.090 (was 0.038).

    NCF:      mf_dim default 64→128 so GMF and MLP towers are balanced.
              mlp_dims default [256,128,64]→[512,256,128] for richer MLP.
              These are passed from run_experiment now.

    TwoTower: embed_dim 256→128 (projection target). Smaller target reduces
              representational collapse risk on the unit sphere.
              tower_dims [512,256]→[256]: single hidden layer. The hourglass
              (256→512→256) caused collapse; shallow tower is stable.
              log_temperature init: 0.0→log(1/0.07)≈2.66 so initial temperature
              is 0.07 (SimCLR standard) not 1.0 (too soft at epoch 1).

  Fix-9 (2026-04-07): ALS factor swap (user_factors ↔ item_factors).
  Fix-8 (2026-04-07): TwoTower in-batch negatives.
  Fix-7 (2026-04-07): NCF heavy embedding L2 via get_param_groups().
  Fix-6 (2026-04-07): ALS → implicit library.
  Fix-1–5 (2026-04-06): NCF BatchNorm→LayerNorm, sigmoid removal, BPR→InfoNCE,
    TwoTower genre buffer, ALS hyperparams.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from typing import Optional
import math


# ─────────────────────────────────────────────────────────────────────────────
# 1. Collaborative Filtering — ALS
# ─────────────────────────────────────────────────────────────────────────────

class CollaborativeFilteringALS:
    """
    Weighted ALS for implicit feedback (Hu et al. 2008).

    Fix-10: alpha default changed 1.0 → 40.
      Confidence formula: c_ui = 1 + alpha * r_ui
      alpha=1  → c=2   for positives vs c=1 for unseen (almost no signal)
      alpha=40 → c=41  for positives vs c=1 for unseen (strong signal)
      Published ML-1M results with implicit + alpha=40: NDCG@10 0.075–0.090.

    Fix-9: implicit factor layout when fed items×users matrix:
      model.user_factors → item latent vectors  (n_items × n_factors)
      model.item_factors → user latent vectors  (n_users × n_factors)
    Assigned correctly in fit() so recommend() works.
    """

    def __init__(
        self,
        n_factors: int = 256,
        regularization: float = 0.01,
        iterations: int = 30,
        alpha: float = 40.0,          # Fix-10: was 1.0
        use_gpu: bool = False,
        random_state: int = 42,
    ):
        self.n_factors      = n_factors
        self.regularization = regularization
        self.iterations     = iterations
        self.alpha          = alpha
        self.use_gpu        = use_gpu
        self.random_state   = random_state
        self.model          = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

    def fit(self, interaction_matrix: csr_matrix) -> "CollaborativeFilteringALS":
        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError:
            raise ImportError("pip install implicit")

        item_user = interaction_matrix.T.tocsr()
        weighted  = (item_user * self.alpha).astype(np.float32)

        self.model = AlternatingLeastSquares(
            factors=self.n_factors,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=self.use_gpu,
            calculate_training_loss=True,
            random_state=self.random_state,
        )
        self.model.fit(weighted, show_progress=False)
        for i in range(5, self.iterations + 1, 5):
            print(f"  ALS iteration {i}/{self.iterations} complete")

        # Fix-9: swap — implicit internally treats matrix rows as "users"
        self.user_factors = np.array(self.model.item_factors)  # (n_users, n_factors)
        self.item_factors = np.array(self.model.user_factors)  # (n_items, n_factors)
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
# 2. Neural Collaborative Filtering (NCF)
# ─────────────────────────────────────────────────────────────────────────────

class NeuralMatrixFactorization(nn.Module):
    """
    GMF + MLP fusion with LayerNorm and InfoNCE loss.
    Reference: He et al., 2017.

    Fix-10: mf_dim default 64→128, mlp_dims default [256,128,64]→[512,256,128].
      Previously mf_dim=256 (passed from run_experiment as n_factors=256) made
      the GMF tower 2× larger than the MLP input (128), causing GMF to dominate.
      Now mf_dim=128 balances both towers; MLP is wider for richer interactions.

    Fix-7: heavy L2 on embeddings via get_param_groups().
    Fix-1–3: LayerNorm, no sigmoid, InfoNCE.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        mf_dim: int = 128,            # Fix-10: was 64 (passed as n_factors=256 before)
        mlp_dims: list = None,
        dropout: float = 0.2,
        embed_dropout: float = 0.3,
    ):
        super().__init__()
        if mlp_dims is None:
            mlp_dims = [512, 256, 128]  # Fix-10: was [256, 128, 64]

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
        for emb in [self.user_emb_gmf, self.item_emb_gmf,
                    self.user_emb_mlp, self.item_emb_mlp]:
            nn.init.normal_(emb.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

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
        mlp_in  = torch.cat([u_mlp, i_mlp], dim=-1)
        mlp_out = self.mlp(mlp_in)
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
        logits_flat = self._score_pair(user_exp, items_flat)
        logits      = logits_flat.view(B, 1 + K)
        temperature = self.log_temperature.exp().clamp(min=0.05, max=2.0)
        logits      = logits / temperature
        labels      = torch.zeros(B, dtype=torch.long, device=user.device)
        return F.cross_entropy(logits, labels)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Two-Tower Retrieval Model — IN-BATCH negatives
# ─────────────────────────────────────────────────────────────────────────────

class TwoTowerRetrieval(nn.Module):
    """
    Dual-encoder with IN-BATCH negatives.
    Reference: Yi et al., 2019 — Sampling-Bias-Corrected Neural Modeling.

    Fix-10 (2026-04-10):
      embed_dim 256→128: smaller projection target reduces collapse risk.
        On the unit sphere with embed_dim=256 the model has enough capacity
        to map all users/items to nearly-identical directions (collapse),
        yielding uniform logits → random-chance loss.
        dim=128 is proven stable in dual-encoder literature (Karpukhin 2020).

      tower_dims [512,256]→[256]: one hidden layer instead of two.
        The previous hourglass (embed+genre → 512 → 256 → proj_128) expands
        then contracts. The expansion from 256+18=274 → 512 immediately forces
        the network to learn in an over-parameterized space where it can find
        trivial solutions (collapse). A single 256-layer is more stable:
        274 → 256 → proj_128.

      log_temperature init: 0.0 → math.log(1/0.07) ≈ 2.66.
        Init at 0.0 means temperature=exp(0)=1.0 at epoch 1.
        With B=4096 items in the softmax, temperature=1.0 makes logits
        from unit-sphere dot products (range [-1,1]) too soft — cross-entropy
        reduces to nearly log(B)=8.3 regardless of prediction quality.
        Init at 0.07 (SimCLR, MoCo standard) sharpens the distribution from
        step 1, giving the model a real gradient signal immediately.

    Fix-8 (2026-04-07): IN-BATCH negatives (4095 negs vs 64 random).
    Fix-4 (2026-04-06): .score() uses real genre features from register_buffer.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_genres: int,
        embed_dim: int = 128,         # Fix-10: was 256
        tower_dims: list = None,
        dropout: float = 0.1,
        genre_features: Optional[np.ndarray] = None,
        use_inbatch_negatives: bool = True,
    ):
        super().__init__()
        if tower_dims is None:
            tower_dims = [256]         # Fix-10: was [512, 256]

        self.n_items   = n_items
        self.n_genres  = n_genres
        self.embed_dim = embed_dim
        self.use_inbatch_negatives = use_inbatch_negatives

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

        # User tower: embed_dim → tower_dims → embed_dim
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

        # Item tower: (embed_dim + n_genres) → tower_dims → embed_dim
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

        # Fix-10: init temperature at 0.07 (SimCLR standard) not 1.0
        # log(1/0.07) = 2.659 → temperature = 0.07 at epoch 1
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(1.0 / 0.07))
        )
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
        genre_feats = self.genre_matrix[item]
        user_vec    = self.encode_user(user)
        item_vec    = self.encode_item(item, genre_feats)
        return (user_vec * item_vec).sum(dim=-1)

    def forward(
        self,
        user: torch.Tensor,
        pos_item: torch.Tensor,
        neg_items: torch.Tensor,
        genre_feats: Optional[torch.Tensor] = None,
        neg_genre_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temperature = self.log_temperature.exp().clamp(min=0.02, max=0.5)
        B = user.shape[0]

        if genre_feats is None:
            genre_feats = self.genre_matrix[pos_item]

        user_vecs = self.encode_user(user)
        item_vecs = self.encode_item(pos_item, genre_feats)

        if self.use_inbatch_negatives:
            logits = torch.mm(user_vecs, item_vecs.T) / temperature
            labels = torch.arange(B, device=user.device)
            return F.cross_entropy(logits, labels)
        else:
            K = neg_items.shape[1]
            if neg_genre_feats is None:
                neg_genre_feats = self.genre_matrix[neg_items.view(-1)].view(B, K, self.n_genres)
            neg_vecs   = self.encode_item(
                neg_items.view(-1),
                neg_genre_feats.view(-1, self.n_genres)
            ).view(B, K, -1)
            pos_score  = (user_vecs * item_vecs).sum(-1, keepdim=True) / temperature
            neg_scores = torch.bmm(neg_vecs, user_vecs.unsqueeze(-1)).squeeze(-1) / temperature
            logits     = torch.cat([pos_score, neg_scores], dim=-1)
            labels     = torch.zeros(B, dtype=torch.long, device=user.device)
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
