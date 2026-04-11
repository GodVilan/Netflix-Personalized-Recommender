"""
api/main.py  —  DARE-Rec FastAPI serving layer (2026-04-10, v4.1)

Endpoints:
  GET  /recommend/{user_id}  — top-K personalized recommendations
  GET  /similar/{item_id}    — item-to-item similarity
  POST /feedback             — record implicit feedback (click/watch)
  GET  /health               — model readiness check
  GET  /metrics              — offline evaluation metrics (cached from results.json)

Models served:
  dare  — DAREnsemble (TemporalEASE + LightGCN + iALS blended scores) [default]
  ease  — TemporalEASE only
  lgcn  — LightGCN only
  ials  — ImplicitALS only

Checkpoints expected (all written by src/run_experiment.py):
  checkpoints/ease_B.npz              — TemporalEASE B matrix
  checkpoints/lightgcn.pt             — LightGCN state dict
  checkpoints/ials_factors.npz        — iALS user/item factors
  checkpoints/ensemble_weights.json   — DAREnsemble alpha/beta/gamma
  checkpoints/genre_matrix.npy        — (n_items, n_genres) float32
  checkpoints/item_metadata.json      — {item_idx: title}
  checkpoints/scores_ease.npy         — (n_users, n_items) precomputed EASE scores
  checkpoints/scores_ials.npy         — (n_users, n_items) precomputed iALS scores
  checkpoints/scores_lgcn.npy         — (n_users, n_items) precomputed LightGCN scores
  checkpoints/scores_dare.npy         — (n_users, n_items) precomputed DARE ensemble scores

v4.1 fix (2026-04-10):
  /similar/0 (Toy Story) returned similarity=3.39 — impossible for cosine similarity.
  Root cause: raw dot product on un-normalized embeddings. Popular items (Toy Story,
  Star Wars) receive the most gradient updates during BPR training, resulting in
  large embedding norms. The dot product A·B is not cosine similarity unless both
  vectors are unit-length — it equals ||A|| * ||B|| * cos(θ), so high-norm items
  dominate purely by magnitude, not semantic proximity.
  Fix: L2-normalize lgcn_item_emb and lgcn_user_emb immediately after numpy
  conversion at load time. Dot product on unit vectors = true cosine similarity,
  guaranteed in [-1, 1]. Normalization is applied once at startup — no overhead
  at request time. No retraining needed.

v4 fix (2026-04-10):
  /similar/{item_id} returned 500 due to 3 stacked tensor/numpy type bugs.
  Fix: convert embeddings to numpy at load time. See commit 108543b for details.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import torch
import json
import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dare-rec-api")

app = FastAPI(
    title="DARE-Rec Recommendation API",
    description="TemporalEASE + LightGCN + iALS ensemble recommendation engine",
    version="4.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ────────────────────────────────────────────────────

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[dict]
    model_used: str
    latency_ms: float

class FeedbackEvent(BaseModel):
    user_id: int
    item_id: int
    event_type: str
    rating: Optional[float] = None
    watch_duration_sec: Optional[float] = None
    experiment_arm: Optional[str] = None

class SimilarItemsResponse(BaseModel):
    item_id: int
    similar_items: List[dict]
    method: str

# ── Constants ──────────────────────────────────────────────────────────

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")
N_ITEMS   = 3706
N_USERS   = 6040
EMBED_DIM = 64
N_LAYERS  = 3

# ── Helpers ────────────────────────────────────────────────────────────

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization. After this, dot(a, b) == cosine_similarity(a, b)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return (mat / norms).astype(np.float32)

# ── ModelStore ─────────────────────────────────────────────────────────

class ModelStore:
    """
    Loads DARE-Rec checkpoints at startup.

    Priority for scoring:
      1. Precomputed score matrices (scores_*.npy) — exact offline scores, fastest
      2. Live recompute from model weights — fallback if matrices absent

    All embedding tensors are:
      (a) converted to numpy float32 at load time  — no tensor/numpy impedance
      (b) L2-normalized at load time               — dot product == cosine similarity
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Precomputed score matrices (n_users, n_items)
        self.S_ease: Optional[np.ndarray] = None
        self.S_ials: Optional[np.ndarray] = None
        self.S_lgcn: Optional[np.ndarray] = None
        self.S_dare: Optional[np.ndarray] = None

        # Live fallback — numpy float32, L2-normalized where applicable
        self.ease_B:        Optional[np.ndarray] = None  # (n_items, n_items)
        self.lgcn_user_emb: Optional[np.ndarray] = None  # (n_users, embed_dim) — unit vectors
        self.lgcn_item_emb: Optional[np.ndarray] = None  # (n_items, embed_dim) — unit vectors
        self.ials_U:        Optional[np.ndarray] = None  # (n_users, factors)
        self.ials_V:        Optional[np.ndarray] = None  # (n_items, factors)

        # Ensemble weights
        self.alpha: float = 0.9
        self.beta:  float = 0.05
        self.gamma: float = 0.05

        # Metadata
        self.item_metadata: dict = {}

        self._load_all()

    def _p(self, filename: str) -> str:
        return os.path.join(CHECKPOINT_DIR, filename)

    def _load_all(self):
        src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
        if src_dir not in sys.path:
            sys.path.insert(0, os.path.abspath(src_dir))

        # ── item metadata ──────────────────────────────────────────────
        meta_path = self._p("item_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                raw = json.load(f)
            self.item_metadata = {int(k): v for k, v in raw.items()}
            sample = {k: self.item_metadata[k] for k in list(self.item_metadata)[:3]}
            logger.info(f"item_metadata loaded: {len(self.item_metadata)} titles. Sample: {sample}")
        else:
            self.item_metadata = {i: f"Movie {i}" for i in range(N_ITEMS)}
            logger.warning(f"item_metadata.json not found — using placeholders.")

        # ── precomputed score matrices (preferred) ─────────────────────
        for attr, fname in [
            ("S_ease", "scores_ease.npy"),
            ("S_ials", "scores_ials.npy"),
            ("S_lgcn", "scores_lgcn.npy"),
            ("S_dare", "scores_dare.npy"),
        ]:
            path = self._p(fname)
            if os.path.exists(path):
                mat = np.load(path)
                setattr(self, attr, mat.astype(np.float32))
                logger.info(f"Loaded {fname}: {mat.shape}")
            else:
                logger.info(f"{fname} not found — will use live recompute fallback.")

        # ── live fallback: TemporalEASE B matrix ───────────────────────
        ease_path = self._p("ease_B.npz")
        if os.path.exists(ease_path):
            try:
                self.ease_B = np.load(ease_path)["B"].astype(np.float32)
                logger.info(f"TemporalEASE B matrix loaded: {self.ease_B.shape}")
            except Exception as e:
                logger.warning(f"Could not load TemporalEASE: {e}")

        # ── live fallback: iALS factors ────────────────────────────────
        ials_path = self._p("ials_factors.npz")
        if os.path.exists(ials_path):
            try:
                data = np.load(ials_path)
                self.ials_U = data["user_factors"].astype(np.float32)
                self.ials_V = data["item_factors"].astype(np.float32)
                logger.info(f"iALS factors loaded: U={self.ials_U.shape}, V={self.ials_V.shape}")
            except Exception as e:
                logger.warning(f"Could not load iALS: {e}")

        # ── live fallback: LightGCN embeddings ────────────────────────
        # Step 1: tensor → numpy (no more tensor/numpy impedance downstream)
        # Step 2: L2-normalize rows so dot(a,b) == cosine_similarity(a,b)
        #         Without this, popular items (Toy Story, Star Wars) have large
        #         embedding norms from many gradient updates, causing dot products
        #         of 3.39+ instead of the correct [-1, 1] cosine range.
        lgcn_path = self._p("lightgcn.pt")
        if os.path.exists(lgcn_path):
            try:
                from models import LightGCN
                lgcn = LightGCN(
                    n_users=N_USERS, n_items=N_ITEMS,
                    embed_dim=EMBED_DIM, n_layers=N_LAYERS,
                )
                state = torch.load(lgcn_path, map_location="cpu", weights_only=True)
                lgcn.load_state_dict(state)
                lgcn.eval()
                with torch.no_grad():
                    raw_item = lgcn.item_embedding.weight.detach().cpu().numpy().astype(np.float32)
                    raw_user = lgcn.user_embedding.weight.detach().cpu().numpy().astype(np.float32)

                # L2-normalize: after this, all dot products are true cosine similarities
                self.lgcn_item_emb = l2_normalize(raw_item)
                self.lgcn_user_emb = l2_normalize(raw_user)

                # Sanity check: norms should all be ~1.0
                item_norms = np.linalg.norm(self.lgcn_item_emb, axis=1)
                logger.info(
                    f"LightGCN embeddings loaded + L2-normalized: "
                    f"item={self.lgcn_item_emb.shape}, user={self.lgcn_user_emb.shape} | "
                    f"item norm mean={item_norms.mean():.4f} min={item_norms.min():.4f} max={item_norms.max():.4f}"
                )
            except Exception as e:
                logger.warning(f"Could not load LightGCN: {e}")

        # ── ensemble weights ───────────────────────────────────────────
        ew_path = self._p("ensemble_weights.json")
        if os.path.exists(ew_path):
            try:
                with open(ew_path) as f:
                    ew = json.load(f)
                self.alpha = float(ew.get("alpha", 0.9))
                self.beta  = float(ew.get("beta",  0.05))
                self.gamma = float(ew.get("gamma", 0.05))
                logger.info(f"Ensemble weights: α={self.alpha} β={self.beta} γ={self.gamma}")
            except Exception as e:
                logger.warning(f"Could not load ensemble weights: {e}")

    # ── readiness ──────────────────────────────────────────────────────

    def is_ready(self, model: str) -> bool:
        if model == "ease":
            return self.S_ease is not None or self.ease_B is not None
        if model == "lgcn":
            return self.S_lgcn is not None or self.lgcn_item_emb is not None
        if model == "ials":
            return self.S_ials is not None or self.ials_U is not None
        if model == "dare":
            return self.S_dare is not None or self.ease_B is not None
        return False

    # ── scoring helpers ────────────────────────────────────────────────

    def _normalize(self, s: np.ndarray) -> np.ndarray:
        lo, hi = s.min(), s.max()
        if hi > lo:
            return (s - lo) / (hi - lo)
        return np.zeros_like(s)

    def _get_scores(self, user_id: int, model: str) -> np.ndarray:
        """Return (n_items,) float32 score vector for user_id under model."""
        # 1. Precomputed matrix — fastest, exact
        mat = {"ease": self.S_ease, "ials": self.S_ials,
               "lgcn": self.S_lgcn, "dare": self.S_dare}.get(model)
        if mat is not None:
            return mat[user_id]

        # 2. Live recompute fallback (pure numpy)
        if model == "ease":
            if self.ease_B is not None:
                return self.ease_B.diagonal()
            return np.zeros(N_ITEMS, dtype=np.float32)

        if model == "ials":
            if self.ials_U is not None:
                return self.ials_U[user_id] @ self.ials_V.T
            return np.zeros(N_ITEMS, dtype=np.float32)

        if model == "lgcn":
            if self.lgcn_user_emb is not None:
                # Both are L2-normalized → dot product == cosine similarity
                return self.lgcn_item_emb @ self.lgcn_user_emb[user_id]
            return np.zeros(N_ITEMS, dtype=np.float32)

        if model == "dare":
            s = np.zeros(N_ITEMS, dtype=np.float32)
            if self.ease_B is not None:
                s += self.alpha * self._normalize(self.ease_B.diagonal())
            if self.ials_U is not None:
                s += self.gamma * self._normalize(self.ials_U[user_id] @ self.ials_V.T)
            if self.lgcn_user_emb is not None:
                s += self.beta * self._normalize(
                    self.lgcn_item_emb @ self.lgcn_user_emb[user_id]
                )
            return s

        return np.zeros(N_ITEMS, dtype=np.float32)

    def _top_k(self, scores: np.ndarray, k: int) -> List[dict]:
        top = np.argsort(scores)[::-1][:k]
        return [
            {
                "item_id": int(i),
                "score":   round(float(scores[i]), 4),
                "title":   self.item_metadata.get(int(i), f"Movie {i}"),
            }
            for i in top
        ]

    # ── public recommend ───────────────────────────────────────────────

    def recommend(self, user_id: int, k: int, model: str) -> List[dict]:
        if model not in ("ease", "lgcn", "ials", "dare"):
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model '{model}'. Use: dare | ease | lgcn | ials",
            )
        scores = self._get_scores(user_id, model)
        return self._top_k(scores, k)

    # ── similar items ──────────────────────────────────────────────────

    def similar_items(self, item_id: int, k: int) -> tuple:
        """
        Returns (items_list, method_str).

        Priority: LightGCN embeddings → iALS item factors → EASE B column.
        LightGCN embeddings are L2-normalized at load time, so
        dot(a, b) here is true cosine similarity in [-1, 1].
        """
        if self.lgcn_item_emb is not None:
            query         = self.lgcn_item_emb[item_id]   # unit vector (embed_dim,)
            sims          = self.lgcn_item_emb @ query     # cosine sims (n_items,)
            sims[item_id] = -np.inf                        # exclude self
            top_k         = np.argsort(sims)[::-1][:k]
            items = [
                {
                    "item_id":    int(i),
                    "similarity": round(float(sims[i]), 4),
                    "title":      self.item_metadata.get(int(i), f"Movie {i}"),
                }
                for i in top_k
            ]
            return items, "cosine_lightgcn"

        if self.ials_V is not None:
            q    = self.ials_V[item_id]
            norm = np.linalg.norm(self.ials_V, axis=1) + 1e-8
            sims = (self.ials_V @ q) / (norm * (np.linalg.norm(q) + 1e-8))
            sims[item_id] = -np.inf
            top_k = np.argsort(sims)[::-1][:k]
            items = [
                {
                    "item_id":    int(i),
                    "similarity": round(float(sims[i]), 4),
                    "title":      self.item_metadata.get(int(i), f"Movie {i}"),
                }
                for i in top_k
            ]
            return items, "cosine_ials"

        if self.ease_B is not None:
            sims          = self.ease_B[:, item_id].copy()
            sims[item_id] = -np.inf
            top_k         = np.argsort(sims)[::-1][:k]
            items = [
                {
                    "item_id":    int(i),
                    "similarity": round(float(sims[i]), 4),
                    "title":      self.item_metadata.get(int(i), f"Movie {i}"),
                }
                for i in top_k
            ]
            return items, "ease_column"

        raise HTTPException(status_code=503, detail="No model checkpoints loaded.")


store = ModelStore()
feedback_buffer: list = []

# ── Routes ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": time.time(),
        "models": {
            "ease": store.is_ready("ease"),
            "lgcn": store.is_ready("lgcn"),
            "ials": store.is_ready("ials"),
            "dare": store.is_ready("dare"),
        },
        "ensemble_weights": {
            "alpha_ease": store.alpha,
            "beta_lgcn":  store.beta,
            "gamma_ials": store.gamma,
        },
        "titles_loaded": (
            len(store.item_metadata) > 0
            and store.item_metadata.get(0, "Movie 0") != "Movie 0"
        ),
        "score_matrices": {
            "ease": store.S_ease is not None,
            "ials": store.S_ials is not None,
            "lgcn": store.S_lgcn is not None,
            "dare": store.S_dare is not None,
        },
    }


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(user_id: int, k: int = 10, model: str = "dare"):
    if user_id < 0 or user_id >= N_USERS:
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} not found (valid: 0–{N_USERS - 1})",
        )
    if not store.is_ready(model):
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model}' not loaded. Run: python src/run_experiment.py --data_dir ml-1m/",
        )

    t0   = time.perf_counter()
    recs = store.recommend(user_id, k, model)
    ms   = (time.perf_counter() - t0) * 1000

    logger.info(f"user={user_id} model={model} k={k} latency={ms:.2f}ms")
    return RecommendationResponse(
        user_id=user_id,
        recommendations=recs,
        model_used=model,
        latency_ms=round(ms, 2),
    )


@app.get("/similar/{item_id}", response_model=SimilarItemsResponse)
def similar_items(item_id: int, k: int = 10):
    if item_id < 0 or item_id >= N_ITEMS:
        raise HTTPException(
            status_code=404,
            detail=f"Item {item_id} not found (valid: 0–{N_ITEMS - 1})",
        )
    items, method = store.similar_items(item_id, k)
    return SimilarItemsResponse(item_id=item_id, similar_items=items, method=method)


@app.post("/feedback", status_code=202)
def record_feedback(event: FeedbackEvent, background: BackgroundTasks):
    background.add_task(_process_feedback, event.dict())
    return {"status": "accepted"}


def _process_feedback(event: dict):
    feedback_buffer.append(event)
    if len(feedback_buffer) % 100 == 0:
        logger.info(f"Feedback buffer: {len(feedback_buffer)} events")


@app.get("/metrics")
def get_metrics():
    """Returns cached offline metrics from the last run_experiment.py run."""
    for path in ["results.json", os.path.join(CHECKPOINT_DIR, "metrics.json")]:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return {
        "warning": "No results.json found. Run: python src/run_experiment.py --data_dir ml-1m/",
        "published_baselines": {
            "EASE^R (Anelli 2022)": {"NDCG@10": 0.336},
            "iALS  (Anelli 2022)": {"NDCG@10": 0.306},
            "NeuMF (Anelli 2022)": {"NDCG@10": 0.277},
        },
    }
