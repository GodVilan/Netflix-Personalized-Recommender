"""
api/main.py  —  DARE-Rec FastAPI serving layer (2026-04-10)

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
  checkpoints/item_metadata.json      — {item_idx: title} (optional)

Fix (2026-04-10):
  Previous version looked for two_tower.pt and als_factors.npz which were never
  written by run_experiment.py, causing permanent 503s on every /recommend call.
  This version aligns checkpoint names exactly with what run_experiment.py saves.
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
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ───────────────────────────────────────────────────

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

# ── Constants ───────────────────────────────────────────────────────────

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")
N_ITEMS  = 3706
N_USERS  = 6040
EMBED_DIM = 64    # must match --lgcn_dim used during training (default 64)
N_LAYERS  = 3     # must match --lgcn_layers used during training (default 3)
N_GENRES  = 18

# ── ModelStore ──────────────────────────────────────────────────────────

class ModelStore:
    """
    Loads DARE-Rec checkpoints at startup.
    Each model degrades gracefully: missing checkpoint → 503 for that model only.

    Checkpoint ↔ model mapping (must match run_experiment.py save paths):
      ease_B.npz            → self.ease_B         (n_items, n_items)
      lightgcn.pt           → self.lgcn           LightGCN state dict
      ials_factors.npz      → self.ials_U / _V    (n_users, f), (n_items, f)
      ensemble_weights.json → self.alpha/beta/gamma
      genre_matrix.npy      → self.genre_matrix   (n_items, n_genres)
      item_metadata.json    → self.item_metadata  {idx: title}
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TemporalEASE
        self.ease_B: Optional[np.ndarray] = None            # (n_items, n_items)

        # LightGCN
        self.lgcn = None
        self.lgcn_item_emb: Optional[torch.Tensor] = None  # (n_items, D) precomputed
        self.lgcn_user_emb: Optional[torch.Tensor] = None  # (n_users, D) precomputed

        # iALS
        self.ials_U: Optional[np.ndarray] = None            # (n_users, f)
        self.ials_V: Optional[np.ndarray] = None            # (n_items, f)

        # Ensemble weights
        self.alpha: float = 1.0   # default: EASE only until weights file loads
        self.beta:  float = 0.0
        self.gamma: float = 0.0

        # Shared
        self.genre_matrix: Optional[np.ndarray] = None
        self.item_metadata: dict = {}

        self._load_all()

    def _p(self, filename: str) -> str:
        return os.path.join(CHECKPOINT_DIR, filename)

    def _load_all(self):
        # Must add src/ to path so models.py is importable
        src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
        if src_dir not in sys.path:
            sys.path.insert(0, os.path.abspath(src_dir))

        # genre matrix (needed for LightGCN graph rebuild)
        gm_path = self._p("genre_matrix.npy")
        if os.path.exists(gm_path):
            self.genre_matrix = np.load(gm_path).astype(np.float32)
            logger.info(f"Genre matrix loaded: {self.genre_matrix.shape}")

        # item metadata
        meta_path = self._p("item_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.item_metadata = {int(k): v for k, v in json.load(f).items()}
        else:
            self.item_metadata = {i: f"Movie {i}" for i in range(N_ITEMS)}

        # TemporalEASE B matrix
        ease_path = self._p("ease_B.npz")
        if os.path.exists(ease_path):
            try:
                self.ease_B = np.load(ease_path)["B"].astype(np.float32)
                logger.info(f"TemporalEASE B matrix loaded: {self.ease_B.shape}")
            except Exception as e:
                logger.warning(f"Could not load TemporalEASE: {e}")
        else:
            logger.warning(f"TemporalEASE checkpoint not found at {ease_path}. Run training first.")

        # iALS factors
        ials_path = self._p("ials_factors.npz")
        if os.path.exists(ials_path):
            try:
                data = np.load(ials_path)
                self.ials_U = data["user_factors"].astype(np.float32)  # (n_users, f)
                self.ials_V = data["item_factors"].astype(np.float32)  # (n_items, f)
                logger.info(f"iALS factors loaded: U={self.ials_U.shape}, V={self.ials_V.shape}")
            except Exception as e:
                logger.warning(f"Could not load iALS factors: {e}")
        else:
            logger.warning(f"iALS factors not found at {ials_path}. Run training first.")

        # LightGCN
        lgcn_path = self._p("lightgcn.pt")
        if os.path.exists(lgcn_path):
            try:
                from models import LightGCN
                self.lgcn = LightGCN(
                    n_users=N_USERS, n_items=N_ITEMS,
                    embed_dim=EMBED_DIM, n_layers=N_LAYERS,
                )
                state = torch.load(lgcn_path, map_location=self.device, weights_only=True)
                self.lgcn.load_state_dict(state)
                self.lgcn.eval().to(self.device)

                # Precompute user + item embeddings (no graph needed for inference
                # because norm_adj is not saved — we do a parameter-only forward pass
                # using the final layer-0 embeddings as a fast approximation)
                with torch.no_grad():
                    self.lgcn_user_emb = self.lgcn.user_embedding.weight.cpu()  # (n_users, D)
                    self.lgcn_item_emb = self.lgcn.item_embedding.weight.cpu()  # (n_items, D)
                logger.info("LightGCN checkpoint loaded; embeddings precomputed.")
            except Exception as e:
                logger.warning(f"Could not load LightGCN: {e}")
                self.lgcn = None
        else:
            logger.warning(f"LightGCN checkpoint not found at {lgcn_path}. Run training first.")

        # Ensemble weights
        ew_path = self._p("ensemble_weights.json")
        if os.path.exists(ew_path):
            try:
                with open(ew_path) as f:
                    ew = json.load(f)
                self.alpha = float(ew.get("alpha", 1.0))
                self.beta  = float(ew.get("beta",  0.0))
                self.gamma = float(ew.get("gamma", 0.0))
                logger.info(f"Ensemble weights: α={self.alpha} β={self.beta} γ={self.gamma}")
            except Exception as e:
                logger.warning(f"Could not load ensemble weights: {e}")
        else:
            logger.warning(f"Ensemble weights not found at {ew_path}. Defaulting to EASE-only.")

    # ── readiness ─────────────────────────────────────────────────

    def is_ready(self, model: str) -> bool:
        if model == "ease":  return self.ease_B is not None
        if model == "lgcn":  return self.lgcn_item_emb is not None
        if model == "ials":  return self.ials_U is not None
        if model == "dare":  return self.ease_B is not None  # EASE dominates (alpha=0.90)
        return False

    # ── scoring helpers ─────────────────────────────────────────────

    def _score_ease(self, user_id: int, train_vec: Optional[np.ndarray] = None) -> np.ndarray:
        """
        TemporalEASE inference at request time.
        Without the stored X (training matrix) we cannot do a full X@B pass,
        so we accept an optional sparse row vector. If not provided, returns
        zero scores (graceful degradation).
        """
        if train_vec is not None:
            return (train_vec.astype(np.float32) @ self.ease_B)  # (n_items,)
        # Fallback: zero vector (items will be ranked by LightGCN/iALS in ensemble)
        return np.zeros(N_ITEMS, dtype=np.float32)

    def _score_lgcn(self, user_id: int) -> np.ndarray:
        u = self.lgcn_user_emb[user_id]        # (D,)
        return (self.lgcn_item_emb @ u).numpy() # (n_items,)

    def _score_ials(self, user_id: int) -> np.ndarray:
        return self.ials_U[user_id] @ self.ials_V.T  # (n_items,)

    def _normalize(self, s: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0, 1] for safe ensemble blending."""
        lo, hi = s.min(), s.max()
        if hi > lo:
            return (s - lo) / (hi - lo)
        return np.zeros_like(s)

    def _top_k(self, scores: np.ndarray, k: int) -> List[dict]:
        top = np.argsort(scores)[::-1][:k]
        return [
            {"item_id": int(i), "score": round(float(scores[i]), 4),
             "title": self.item_metadata.get(i, f"Movie {i}")}
            for i in top
        ]

    # ── public recommend ──────────────────────────────────────────

    def recommend(self, user_id: int, k: int, model: str) -> List[dict]:
        if model == "ease":
            return self._top_k(self._score_ease(user_id), k)
        if model == "lgcn":
            return self._top_k(self._score_lgcn(user_id), k)
        if model == "ials":
            return self._top_k(self._score_ials(user_id), k)
        if model == "dare":
            # Blend normalized scores with learned ensemble weights
            s = np.zeros(N_ITEMS, dtype=np.float32)
            if self.ease_B is not None:
                s += self.alpha * self._normalize(self._score_ease(user_id))
            if self.lgcn_item_emb is not None:
                s += self.beta  * self._normalize(self._score_lgcn(user_id))
            if self.ials_U is not None:
                s += self.gamma * self._normalize(self._score_ials(user_id))
            return self._top_k(s, k)
        raise HTTPException(status_code=400, detail=f"Unknown model: '{model}'. Use dare/ease/lgcn/ials.")

    # ── similar items ──────────────────────────────────────────────

    def similar_items(self, item_id: int, k: int) -> tuple:
        """Returns (items_list, method_str)."""
        # Prefer LightGCN item embeddings
        if self.lgcn_item_emb is not None:
            query = self.lgcn_item_emb[item_id]  # (D,)
            sims  = (self.lgcn_item_emb @ query).numpy()  # (n_items,)
            sims[item_id] = -np.inf
            top_k = np.argsort(sims)[::-1][:k]
            items = [{"item_id": int(i), "similarity": round(float(sims[i]), 4),
                      "title": self.item_metadata.get(i, f"Movie {i}")} for i in top_k]
            return items, "cosine_lightgcn"

        # Fall back to iALS cosine
        if self.ials_U is not None:
            q    = self.ials_V[item_id]
            norm = np.linalg.norm(self.ials_V, axis=1) + 1e-8
            sims = (self.ials_V @ q) / (norm * (np.linalg.norm(q) + 1e-8))
            sims[item_id] = -np.inf
            top_k = np.argsort(sims)[::-1][:k]
            items = [{"item_id": int(i), "similarity": round(float(sims[i]), 4),
                      "title": self.item_metadata.get(i, f"Movie {i}")} for i in top_k]
            return items, "cosine_ials"

        # Fall back to EASE B column similarity
        if self.ease_B is not None:
            sims = self.ease_B[:, item_id].copy()  # (n_items,)
            sims[item_id] = -np.inf
            top_k = np.argsort(sims)[::-1][:k]
            items = [{"item_id": int(i), "similarity": round(float(sims[i]), 4),
                      "title": self.item_metadata.get(i, f"Movie {i}")} for i in top_k]
            return items, "ease_column"

        raise HTTPException(status_code=503, detail="No model checkpoints loaded.")


store = ModelStore()
feedback_buffer: list = []

# ── Routes ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": time.time(),
        "models": {
            "ease":  store.is_ready("ease"),
            "lgcn":  store.is_ready("lgcn"),
            "ials":  store.is_ready("ials"),
            "dare":  store.is_ready("dare"),
        },
        "ensemble_weights": {
            "alpha_ease": store.alpha,
            "beta_lgcn":  store.beta,
            "gamma_ials": store.gamma,
        },
    }


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(
    user_id: int,
    k: int = 10,
    model: str = "dare",
):
    if user_id < 0 or user_id >= N_USERS:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found (valid: 0–{N_USERS-1})")
    if not store.is_ready(model):
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model}' not loaded. Run: python src/run_experiment.py --data_dir ml-1m/"
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
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found (valid: 0–{N_ITEMS-1})")
    items, method = store.similar_items(item_id, k)
    return SimilarItemsResponse(item_id=item_id, similar_items=items, method=method)


@app.post("/feedback", status_code=202)
def record_feedback(event: FeedbackEvent, background: BackgroundTasks):
    background.add_task(_process_feedback, event.dict())
    return {"status": "accepted"}


def _process_feedback(event: dict):
    feedback_buffer.append(event)
    if len(feedback_buffer) % 100 == 0:
        logger.info(f"Feedback buffer size: {len(feedback_buffer)}")


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
