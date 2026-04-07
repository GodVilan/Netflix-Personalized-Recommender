"""
api/main.py
FastAPI serving layer for Netflix-style recommendation inference.
Endpoints:
  GET  /recommend/{user_id}   – top-K personalized recommendations
  GET  /similar/{item_id}     – item-to-item similarity
  POST /feedback              – record implicit feedback (click/watch)
  GET  /health                – health check
  GET  /metrics               – offline evaluation metrics (cached)

Fix (2026-04-06) — Bug 6:
  ModelStore now loads real TwoTower and ALS checkpoints from disk instead of
  returning random Dirichlet scores. The MockModelStore was serving random
  recommendations even though checkpoints/ncf.pt and checkpoints/two_tower.pt
  were saved at the end of run_experiment.py.

  Loading strategy:
    - TwoTower: loads state dict, precomputes all item embeddings via
      get_all_item_embeddings(), stores them on CPU. At request time,
      encodes the user and does a dot-product scan. For production,
      replace the scan with a FAISS index for sub-millisecond ANN retrieval.
    - ALS: loads numpy arrays from checkpoints/als_factors.npz (added to the
      save step in run_experiment.py — see below). At request time, scores
      all items with a single matmul.
    - Graceful fallback: if checkpoints are absent (first run, CI), the server
      starts with a warning and returns 503 until models are available.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import torch
import json
import os
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("netflix-rec-api")

app = FastAPI(
    title="Netflix-Style Recommendation API",
    description="Two-Tower + ALS Collaborative Filtering recommendation engine",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ──────────────────────────────────────

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

# ── Model store ──────────────────────────────────────────

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")
N_ITEMS  = 3706   # ML-1M encoded item count
N_USERS  = 6040
N_GENRES = 18
EMBED_DIM = 128


class ModelStore:
    """
    Loads real model checkpoints.  Falls back to None if unavailable.
    Two-Tower: precomputes item embeddings once at startup.
    ALS: loads user/item factor matrices.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.two_tower = None
        self.als_user_factors: Optional[np.ndarray] = None
        self.als_item_factors: Optional[np.ndarray] = None
        self.item_embeddings: Optional[torch.Tensor] = None   # precomputed TwoTower item vecs
        self.item_metadata = self._load_metadata()
        self._load_models()

    def _load_metadata(self) -> dict:
        meta_path = os.path.join(CHECKPOINT_DIR, "item_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                return {int(k): v for k, v in json.load(f).items()}
        return {i: f"Movie {i}" for i in range(N_ITEMS)}

    def _load_models(self):
        # --- Two-Tower ---
        tt_path = os.path.join(CHECKPOINT_DIR, "two_tower.pt")
        gm_path = os.path.join(CHECKPOINT_DIR, "genre_matrix.npy")
        if os.path.exists(tt_path):
            try:
                from models import TwoTowerRetrieval
                genre_features = np.load(gm_path) if os.path.exists(gm_path) else None
                self.two_tower = TwoTowerRetrieval(
                    n_users=N_USERS, n_items=N_ITEMS,
                    n_genres=N_GENRES, embed_dim=EMBED_DIM,
                    genre_features=genre_features,
                )
                state = torch.load(tt_path, map_location=self.device, weights_only=True)
                self.two_tower.load_state_dict(state)
                self.two_tower.eval().to(self.device)

                # Precompute all item embeddings (done once at startup).
                # In production replace this scan with a FAISS IndexFlatIP.
                item_ids = torch.arange(N_ITEMS, dtype=torch.long, device=self.device)
                gm = self.two_tower.genre_matrix
                with torch.no_grad():
                    self.item_embeddings = self.two_tower.get_all_item_embeddings(
                        item_ids, gm, batch_size=1024
                    )   # (N_ITEMS, EMBED_DIM) on CPU
                logger.info("Two-Tower checkpoint loaded; item embeddings precomputed.")
            except Exception as e:
                logger.warning(f"Could not load Two-Tower checkpoint: {e}")
                self.two_tower = None
        else:
            logger.warning(f"Two-Tower checkpoint not found at {tt_path}. Run training first.")

        # --- ALS ---
        als_path = os.path.join(CHECKPOINT_DIR, "als_factors.npz")
        if os.path.exists(als_path):
            try:
                data = np.load(als_path)
                self.als_user_factors = data["user_factors"]
                self.als_item_factors = data["item_factors"]
                logger.info("ALS factors loaded.")
            except Exception as e:
                logger.warning(f"Could not load ALS factors: {e}")
        else:
            logger.warning(f"ALS factors not found at {als_path}. Run training first.")

    def _is_ready(self, model: str) -> bool:
        if model == "two_tower":
            return self.two_tower is not None and self.item_embeddings is not None
        if model == "als":
            return self.als_user_factors is not None
        return False

    def two_tower_recommend(self, user_id: int, k: int, exclude_seen: bool) -> List[dict]:
        with torch.no_grad():
            user_t = torch.tensor([user_id], dtype=torch.long, device=self.device)
            user_vec = self.two_tower.encode_user(user_t).cpu()   # (1, D)
        scores = (self.item_embeddings @ user_vec.T).squeeze(-1).numpy()   # (N_ITEMS,)
        top_k  = np.argsort(scores)[::-1][:k]
        return [{"item_id": int(i), "score": round(float(scores[i]), 4),
                 "title": self.item_metadata.get(i, f"Movie {i}")} for i in top_k]

    def als_recommend(self, user_id: int, k: int) -> List[dict]:
        scores = self.als_user_factors[user_id] @ self.als_item_factors.T
        top_k  = np.argsort(scores)[::-1][:k]
        return [{"item_id": int(i), "score": round(float(scores[i]), 4),
                 "title": self.item_metadata.get(i, f"Movie {i}")} for i in top_k]

    def similar_items(self, item_id: int, k: int = 10) -> List[dict]:
        if self.item_embeddings is not None:
            query = self.item_embeddings[item_id]   # (D,)
            sims  = (self.item_embeddings @ query).numpy()
            sims[item_id] = -np.inf
            top_k = np.argsort(sims)[::-1][:k]
            return [{"item_id": int(i), "similarity": round(float(sims[i]), 4),
                     "title": self.item_metadata.get(i, f"Movie {i}")} for i in top_k]
        if self.als_item_factors is not None:
            query = self.als_item_factors[item_id]
            norms = np.linalg.norm(self.als_item_factors, axis=1) + 1e-8
            sims  = (self.als_item_factors @ query) / (norms * (np.linalg.norm(query) + 1e-8))
            sims[item_id] = -np.inf
            top_k = np.argsort(sims)[::-1][:k]
            return [{"item_id": int(i), "similarity": round(float(sims[i]), 4),
                     "title": self.item_metadata.get(i, f"Movie {i}")} for i in top_k]
        raise HTTPException(status_code=503, detail="No model checkpoints loaded.")


store = ModelStore()
feedback_buffer = []

# ── Routes ───────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": time.time(),
        "two_tower_ready": store._is_ready("two_tower"),
        "als_ready": store._is_ready("als"),
    }


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(
    user_id: int,
    k: int = 10,
    model: str = "two_tower",
    exclude_seen: bool = True,
):
    if user_id < 0 or user_id >= N_USERS:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    if not store._is_ready(model):
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model}' not loaded. Run training first and ensure checkpoints exist."
        )

    t0 = time.perf_counter()
    if model == "two_tower":
        recs = store.two_tower_recommend(user_id, k, exclude_seen)
    elif model == "als":
        recs = store.als_recommend(user_id, k)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")
    latency = (time.perf_counter() - t0) * 1000

    logger.info(f"user={user_id} model={model} k={k} latency={latency:.2f}ms")
    return RecommendationResponse(
        user_id=user_id,
        recommendations=recs,
        model_used=model,
        latency_ms=round(latency, 2),
    )


@app.get("/similar/{item_id}", response_model=SimilarItemsResponse)
def similar_items(item_id: int, k: int = 10):
    if item_id < 0 or item_id >= N_ITEMS:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    similar = store.similar_items(item_id, k)
    method  = "cosine_two_tower" if store._is_ready("two_tower") else "cosine_als"
    return SimilarItemsResponse(item_id=item_id, similar_items=similar, method=method)


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
    """Returns cached offline evaluation metrics from last model evaluation run."""
    metrics_path = os.path.join(CHECKPOINT_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return {
        "warning": "No metrics.json found. Run training to populate.",
        "model": "two_tower_v1",
        "dataset": "MovieLens-1M",
    }
