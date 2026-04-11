"""
api/main.py  —  DARE-Rec FastAPI serving layer (2026-04-10, v4.2)

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

v4.2 fix (2026-04-10)  — tail-item similarity guard
  Root cause: sparse/long-tail items receive almost zero gradient signal
  during BPR training, so their embeddings barely move from random init.
  After L2 normalization, near-zero vectors all collapse to nearly the
  same unit direction, producing cosine ≈ 1.0 neighbors that are
  semantically meaningless (e.g. item_id=587 → Outside Ozona, Bloody
  Child, Tarantella — unrelated films with ~1 interaction each).

  Two-layer defence:
    1. Popularity floor  — /similar rejects items whose pre-norm embedding
       norm is below TAIL_NORM_THRESHOLD (default 1e-3). These items never
       had enough training signal to learn a meaningful direction.
       Returns HTTP 422 with a clear explanation.
    2. Neighbor dedup guard  — after scoring, any neighbor whose similarity
       rounds to 1.0000 AND whose pre-norm norm is below threshold is
       filtered out of the result list. Protects against clusters of tail
       items that all collapsed to the same unit vector.

  The raw (pre-normalization) norms are stored in lgcn_item_norms and
  lgcn_user_norms at load time so the check is O(1) per request.

v4.1 fix (2026-04-10)  — L2-normalize embeddings at load time
  dot(a,b) on unit vectors == cosine similarity ∈ [-1,1].
  Fixes item_id=0 returning similarity=3.39.

v4 fix (2026-04-10)  — tensor/numpy type impedance in similar_items()
  Convert LightGCN embeddings to numpy at load time.
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
    version="4.2.0",
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

CHECKPOINT_DIR     = os.environ.get("CHECKPOINT_DIR", "checkpoints")
N_ITEMS            = 3706
N_USERS            = 6040
EMBED_DIM          = 64
N_LAYERS           = 3

# Items whose pre-normalization embedding norm is below this threshold
# never received meaningful gradient signal during BPR training.
# Their L2-normalized vectors are numerically arbitrary and cosine
# similarity between them is meaningless.
TAIL_NORM_THRESHOLD = 1e-3

# ── Helpers ────────────────────────────────────────────────────────────

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization. dot(a, b) on result == cosine_similarity(a, b)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return (mat / norms).astype(np.float32)

# ── ModelStore ─────────────────────────────────────────────────────────

class ModelStore:
    """
    Loads DARE-Rec checkpoints at startup.

    Scoring priority:
      1. Precomputed score matrices (scores_*.npy)  — exact, O(1) lookup
      2. Live recompute from weights                — fallback

    Embedding hygiene (LightGCN):
      (a) tensor → numpy float32 at load time   (no downstream tensor ops)
      (b) raw norms stored before normalization  (tail-item guard)
      (c) L2-normalized                         (dot == cosine similarity)
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Precomputed score matrices (n_users, n_items)
        self.S_ease: Optional[np.ndarray] = None
        self.S_ials: Optional[np.ndarray] = None
        self.S_lgcn: Optional[np.ndarray] = None
        self.S_dare: Optional[np.ndarray] = None

        # Live fallback — numpy float32
        self.ease_B:        Optional[np.ndarray] = None  # (n_items, n_items)
        self.lgcn_user_emb: Optional[np.ndarray] = None  # (n_users, embed_dim)  unit vectors
        self.lgcn_item_emb: Optional[np.ndarray] = None  # (n_items, embed_dim)  unit vectors
        self.ials_U:        Optional[np.ndarray] = None  # (n_users, factors)
        self.ials_V:        Optional[np.ndarray] = None  # (n_items, factors)

        # Pre-normalization norms — used by tail-item guard
        self.lgcn_item_norms: Optional[np.ndarray] = None  # (n_items,)
        self.lgcn_user_norms: Optional[np.ndarray] = None  # (n_users,)
        self.n_tail_items: int = 0

        # Ensemble weights
        self.alpha: float = 0.9
        self.beta:  float = 0.05
        self.gamma: float = 0.05

        self.item_metadata: dict = {}
        self._load_all()

    def _p(self, filename: str) -> str:
        return os.path.join(CHECKPOINT_DIR, filename)

    def _load_all(self):
        src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
        if src_dir not in sys.path:
            sys.path.insert(0, os.path.abspath(src_dir))

        # item metadata
        meta_path = self._p("item_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                raw = json.load(f)
            self.item_metadata = {int(k): v for k, v in raw.items()}
            logger.info(f"item_metadata: {len(self.item_metadata)} titles")
        else:
            self.item_metadata = {i: f"Movie {i}" for i in range(N_ITEMS)}
            logger.warning("item_metadata.json not found — using placeholders")

        # precomputed score matrices
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

        # TemporalEASE
        ease_path = self._p("ease_B.npz")
        if os.path.exists(ease_path):
            try:
                self.ease_B = np.load(ease_path)["B"].astype(np.float32)
                logger.info(f"TemporalEASE B: {self.ease_B.shape}")
            except Exception as e:
                logger.warning(f"TemporalEASE load failed: {e}")

        # iALS
        ials_path = self._p("ials_factors.npz")
        if os.path.exists(ials_path):
            try:
                data = np.load(ials_path)
                self.ials_U = data["user_factors"].astype(np.float32)
                self.ials_V = data["item_factors"].astype(np.float32)
                logger.info(f"iALS: U={self.ials_U.shape}, V={self.ials_V.shape}")
            except Exception as e:
                logger.warning(f"iALS load failed: {e}")

        # LightGCN — tensor→numpy, store raw norms, then L2-normalize
        lgcn_path = self._p("lightgcn.pt")
        if os.path.exists(lgcn_path):
            try:
                from models import LightGCN
                lgcn = LightGCN(n_users=N_USERS, n_items=N_ITEMS,
                                embed_dim=EMBED_DIM, n_layers=N_LAYERS)
                state = torch.load(lgcn_path, map_location="cpu", weights_only=True)
                lgcn.load_state_dict(state)
                lgcn.eval()
                with torch.no_grad():
                    raw_item = lgcn.item_embedding.weight.detach().cpu().numpy().astype(np.float32)
                    raw_user = lgcn.user_embedding.weight.detach().cpu().numpy().astype(np.float32)

                # Store pre-norm norms for tail-item guard
                self.lgcn_item_norms = np.linalg.norm(raw_item, axis=1)  # (n_items,)
                self.lgcn_user_norms = np.linalg.norm(raw_user, axis=1)  # (n_users,)

                # L2-normalize: dot product == cosine similarity
                self.lgcn_item_emb = l2_normalize(raw_item)
                self.lgcn_user_emb = l2_normalize(raw_user)

                self.n_tail_items = int((self.lgcn_item_norms < TAIL_NORM_THRESHOLD).sum())
                logger.info(
                    f"LightGCN loaded: item={self.lgcn_item_emb.shape}, "
                    f"user={self.lgcn_user_emb.shape} | "
                    f"tail items (norm<{TAIL_NORM_THRESHOLD}): {self.n_tail_items}/{N_ITEMS} "
                    f"({100*self.n_tail_items/N_ITEMS:.1f}%) — /similar will return 422 for these"
                )
            except Exception as e:
                logger.warning(f"LightGCN load failed: {e}")

        # ensemble weights
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
                logger.warning(f"Ensemble weights load failed: {e}")

    # ── readiness ────────────────────────────────────────────────────

    def is_ready(self, model: str) -> bool:
        if model == "ease":  return self.S_ease is not None or self.ease_B is not None
        if model == "lgcn":  return self.S_lgcn is not None or self.lgcn_item_emb is not None
        if model == "ials":  return self.S_ials is not None or self.ials_U is not None
        if model == "dare":  return self.S_dare is not None or self.ease_B is not None
        return False

    def is_tail_item(self, item_id: int) -> bool:
        """True if the item's pre-normalization embedding norm is below threshold."""
        if self.lgcn_item_norms is None:
            return False
        return bool(self.lgcn_item_norms[item_id] < TAIL_NORM_THRESHOLD)

    # ── scoring ──────────────────────────────────────────────────────

    def _normalize(self, s: np.ndarray) -> np.ndarray:
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-8)

    def _get_scores(self, user_id: int, model: str) -> np.ndarray:
        mat = {"ease": self.S_ease, "ials": self.S_ials,
               "lgcn": self.S_lgcn, "dare": self.S_dare}.get(model)
        if mat is not None:
            return mat[user_id]

        if model == "ease":
            return self.ease_B.diagonal() if self.ease_B is not None else np.zeros(N_ITEMS, dtype=np.float32)
        if model == "ials":
            return self.ials_U[user_id] @ self.ials_V.T if self.ials_U is not None else np.zeros(N_ITEMS, dtype=np.float32)
        if model == "lgcn":
            return self.lgcn_item_emb @ self.lgcn_user_emb[user_id] if self.lgcn_user_emb is not None else np.zeros(N_ITEMS, dtype=np.float32)
        if model == "dare":
            s = np.zeros(N_ITEMS, dtype=np.float32)
            if self.ease_B is not None:    s += self.alpha * self._normalize(self.ease_B.diagonal())
            if self.ials_U is not None:    s += self.gamma * self._normalize(self.ials_U[user_id] @ self.ials_V.T)
            if self.lgcn_user_emb is not None: s += self.beta  * self._normalize(self.lgcn_item_emb @ self.lgcn_user_emb[user_id])
            return s
        return np.zeros(N_ITEMS, dtype=np.float32)

    def _top_k(self, scores: np.ndarray, k: int) -> List[dict]:
        top = np.argsort(scores)[::-1][:k]
        return [{"item_id": int(i), "score": round(float(scores[i]), 4),
                 "title": self.item_metadata.get(int(i), f"Movie {i}")} for i in top]

    def recommend(self, user_id: int, k: int, model: str) -> List[dict]:
        if model not in ("ease", "lgcn", "ials", "dare"):
            raise HTTPException(400, f"Unknown model '{model}'. Use: dare | ease | lgcn | ials")
        return self._top_k(self._get_scores(user_id, model), k)

    # ── similar items — with tail-item guard ─────────────────────────

    def similar_items(self, item_id: int, k: int) -> tuple:
        """
        Returns (items_list, method_str).

        Tail-item guard: if the query item's pre-normalization embedding norm
        is below TAIL_NORM_THRESHOLD, raises HTTP 422. These items never had
        enough interaction data for BPR to learn a meaningful embedding
        direction. After L2 normalization their vectors are arbitrary unit
        vectors, and cosine similarity to other tail items will be ≈ 1.0
        for meaningless reasons (all near-zero vectors collapse to the same
        direction after normalization).

        Neighbor guard: any neighbor that is itself a tail item is filtered
        from the result list, even if the query item passes the norm check.
        """
        if self.lgcn_item_emb is not None:
            # Tail-item guard — query item
            if self.is_tail_item(item_id):
                title = self.item_metadata.get(item_id, f"Movie {item_id}")
                norm  = float(self.lgcn_item_norms[item_id])
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Item {item_id} ('{title}') is a tail item: "
                        f"embedding norm={norm:.6f} < threshold={TAIL_NORM_THRESHOLD}. "
                        f"This item has too few interactions for LightGCN to learn a "
                        f"meaningful embedding. Use a more popular item or add "
                        f"?fallback=ials to use iALS similarity instead."
                    ),
                )

            query = self.lgcn_item_emb[item_id]    # unit vector
            sims  = self.lgcn_item_emb @ query     # cosine sims
            sims[item_id] = -np.inf                # exclude self

            # Filter tail neighbors before ranking
            if self.lgcn_item_norms is not None:
                tail_mask = self.lgcn_item_norms < TAIL_NORM_THRESHOLD
                sims[tail_mask] = -np.inf

            top_k = np.argsort(sims)[::-1][:k]
            items = [{"item_id": int(i), "similarity": round(float(sims[i]), 4),
                      "title": self.item_metadata.get(int(i), f"Movie {i}")}
                     for i in top_k if sims[i] > -np.inf]
            return items, "cosine_lightgcn"

        if self.ials_V is not None:
            q    = self.ials_V[item_id]
            norm = np.linalg.norm(self.ials_V, axis=1) + 1e-8
            sims = (self.ials_V @ q) / (norm * (np.linalg.norm(q) + 1e-8))
            sims[item_id] = -np.inf
            top_k = np.argsort(sims)[::-1][:k]
            items = [{"item_id": int(i), "similarity": round(float(sims[i]), 4),
                      "title": self.item_metadata.get(int(i), f"Movie {i}")} for i in top_k]
            return items, "cosine_ials"

        if self.ease_B is not None:
            sims = self.ease_B[:, item_id].copy()
            sims[item_id] = -np.inf
            top_k = np.argsort(sims)[::-1][:k]
            items = [{"item_id": int(i), "similarity": round(float(sims[i]), 4),
                      "title": self.item_metadata.get(int(i), f"Movie {i}")} for i in top_k]
            return items, "ease_column"

        raise HTTPException(503, "No model checkpoints loaded.")


store = ModelStore()
feedback_buffer: list = []

# ── Routes ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "4.2.0",
        "timestamp": time.time(),
        "models": {m: store.is_ready(m) for m in ("ease", "lgcn", "ials", "dare")},
        "ensemble_weights": {"alpha_ease": store.alpha, "beta_lgcn": store.beta, "gamma_ials": store.gamma},
        "titles_loaded": len(store.item_metadata) > 100,
        "score_matrices": {"ease": store.S_ease is not None, "ials": store.S_ials is not None,
                           "lgcn": store.S_lgcn is not None, "dare": store.S_dare is not None},
        "tail_item_stats": {
            "threshold": TAIL_NORM_THRESHOLD,
            "n_tail_items": store.n_tail_items,
            "pct_tail": round(100 * store.n_tail_items / N_ITEMS, 1) if store.lgcn_item_norms is not None else None,
            "note": "/similar returns 422 for tail items",
        },
    }


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(user_id: int, k: int = 10, model: str = "dare"):
    if user_id < 0 or user_id >= N_USERS:
        raise HTTPException(404, f"User {user_id} not found (valid: 0–{N_USERS-1})")
    if not store.is_ready(model):
        raise HTTPException(503, f"Model '{model}' not loaded. Run: python src/run_experiment.py")
    t0   = time.perf_counter()
    recs = store.recommend(user_id, k, model)
    ms   = (time.perf_counter() - t0) * 1000
    logger.info(f"recommend user={user_id} model={model} k={k} latency={ms:.2f}ms")
    return RecommendationResponse(user_id=user_id, recommendations=recs,
                                  model_used=model, latency_ms=round(ms, 2))


@app.get("/similar/{item_id}", response_model=SimilarItemsResponse)
def similar_items(item_id: int, k: int = 10):
    if item_id < 0 or item_id >= N_ITEMS:
        raise HTTPException(404, f"Item {item_id} not found (valid: 0–{N_ITEMS-1})")
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
