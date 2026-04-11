"""
api/main.py  —  DARE-Rec FastAPI serving layer (2026-04-10, v4.4)

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

v4.4 fix (2026-04-10)  — load propagated LightGCN embeddings for /similar
  Root cause of cosine≈1.0 on /similar (confirmed via embedding diagnostics):

  LightGCN's actual item representations are:
    E_final = mean(E0, A_hat@E0, A_hat^2@E0, A_hat^3@E0)
  where A_hat is the normalized user-item bipartite adjacency matrix.

  api/main.py was loading only lightgcn.pt (which contains E0 — the base
  learnable embedding weight) and calling l2_normalize(E0). A_hat is not
  stored in the checkpoint; the API has no interaction matrix at serve time
  and cannot reconstruct it. So propagate() was never called, and all
  similarity was computed on raw E0.

  Diagnostics confirmed the collapse:
    - median pairwise cosine on E0 = 0.84  (should be ~0.05-0.15)
    - 42.5% of random items have cosine > 0.999 with some other item
    - p90 pairwise cosine = 0.9985
  This is NOT undertrained weights — BPR did run. E0 simply lacks the
  neighborhood-aggregation signal that makes LightGCN representations
  discriminative.

  Fix (two parts):
    1. run_experiment.py now calls lgcn.propagate() immediately after
       training (while norm_adj is live), extracts E_final, L2-normalizes,
       and saves to checkpoints/lgcn_item_emb.npy + lgcn_user_emb.npy.
    2. This file (v4.4) loads those .npy files directly at startup.
       Falls back to E0 with a loud WARNING if they don’t exist (i.e.,
       the checkpoint was produced by run_experiment.py v3 or earlier).

  The /health endpoint now reports lgcn_emb_source so the operator can
  confirm which path is active.

v4.3 fix (2026-04-10)  — calibrate TAIL_NORM_THRESHOLD to 0.05
  Empirical norm distribution: min=0.0049, p5=0.0167, median=0.1595.
  Previous 1e-3 was below the minimum, catching 0 items.
  Note: tail norm guard is kept but is secondary to the propagation fix.

v4.2 fix (2026-04-10)  — two-layer tail-item guard in /similar
v4.1 fix (2026-04-10)  — L2-normalize embeddings at load time
v4   fix (2026-04-10)  — tensor/numpy type impedance
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
    version="4.4.0",
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

# Tail-item guard: items whose E0 norm is below threshold are still filtered
# as neighbors even after the propagation fix, since propagation amplifies
# but does not fully correct near-zero base embeddings.
TAIL_NORM_THRESHOLD = 0.05

# ── Helpers ────────────────────────────────────────────────────────────

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization. dot(a, b) on result == cosine_similarity(a, b)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return (mat / norms).astype(np.float32)

# ── ModelStore ─────────────────────────────────────────────────────────

class ModelStore:
    """
    Loads DARE-Rec checkpoints at startup.

    LightGCN embedding loading priority (v4.4):
      1. lgcn_item_emb.npy + lgcn_user_emb.npy  [PREFERRED]
         Propagated E_final = mean(E0..EL), L2-normalized.
         Produced by run_experiment.py v4+.
         These are the real LightGCN representations.

      2. lightgcn.pt state_dict (E0 only)  [FALLBACK, DEGRADED]
         Raw base embeddings before graph propagation.
         Logs a WARNING. /similar results will be low quality.
         Only used if .npy files are absent (old checkpoint).

    Scoring priority for /recommend:
      1. Precomputed score matrices (scores_*.npy)  — exact, O(1) lookup
      2. Live recompute from weights                — fallback
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Precomputed score matrices (n_users, n_items)
        self.S_ease: Optional[np.ndarray] = None
        self.S_ials: Optional[np.ndarray] = None
        self.S_lgcn: Optional[np.ndarray] = None
        self.S_dare: Optional[np.ndarray] = None

        # Live fallback — numpy float32
        self.ease_B:        Optional[np.ndarray] = None
        self.lgcn_user_emb: Optional[np.ndarray] = None  # (n_users, d) unit vectors
        self.lgcn_item_emb: Optional[np.ndarray] = None  # (n_items, d) unit vectors
        self.ials_U:        Optional[np.ndarray] = None
        self.ials_V:        Optional[np.ndarray] = None

        # E0 norms — used by tail-item neighbor filter
        self.lgcn_item_norms: Optional[np.ndarray] = None
        self.n_tail_items: int = 0

        # Source flag for /health diagnostics
        self.lgcn_emb_source: str = "not_loaded"

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

        # ──────────────────────────────────────────────────────────────
        # LightGCN embeddings — prefer propagated .npy, fall back to E0
        # ──────────────────────────────────────────────────────────────
        item_emb_path = self._p("lgcn_item_emb.npy")
        user_emb_path = self._p("lgcn_user_emb.npy")

        if os.path.exists(item_emb_path) and os.path.exists(user_emb_path):
            # PRIMARY PATH: propagated + L2-normalized embeddings from run_experiment.py v4
            self.lgcn_item_emb = np.load(item_emb_path).astype(np.float32)
            self.lgcn_user_emb = np.load(user_emb_path).astype(np.float32)
            self.lgcn_emb_source = "propagated"
            logger.info(
                f"LightGCN propagated embeddings loaded: "
                f"item={self.lgcn_item_emb.shape}, user={self.lgcn_user_emb.shape} "
                f"[source: lgcn_item_emb.npy — CORRECT]"
            )
            # E0 norms for tail-item neighbor filter
            # Load E0 weights just to get norms (not used for similarity)
            lgcn_pt = self._p("lightgcn.pt")
            if os.path.exists(lgcn_pt):
                try:
                    from models import LightGCN
                    lgcn = LightGCN(n_users=N_USERS, n_items=N_ITEMS,
                                    embed_dim=EMBED_DIM, n_layers=N_LAYERS)
                    state = torch.load(lgcn_pt, map_location="cpu", weights_only=True)
                    lgcn.load_state_dict(state)
                    with torch.no_grad():
                        raw_item = lgcn.item_embedding.weight.detach().cpu().numpy()
                    self.lgcn_item_norms = np.linalg.norm(raw_item, axis=1)
                    self.n_tail_items = int((self.lgcn_item_norms < TAIL_NORM_THRESHOLD).sum())
                    logger.info(f"E0 norms loaded for tail filter: {self.n_tail_items} tail items")
                except Exception as e:
                    logger.warning(f"E0 norm load failed (tail filter disabled): {e}")
        else:
            # FALLBACK PATH: E0 only — /similar quality will be degraded
            logger.warning(
                "lgcn_item_emb.npy not found. Falling back to raw E0 embeddings. "
                "Run run_experiment.py v4+ to generate propagated embeddings. "
                "/similar results will be low quality until then."
            )
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
                    self.lgcn_item_norms = np.linalg.norm(raw_item, axis=1)
                    self.lgcn_item_emb = l2_normalize(raw_item)
                    self.lgcn_user_emb = l2_normalize(raw_user)
                    self.n_tail_items = int((self.lgcn_item_norms < TAIL_NORM_THRESHOLD).sum())
                    self.lgcn_emb_source = "e0_fallback"
                    logger.warning(
                        f"LightGCN E0 fallback loaded: item={self.lgcn_item_emb.shape} "
                        f"[DEGRADED — no graph propagation]"
                    )
                except Exception as e:
                    logger.warning(f"LightGCN fallback load failed: {e}")

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
            if self.ease_B is not None:        s += self.alpha * self._normalize(self.ease_B.diagonal())
            if self.ials_U is not None:        s += self.gamma * self._normalize(self.ials_U[user_id] @ self.ials_V.T)
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

    # ── similar items ─────────────────────────────────────────────────────

    def similar_items(self, item_id: int, k: int) -> tuple:
        if self.lgcn_item_emb is not None:
            # Tail-item guard on query item
            if self.is_tail_item(item_id):
                title = self.item_metadata.get(item_id, f"Movie {item_id}")
                norm  = float(self.lgcn_item_norms[item_id])
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Item {item_id} ('{title}') is a tail item: "
                        f"E0 embedding norm={norm:.6f} < threshold={TAIL_NORM_THRESHOLD}. "
                        f"Too few interactions for BPR to learn a meaningful embedding."
                    ),
                )

            query = self.lgcn_item_emb[item_id]
            sims  = self.lgcn_item_emb @ query
            sims[item_id] = -np.inf

            # Filter E0-tail neighbors
            if self.lgcn_item_norms is not None:
                sims[self.lgcn_item_norms < TAIL_NORM_THRESHOLD] = -np.inf

            top_k = np.argsort(sims)[::-1][:k]
            items = [{"item_id": int(i), "similarity": round(float(sims[i]), 4),
                      "title": self.item_metadata.get(int(i), f"Movie {i}")}
                     for i in top_k if sims[i] > -np.inf]
            method = f"cosine_lightgcn_{self.lgcn_emb_source}"
            return items, method

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
        "version": "4.4.0",
        "timestamp": time.time(),
        "models": {m: store.is_ready(m) for m in ("ease", "lgcn", "ials", "dare")},
        "ensemble_weights": {"alpha_ease": store.alpha, "beta_lgcn": store.beta, "gamma_ials": store.gamma},
        "titles_loaded": len(store.item_metadata) > 100,
        "score_matrices": {"ease": store.S_ease is not None, "ials": store.S_ials is not None,
                           "lgcn": store.S_lgcn is not None, "dare": store.S_dare is not None},
        "lgcn_emb_source": store.lgcn_emb_source,
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
