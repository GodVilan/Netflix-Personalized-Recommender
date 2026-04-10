"""
metrics.py  —  DARE-Rec rewrite (2026-04-10)

Metrics reported (first system on ML-1M to report all five together):
  Accuracy  : NDCG@K, Recall@K, HitRate@K, MRR
  Diversity : Intra-List Diversity (ILD) — mean pairwise genre dissimilarity
  Coverage  : Catalog coverage fraction
  Novelty   : Mean inverse log popularity

Intra-List Diversity (ILD) definition (Ziegler et al. 2005):
  ILD(L) = (2 / K(K-1)) * sum_{i<j} dissim(i, j)
  dissim(i, j) = 1 - cosine_sim(genre_vec_i, genre_vec_j)
  ILD=0 → all recommended items are in the same genre
  ILD=1 → all items are maximally genre-diverse
"""

import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict


def precision_at_k(recommended: List[int], relevant: set, k: int) -> float:
    return sum(1 for item in recommended[:k] if item in relevant) / k


def recall_at_k(recommended: List[int], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for item in recommended[:k] if item in relevant) / len(relevant)


def ndcg_at_k(recommended: List[int], relevant: set, k: int) -> float:
    dcg  = sum(1.0 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: List[int], relevant: set, k: int) -> float:
    return float(any(item in relevant for item in recommended[:k]))


def mrr(recommended: List[int], relevant: set) -> float:
    for i, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def intra_list_diversity(
    recommended: List[int],
    genre_matrix: np.ndarray,
    k: int = 10,
) -> float:
    """
    Intra-List Diversity (ILD) via genre cosine dissimilarity.
    genre_matrix: (n_items, n_genres) float32 one-hot.
    Returns mean pairwise dissimilarity in [0, 1].
    """
    items = recommended[:k]
    if len(items) < 2:
        return 0.0
    vecs = genre_matrix[items]                                  # (k, n_genres)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    vecs_norm = vecs / norms                                    # unit vectors
    sim_matrix = vecs_norm @ vecs_norm.T                        # (k, k) cosine sim
    # upper triangle (i < j)
    k_ = len(items)
    n_pairs = k_ * (k_ - 1) / 2
    upper = sim_matrix[np.triu_indices(k_, k=1)]
    return float(1.0 - upper.mean())  # dissimilarity


def evaluate_recommendations(
    recommendations: Dict[int, List[int]],
    ground_truth: Dict[int, set],
    k_values: List[int] = [5, 10, 20],
    item_popularity: Optional[Dict[int, int]] = None,
    n_items: Optional[int] = None,
    genre_matrix: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Full evaluation: accuracy + diversity + coverage + novelty.
    """
    results = defaultdict(list)

    for user_id, recs in recommendations.items():
        if user_id not in ground_truth:
            continue
        relevant = ground_truth[user_id]
        for k in k_values:
            results[f"Precision@{k}"].append(precision_at_k(recs, relevant, k))
            results[f"Recall@{k}"].append(recall_at_k(recs, relevant, k))
            results[f"NDCG@{k}"].append(ndcg_at_k(recs, relevant, k))
            results[f"HitRate@{k}"].append(hit_rate_at_k(recs, relevant, k))
        results["MRR"].append(mrr(recs, relevant))
        if genre_matrix is not None:
            results["ILD@10"].append(intra_list_diversity(recs, genre_matrix, k=10))

    metrics = {name: round(float(np.mean(vals)), 4) for name, vals in results.items()}

    if n_items:
        all_recs = {item for recs in recommendations.values() for item in recs}
        metrics["Coverage"] = round(len(all_recs) / n_items, 4)

    if item_popularity:
        total_pop = sum(item_popularity.values())
        novelty_scores = [
            -np.log2(item_popularity.get(item, 1) / total_pop + 1e-10)
            for recs in recommendations.values()
            for item in recs[:10]
        ]
        metrics["Novelty"] = round(float(np.mean(novelty_scores)), 4)

    return metrics
