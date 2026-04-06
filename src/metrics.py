"""
metrics.py
Evaluation metrics for recommendation systems.
  - Precision@K
  - Recall@K
  - NDCG@K  (Normalized Discounted Cumulative Gain)
  - Hit Rate@K
  - Mean Reciprocal Rank (MRR)
  - Coverage (catalog coverage)
  - Novelty (average inverse popularity)
"""

import numpy as np
from typing import List, Dict
from collections import defaultdict


def precision_at_k(recommended: List[int], relevant: set, k: int) -> float:
    recommended_k = recommended[:k]
    hits = sum(1 for item in recommended_k if item in relevant)
    return hits / k


def recall_at_k(recommended: List[int], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    recommended_k = recommended[:k]
    hits = sum(1 for item in recommended_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: List[int], relevant: set, k: int) -> float:
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: List[int], relevant: set, k: int) -> float:
    return float(any(item in relevant for item in recommended[:k]))


def mrr(recommended: List[int], relevant: set) -> float:
    for i, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_recommendations(
    recommendations: Dict[int, List[int]],
    ground_truth: Dict[int, set],
    k_values: List[int] = [5, 10, 20],
    item_popularity: Dict[int, int] = None,
    n_items: int = None,
) -> Dict[str, float]:
    """
    Evaluates a dict of {user_id: [ranked item list]} against ground truth.
    Returns a flat dict of metric_name -> mean value.
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
        results["MRR"].append(mrr(recs, ground_truth[user_id]))

    metrics = {name: float(np.mean(vals)) for name, vals in results.items()}

    # Coverage: fraction of catalog recommended at least once
    if n_items:
        all_recs = {item for recs in recommendations.values() for item in recs}
        metrics["Coverage"] = len(all_recs) / n_items

    # Novelty: mean negative log popularity (higher = more novel)
    if item_popularity:
        total_pop = sum(item_popularity.values())
        novelty_scores = []
        for recs in recommendations.values():
            for item in recs[:10]:
                pop = item_popularity.get(item, 1) / total_pop
                novelty_scores.append(-np.log2(pop + 1e-10))
        metrics["Novelty"] = float(np.mean(novelty_scores))

    return metrics
