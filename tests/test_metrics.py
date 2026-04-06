"""Unit tests for recommendation metrics."""
import sys; sys.path.insert(0, "src")
from metrics import precision_at_k, recall_at_k, ndcg_at_k, hit_rate_at_k, mrr


def test_precision_at_k():
    recs = [1, 2, 3, 4, 5]
    relevant = {1, 3, 5}
    assert precision_at_k(recs, relevant, 5) == 3/5
    assert precision_at_k(recs, relevant, 1) == 1.0


def test_recall_at_k():
    recs = [1, 2, 3, 4, 5]
    relevant = {1, 3, 6}
    assert recall_at_k(recs, relevant, 5) == 2/3


def test_ndcg_at_k():
    recs = [1, 2, 3]
    relevant = {1}
    assert ndcg_at_k(recs, relevant, 3) == 1.0   # hit at position 1 → ideal
    recs2 = [2, 1, 3]
    assert ndcg_at_k(recs2, relevant, 3) < 1.0    # hit at position 2 → subideal


def test_hit_rate():
    recs = [1, 2, 3]
    assert hit_rate_at_k(recs, {1}, 3) == 1.0
    assert hit_rate_at_k(recs, {99}, 3) == 0.0


def test_mrr():
    recs = [1, 2, 3, 4]
    assert mrr(recs, {3}) == 1/3
    assert mrr(recs, {99}) == 0.0


def test_empty_relevant():
    recs = [1, 2, 3]
    assert recall_at_k(recs, set(), 3) == 0.0
