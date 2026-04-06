"""Unit tests for A/B testing framework."""
import sys; sys.path.insert(0, "src")
from ab_testing import RecommendationExperiment, ExperimentArm
import numpy as np


def test_arm_ctr():
    arm = ExperimentArm(name="test")
    arm.record_impression(True, 300.0)
    arm.record_impression(False)
    arm.record_impression(True, 250.0)
    assert arm.ctr == 2/3
    assert arm.impressions == 3
    assert arm.clicks == 2


def test_thompson_routing():
    exp = RecommendationExperiment("test_exp", ["control", "treatment"], mode="thompson")
    np.random.seed(42)
    routes = [exp.route() for _ in range(100)]
    assert all(r in ["control", "treatment"] for r in routes)


def test_chi_squared_insufficient():
    exp = RecommendationExperiment("test_exp", ["a", "b"], mode="fixed")
    result = exp.chi_squared_test("a", "b")
    assert result["status"] == "insufficient_data"


def test_power_analysis():
    exp = RecommendationExperiment("test", ["a", "b"])
    n = exp.power_analysis(baseline_ctr=0.12, minimum_detectable_effect=0.02)
    assert n > 1000  # needs at least 1000 samples to detect 2% MDE


def test_summary():
    exp = RecommendationExperiment("summ_test", ["als", "two_tower"])
    for _ in range(50):
        exp.record("als", True, 400)
        exp.record("two_tower", True, 450)
    s = exp.summary()
    assert "arms" in s
    assert s["arms"]["als"]["clicks"] == 50
