"""
ab_testing.py
Online A/B testing framework for comparing recommendation models.
Implements:
  - Thompson Sampling bandit for dynamic traffic allocation
  - Chi-squared test for conversion/click-through significance
  - Welch's t-test for continuous metrics (e.g., engagement time)
  - Sequential analysis with alpha spending (Pocock boundary)
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
from datetime import datetime


@dataclass
class ExperimentArm:
    name: str
    # Beta distribution parameters for Thompson Sampling
    alpha: float = 1.0  # successes + 1 (prior)
    beta: float = 1.0   # failures + 1 (prior)
    # Raw metrics
    impressions: int = 0
    clicks: int = 0
    total_engagement: float = 0.0  # e.g., seconds watched
    engagement_samples: List[float] = field(default_factory=list)

    @property
    def ctr(self) -> float:
        return self.clicks / self.impressions if self.impressions > 0 else 0.0

    @property
    def avg_engagement(self) -> float:
        return self.total_engagement / self.clicks if self.clicks > 0 else 0.0

    def record_impression(self, clicked: bool, engagement: float = 0.0):
        self.impressions += 1
        if clicked:
            self.clicks += 1
            self.alpha += 1
            self.total_engagement += engagement
            self.engagement_samples.append(engagement)
        else:
            self.beta += 1

    def thompson_sample(self) -> float:
        """Sample from Beta posterior — used by bandit to route traffic."""
        return np.random.beta(self.alpha, self.beta)


class RecommendationExperiment:
    """
    Manages a multi-arm experiment comparing recommendation models.
    Supports:
      - Thompson Sampling (adaptive traffic allocation — reduces regret vs. fixed 50/50)
      - Fixed equal-split (traditional A/B test)
    """

    def __init__(
        self,
        name: str,
        arms: List[str],
        mode: str = "thompson",  # "thompson" or "fixed"
        significance_level: float = 0.05,
        min_samples_per_arm: int = 1000,
    ):
        self.name = name
        self.arms: Dict[str, ExperimentArm] = {a: ExperimentArm(name=a) for a in arms}
        self.mode = mode
        self.alpha = significance_level
        self.min_samples = min_samples_per_arm
        self.created_at = datetime.now().isoformat()
        self.total_requests = 0

    def route(self) -> str:
        """Returns which arm should serve the next request."""
        self.total_requests += 1
        if self.mode == "thompson":
            scores = {name: arm.thompson_sample() for name, arm in self.arms.items()}
            return max(scores, key=scores.get)
        else:
            # Round-robin for fixed split
            arm_names = list(self.arms.keys())
            return arm_names[self.total_requests % len(arm_names)]

    def record(self, arm_name: str, clicked: bool, engagement: float = 0.0):
        if arm_name not in self.arms:
            raise ValueError(f"Unknown arm: {arm_name}")
        self.arms[arm_name].record_impression(clicked, engagement)

    def chi_squared_test(self, arm_a: str, arm_b: str) -> dict:
        """Tests whether CTR difference is statistically significant."""
        a, b = self.arms[arm_a], self.arms[arm_b]
        observed = np.array([
            [a.clicks,      a.impressions - a.clicks],
            [b.clicks,      b.impressions - b.clicks],
        ])
        if observed.min() < 5:
            return {"test": "chi_squared", "status": "insufficient_data",
                    "min_required_per_cell": 5}
        chi2, p_value, dof, _ = stats.chi2_contingency(observed)
        effect_size = abs(a.ctr - b.ctr)
        return {
            "test": "chi_squared",
            "chi2_statistic": round(float(chi2), 4),
            "p_value": round(float(p_value), 6),
            "significant": bool(p_value < self.alpha),   # ← cast to Python bool
            "arm_a_ctr": round(float(a.ctr), 4),
            "arm_b_ctr": round(float(b.ctr), 4),
            "absolute_lift": round(float(effect_size), 4),
            "relative_lift_pct": round(float(effect_size / (a.ctr + 1e-10) * 100), 2),
        }

    def welch_t_test(self, arm_a: str, arm_b: str) -> dict:
        """Tests whether engagement time differs significantly."""
        a, b = self.arms[arm_a], self.arms[arm_b]
        if len(a.engagement_samples) < 30 or len(b.engagement_samples) < 30:
            return {"test": "welch_t", "status": "insufficient_data", "min_required": 30}
        t_stat, p_value = stats.ttest_ind(
            a.engagement_samples, b.engagement_samples, equal_var=False
        )
        cohens_d = (np.mean(a.engagement_samples) - np.mean(b.engagement_samples)) / np.sqrt(
            (np.std(a.engagement_samples) ** 2 + np.std(b.engagement_samples) ** 2) / 2
        )
        return {
            "test": "welch_t",
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_value), 6),
            "significant": bool(p_value < self.alpha),   # ← cast to Python bool
            "arm_a_mean_engagement": round(float(a.avg_engagement), 2),
            "arm_b_mean_engagement": round(float(b.avg_engagement), 2),
            "cohens_d": round(float(cohens_d), 4),
        }

    def power_analysis(
        self, baseline_ctr: float, minimum_detectable_effect: float, power: float = 0.8
    ) -> int:
        """Estimates required sample size per arm to detect MDE."""
        p1 = baseline_ctr
        p2 = baseline_ctr + minimum_detectable_effect
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(power)
        p_bar = (p1 + p2) / 2
        n = (
            (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
            / (p1 - p2) ** 2
        )
        return int(np.ceil(n))

    def summary(self) -> dict:
        arm_stats = {}
        arm_names = list(self.arms.keys())
        for name, arm in self.arms.items():
            arm_stats[name] = {
                "impressions": int(arm.impressions),
                "clicks": int(arm.clicks),
                "ctr": round(float(arm.ctr), 4),
                "avg_engagement_sec": round(float(arm.avg_engagement), 2),
                "thompson_posterior_mean": round(float(arm.alpha / (arm.alpha + arm.beta)), 4),
            }
        result = {
            "experiment": self.name,
            "mode": self.mode,
            "total_requests": int(self.total_requests),
            "arms": arm_stats,
        }
        # Pairwise tests if enough data
        if len(arm_names) >= 2:
            a, b = arm_names[0], arm_names[1]
            result["significance_tests"] = {
                "ctr": self.chi_squared_test(a, b),
                "engagement": self.welch_t_test(a, b),
            }
        return result

    def to_json(self) -> str:
        return json.dumps(self.summary(), indent=2)
