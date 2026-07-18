"""
Consistency Module for Gower Layer
==================================
Checks consistency of hypotheses using similarity measures.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from scipy.stats import zscore


@dataclass
class ConsistencyConfig:
    """Configuration for consistency checking."""

    threshold: float = 0.7
    method: str = "mean"  # mean, median, percentile
    percentile: float = 0.25
    robust: bool = True
    min_references: int = 3


class ConsistencyChecker:
    """
    Checks consistency of hypotheses against reference set.
    """

    def __init__(self, config: Optional[ConsistencyConfig] = None):
        self.config = config or ConsistencyConfig()

    def compute_consistency(
        self,
        hypothesis: np.ndarray,
        reference_set: np.ndarray,
        similarity_matrix: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute consistency of a hypothesis with reference set.
        """
        if similarity_matrix is not None:
            # Use precomputed similarity
            n_ref = len(reference_set)
            similarities = []
            for ref in reference_set:
                similarities.append(similarity_matrix[hypothesis, ref])
        else:
            # Compute similarities on the fly
            similarities = self._compute_similarities(hypothesis, reference_set)

        if len(similarities) < self.config.min_references:
            return 0.0

        # Compute consistency score
        if self.config.method == "mean":
            score = np.mean(similarities)
        elif self.config.method == "median":
            score = np.median(similarities)
        elif self.config.method == "percentile":
            score = np.percentile(similarities, self.config.percentile * 100)
        else:
            score = np.mean(similarities)

        # Apply robust scaling if enabled
        if self.config.robust:
            score = self._robust_scale(score, similarities)

        return float(np.clip(score, 0.0, 1.0))

    def compute_consistency_matrix(
        self, hypotheses: np.ndarray, similarity_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Compute consistency matrix for all hypotheses.
        """
        n = len(hypotheses)
        consistency_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    consistency_matrix[i, j] = self.compute_consistency(i, [j], similarity_matrix)

        return consistency_matrix

    def _compute_similarities(self, hypothesis: int, references: np.ndarray) -> List[float]:
        """Compute similarities between hypothesis and references."""
        # Placeholder: Implement actual similarity computation
        return [0.5 + 0.5 * np.random.random() for _ in references]

    def _robust_scale(self, score: float, similarities: List[float]) -> float:
        """Apply robust scaling to consistency score."""
        mean = np.mean(similarities)
        std = np.std(similarities)
        if std > 0:
            z = (score - mean) / std
            return 1.0 / (1.0 + np.exp(-z))
        return score

    def get_consistent_subset(
        self,
        hypotheses: List[str],
        similarity_matrix: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Get subset of consistent hypotheses.
        """
        if threshold is None:
            threshold = self.config.threshold

        n = len(hypotheses)
        consistency_scores = np.zeros(n)

        for i in range(n):
            # Compute consistency against all other hypotheses
            others = [j for j in range(n) if j != i]
            score = self.compute_consistency(i, others, similarity_matrix)
            consistency_scores[i] = score

        # Select hypotheses above threshold
        mask = consistency_scores >= threshold
        consistent_hypotheses = [h for h, m in zip(hypotheses, mask) if m]

        return consistent_hypotheses, consistency_scores

    def detect_outliers(
        self, hypotheses: List[str], similarity_matrix: np.ndarray, n_sigma: float = 2.0
    ) -> Tuple[List[str], List[str]]:
        """
        Detect outlier hypotheses using statistical methods.
        """
        n = len(hypotheses)
        consistency_scores = np.zeros(n)

        for i in range(n):
            others = [j for j in range(n) if j != i]
            score = self.compute_consistency(i, others, similarity_matrix)
            consistency_scores[i] = score

        # Use z-score for outlier detection
        z_scores = zscore(consistency_scores)
        outlier_indices = np.where(np.abs(z_scores) > n_sigma)[0]

        outliers = [hypotheses[i] for i in outlier_indices]
        inliers = [h for i, h in enumerate(hypotheses) if i not in outlier_indices]

        return inliers, outliers


class ConsistencyValidator:
    """
    Validates hypotheses for consistency over time.
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history = []

    def update(self, consistency_scores: np.ndarray):
        """
        Update validation history.
        """
        self.history.append(consistency_scores)
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def get_trend(self, hypothesis_idx: int) -> float:
        """
        Get trend of consistency for a hypothesis.
        """
        if len(self.history) < 2:
            return 0.0

        values = [h[hypothesis_idx] for h in self.history]
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope

    def is_stable(self, hypothesis_idx: int, threshold: float = 0.1) -> bool:
        """
        Check if consistency is stable over time.
        """
        if len(self.history) < self.window_size // 2:
            return True

        values = [h[hypothesis_idx] for h in self.history[-self.window_size :]]
        std = np.std(values)
        return std < threshold

    def get_validation_metrics(self) -> Dict[str, Any]:
        """
        Get validation metrics.
        """
        if not self.history:
            return {}

        metrics = {
            "window_size": self.window_size,
            "history_length": len(self.history),
            "avg_consistency": np.mean(self.history[-1]),
            "std_consistency": np.std(self.history[-1]),
            "min_consistency": np.min(self.history[-1]),
            "max_consistency": np.max(self.history[-1]),
        }

        # Add trend for each hypothesis
        n_hypotheses = len(self.history[-1])
        metrics["trends"] = [self.get_trend(i) for i in range(n_hypotheses)]

        # Add stability
        metrics["stability"] = [self.is_stable(i) for i in range(n_hypotheses)]

        return metrics
