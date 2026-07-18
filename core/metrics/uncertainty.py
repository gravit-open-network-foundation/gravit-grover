"""
Uncertainty Module
===================
Computes epistemic uncertainty measures for consensus.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from scipy.stats import entropy


class UncertaintyComputer:
    """
    Computes various uncertainty measures.
    """

    @staticmethod
    def epistemic_uncertainty(distributions: List[np.ndarray]) -> Dict[str, float]:
        """
        Compute epistemic uncertainty from multiple distributions.

        Epistemic uncertainty = disagreement between distributions.
        """
        if not distributions:
            return {"uncertainty": 0.0, "max_disagreement": 0.0}

        n_distributions = len(distributions)
        n_hypotheses = len(distributions[0])

        # Compute average distribution
        avg = np.mean(distributions, axis=0)
        avg = avg / np.sum(avg)

        # Compute pairwise disagreements
        disagreements = []
        for i in range(n_distributions):
            for j in range(i + 1, n_distributions):
                # KL divergence as disagreement measure
                kl = entropy(distributions[i], distributions[j])
                disagreements.append(kl)

        # Also compute variance across distributions
        variance = np.var(distributions, axis=0)

        return {
            "uncertainty": np.mean(disagreements) if disagreements else 0.0,
            "max_disagreement": max(disagreements) if disagreements else 0.0,
            "avg_variance": np.mean(variance),
            "max_variance": np.max(variance),
            "n_samples": n_distributions,
            "n_hypotheses": n_hypotheses,
        }

    @staticmethod
    def aleatoric_uncertainty(distribution: np.ndarray) -> float:
        """
        Compute aleatoric uncertainty (inherent randomness).

        Aleatoric uncertainty = entropy of distribution.
        """
        distribution = np.clip(distribution, 1e-10, 1.0)
        distribution = distribution / np.sum(distribution)
        return entropy(distribution)

    @staticmethod
    def total_uncertainty(distributions: List[np.ndarray]) -> float:
        """
        Compute total uncertainty = aleatoric + epistemic.
        """
        epistemic = UncertaintyComputer.epistemic_uncertainty(distributions)
        aleatoric = np.mean([UncertaintyComputer.aleatoric_uncertainty(d) for d in distributions])

        return epistemic["uncertainty"] + aleatoric

    @staticmethod
    def disagreement_ratio(distributions: List[np.ndarray]) -> float:
        """
        Compute disagreement ratio.

        Measures how much distributions disagree with each other.
        """
        if len(distributions) < 2:
            return 0.0

        # Compute variance across distributions
        variance = np.var(distributions, axis=0)
        avg_variance = np.mean(variance)

        # Normalize by maximum possible variance
        max_variance = 0.25  # For binary case
        return min(1.0, avg_variance / max_variance)

    @staticmethod
    def consensus_stability(distributions: List[np.ndarray], window: int = 5) -> float:
        """
        Compute consensus stability over time.
        """
        if len(distributions) < window + 1:
            return 0.0

        # Compute divergence between consecutive distributions
        divergences = []
        for i in range(len(distributions) - 1):
            div = entropy(distributions[i], distributions[i + 1])
            divergences.append(div)

        # Compute average divergence
        avg_divergence = np.mean(divergences[-window:])

        # Convert to stability (inverse of divergence)
        return 1.0 / (1.0 + avg_divergence)

    @staticmethod
    def confidence_interval(distribution: np.ndarray, confidence: float = 0.95) -> Dict[str, float]:
        """
        Compute confidence interval for the maximum hypothesis.
        """
        distribution = np.clip(distribution, 0, 1)
        distribution = distribution / np.sum(distribution)

        # Sort in descending order
        sorted_dist = np.sort(distribution)[::-1]

        # Compute cumulative probability
        cum_prob = np.cumsum(sorted_dist)

        # Find index where cumulative probability exceeds confidence
        n_hypotheses = np.where(cum_prob >= confidence)[0]
        if len(n_hypotheses) == 0:
            n = len(distribution)
        else:
            n = n_hypotheses[0] + 1

        return {
            "n_hypotheses": n,
            "confidence": confidence,
            "prob_mass": cum_prob[n - 1] if n <= len(cum_prob) else 1.0,
            "remaining_uncertainty": 1.0 - cum_prob[n - 1] if n <= len(cum_prob) else 0.0,
        }


class UncertaintyMonitor:
    """
    Monitors uncertainty over time.
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.distribution_history = []
        self.uncertainty_history = []

    def update(self, distributions: List[np.ndarray]):
        """
        Update uncertainty history.
        """
        self.distribution_history.append(distributions)

        # Compute uncertainty
        uncertainty = UncertaintyComputer.epistemic_uncertainty(distributions)
        self.uncertainty_history.append(uncertainty)

        # Keep history bounded
        if len(self.distribution_history) > self.window_size:
            self.distribution_history.pop(0)
            self.uncertainty_history.pop(0)

    def get_current_uncertainty(self) -> Dict[str, float]:
        """
        Get current uncertainty.
        """
        if not self.uncertainty_history:
            return {"uncertainty": 0.0}
        return self.uncertainty_history[-1]

    def get_average_uncertainty(self) -> float:
        """
        Get average uncertainty.
        """
        if not self.uncertainty_history:
            return 0.0
        return np.mean([u["uncertainty"] for u in self.uncertainty_history])

    def get_uncertainty_trend(self) -> float:
        """
        Get uncertainty trend.
        """
        if len(self.uncertainty_history) < 2:
            return 0.0

        values = [u["uncertainty"] for u in self.uncertainty_history]
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope

    def get_consensus_quality(self) -> float:
        """
        Get consensus quality (inverse of uncertainty).
        """
        uncertainty = self.get_current_uncertainty()
        return 1.0 / (1.0 + uncertainty.get("uncertainty", 0.0))

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get uncertainty statistics.
        """
        return {
            "current_uncertainty": self.get_current_uncertainty(),
            "average_uncertainty": self.get_average_uncertainty(),
            "uncertainty_trend": self.get_uncertainty_trend(),
            "consensus_quality": self.get_consensus_quality(),
            "history_length": len(self.distribution_history),
            "window_size": self.window_size,
        }
