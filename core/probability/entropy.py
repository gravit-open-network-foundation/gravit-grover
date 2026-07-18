"""
Entropy Module
==============
Computes entropy-related metrics for probability distributions.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from scipy.special import entr


class EntropyComputer:
    """
    Computes various entropy measures.
    """

    @staticmethod
    def shannon_entropy(probabilities: np.ndarray) -> float:
        """
        Compute Shannon entropy.

        H(p) = -Σ_i p_i * log(p_i)
        """
        probs = np.clip(probabilities, 1e-10, 1.0)
        return -np.sum(probs * np.log(probs))

    @staticmethod
    def renyi_entropy(probabilities: np.ndarray, alpha: float = 2.0) -> float:
        """
        Compute Renyi entropy.

        H_α(p) = 1/(1-α) * log(Σ_i p_i^α)
        """
        probs = np.clip(probabilities, 1e-10, 1.0)
        if alpha == 1:
            return EntropyComputer.shannon_entropy(probs)

        sum_pow = np.sum(probs**alpha)
        if sum_pow == 0:
            return 0.0
        return (1.0 / (1.0 - alpha)) * np.log(sum_pow)

    @staticmethod
    def tsallis_entropy(probabilities: np.ndarray, q: float = 2.0) -> float:
        """
        Compute Tsallis entropy.

        S_q(p) = (1/(q-1)) * (1 - Σ_i p_i^q)
        """
        probs = np.clip(probabilities, 1e-10, 1.0)
        sum_pow = np.sum(probs**q)
        return (1.0 / (q - 1.0)) * (1.0 - sum_pow)

    @staticmethod
    def sample_entropy(probabilities: np.ndarray, n_bins: int = 10) -> float:
        """
        Compute entropy from samples.
        """
        # Create histogram
        hist, _ = np.histogram(probabilities, bins=n_bins, range=(0, 1))
        hist = hist.astype(float) / len(probabilities)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist))

    @staticmethod
    def conditional_entropy(joint: np.ndarray, marginal: np.ndarray) -> float:
        """
        Compute conditional entropy.

        H(Y|X) = -Σ_{x,y} p(x,y) * log(p(y|x))
        """
        joint = np.clip(joint, 1e-10, 1.0)
        marginal = np.clip(marginal, 1e-10, 1.0)

        # p(y|x) = p(x,y) / p(x)
        cond = joint / marginal[:, np.newaxis]
        cond = np.clip(cond, 1e-10, 1.0)

        return -np.sum(joint * np.log(cond))

    @staticmethod
    def mutual_information(
        joint: np.ndarray, marginal_x: np.ndarray, marginal_y: np.ndarray
    ) -> float:
        """
        Compute mutual information.

        I(X;Y) = Σ_{x,y} p(x,y) * log(p(x,y) / (p(x)*p(y)))
        """
        joint = np.clip(joint, 1e-10, 1.0)
        marginal_x = np.clip(marginal_x, 1e-10, 1.0)
        marginal_y = np.clip(marginal_y, 1e-10, 1.0)

        # Compute mutual information
        mi = 0.0
        for i in range(len(marginal_x)):
            for j in range(len(marginal_y)):
                if joint[i, j] > 0:
                    ratio = joint[i, j] / (marginal_x[i] * marginal_y[j])
                    if ratio > 0:
                        mi += joint[i, j] * np.log(ratio)

        return mi


class EntropyMonitor:
    """
    Monitors entropy evolution over time.
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history = []
        self.entropy_values = []

    def update(self, probabilities: np.ndarray):
        """
        Update entropy history.
        """
        entropy = EntropyComputer.shannon_entropy(probabilities)
        self.entropy_values.append(entropy)
        self.history.append(probabilities.copy())

        if len(self.history) > self.window_size:
            self.history.pop(0)
            self.entropy_values.pop(0)

    def get_current_entropy(self) -> float:
        """
        Get current entropy.
        """
        if not self.entropy_values:
            return 0.0
        return self.entropy_values[-1]

    def get_average_entropy(self) -> float:
        """
        Get average entropy over history.
        """
        if not self.entropy_values:
            return 0.0
        return np.mean(self.entropy_values)

    def get_entropy_trend(self) -> float:
        """
        Get entropy trend (slope).
        """
        if len(self.entropy_values) < 2:
            return 0.0

        x = np.arange(len(self.entropy_values))
        slope = np.polyfit(x, self.entropy_values, 1)[0]
        return slope

    def get_entropy_variation(self) -> float:
        """
        Get entropy variation (standard deviation).
        """
        if len(self.entropy_values) < 2:
            return 0.0
        return np.std(self.entropy_values)

    def get_convergence_score(self) -> float:
        """
        Get convergence score based on entropy.

        Lower entropy = higher convergence.
        """
        entropy = self.get_current_entropy()
        max_entropy = np.log(len(self.history[-1])) if self.history else 1.0
        if max_entropy == 0:
            return 1.0
        return 1.0 - (entropy / max_entropy)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get entropy statistics.
        """
        return {
            "current_entropy": self.get_current_entropy(),
            "average_entropy": self.get_average_entropy(),
            "entropy_trend": self.get_entropy_trend(),
            "entropy_variation": self.get_entropy_variation(),
            "convergence_score": self.get_convergence_score(),
            "history_length": len(self.history),
            "window_size": self.window_size,
        }


class EntropyRegularizer:
    """
    Regularizes probability distributions using entropy.
    """

    @staticmethod
    def add_uniform_regularization(probabilities: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        """
        Add uniform regularization.

        p'_i = (1-α)*p_i + α/n
        """
        n = len(probabilities)
        uniform = np.ones(n) / n
        return (1 - alpha) * probabilities + alpha * uniform

    @staticmethod
    def add_entropy_penalty(probabilities: np.ndarray, lambda_reg: float = 0.1) -> np.ndarray:
        """
        Add entropy penalty to distribution.

        This encourages exploration by penalizing low entropy.
        """
        entropy = EntropyComputer.shannon_entropy(probabilities)
        n = len(probabilities)
        max_entropy = np.log(n)

        # If entropy is too low, add regularization
        if entropy < max_entropy * 0.5:
            return EntropyRegularizer.add_uniform_regularization(probabilities, lambda_reg)

        return probabilities

    @staticmethod
    def temper_distribution(probabilities: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Temper probability distribution.

        Higher temperature = more uniform
        Lower temperature = more peaked
        """
        if temperature == 1.0:
            return probabilities

        # Convert to logits and apply temperature
        logits = np.log(np.clip(probabilities, 1e-10, 1.0))
        tempered = np.exp(logits / temperature)
        return tempered / np.sum(tempered)
