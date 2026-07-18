"""
KL Divergence Module
====================
Computes KL divergence between probability distributions.
"""

import numpy as np
from typing import Optional, List, Tuple
from scipy.stats import entropy


class KLDivergence:
    """
    Computes various KL divergence metrics.
    """

    @staticmethod
    def compute(p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute KL divergence D_KL(P || Q).

        D_KL(P || Q) = Σ_i p_i * log(p_i / q_i)
        """
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)

        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)

        return np.sum(p * np.log(p / q))

    @staticmethod
    def compute_symmetric(p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute symmetric KL divergence.

        D_sym(P || Q) = (D_KL(P || Q) + D_KL(Q || P)) / 2
        """
        return (KLDivergence.compute(p, q) + KLDivergence.compute(q, p)) / 2

    @staticmethod
    def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute Jensen-Shannon divergence.

        JSD(P || Q) = 0.5*D_KL(P || M) + 0.5*D_KL(Q || M)
        where M = (P + Q) / 2
        """
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)

        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)

        m = (p + q) / 2
        return 0.5 * KLDivergence.compute(p, m) + 0.5 * KLDivergence.compute(q, m)

    @staticmethod
    def compute_matrix(distributions: List[np.ndarray]) -> np.ndarray:
        """
        Compute pairwise KL divergence matrix.
        """
        n = len(distributions)
        kl_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    kl_matrix[i, j] = KLDivergence.compute(distributions[i], distributions[j])
                else:
                    kl_matrix[i, j] = 0.0

        return kl_matrix

    @staticmethod
    def compute_js_matrix(distributions: List[np.ndarray]) -> np.ndarray:
        """
        Compute pairwise Jensen-Shannon divergence matrix.
        """
        n = len(distributions)
        js_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    js_matrix[i, j] = KLDivergence.jensen_shannon(
                        distributions[i], distributions[j]
                    )
                else:
                    js_matrix[i, j] = 0.0

        return js_matrix


class DivergenceMonitor:
    """
    Monitors divergence between distributions over time.
    """

    def __init__(self, target_distribution: Optional[np.ndarray] = None):
        self.target = target_distribution
        self.history = []
        self.divergences = []

    def update(self, distribution: np.ndarray):
        """
        Update divergence history.
        """
        self.history.append(distribution.copy())

        if self.target is not None:
            divergence = KLDivergence.compute(distribution, self.target)
            self.divergences.append(divergence)

    def get_current_divergence(self) -> float:
        """
        Get current divergence from target.
        """
        if not self.divergences:
            return 0.0
        return self.divergences[-1]

    def get_average_divergence(self) -> float:
        """
        Get average divergence.
        """
        if not self.divergences:
            return 0.0
        return np.mean(self.divergences)

    def get_divergence_trend(self) -> float:
        """
        Get divergence trend.
        """
        if len(self.divergences) < 2:
            return 0.0

        x = np.arange(len(self.divergences))
        slope = np.polyfit(x, self.divergences, 1)[0]
        return slope

    def is_converged(self, tolerance: float = 1e-3, window: int = 5) -> bool:
        """
        Check if divergence has converged.
        """
        if len(self.divergences) < window:
            return False

        recent = self.divergences[-window:]
        return np.std(recent) < tolerance

    def get_statistics(self) -> dict:
        """
        Get divergence statistics.
        """
        return {
            "current_divergence": self.get_current_divergence(),
            "average_divergence": self.get_average_divergence(),
            "divergence_trend": self.get_divergence_trend(),
            "is_converged": self.is_converged(),
            "history_length": len(self.history),
        }


class DivergenceOptimizer:
    """
    Optimizes distribution to minimize divergence.
    """

    @staticmethod
    def project_to_simplex(vector: np.ndarray) -> np.ndarray:
        """
        Project vector to probability simplex.
        """
        vector = np.clip(vector, 0, 1)
        if np.sum(vector) == 0:
            return np.ones_like(vector) / len(vector)
        return vector / np.sum(vector)

    @staticmethod
    def optimize_kl(
        target: np.ndarray,
        initial: Optional[np.ndarray] = None,
        learning_rate: float = 0.1,
        iterations: int = 100,
    ) -> np.ndarray:
        """
        Optimize distribution to minimize KL divergence from target.
        """
        n = len(target)
        p = initial if initial is not None else np.ones(n) / n

        for _ in range(iterations):
            # Compute gradient
            p = np.clip(p, 1e-10, 1.0)
            gradient = np.log(p) - np.log(target)

            # Update
            p = p - learning_rate * gradient
            p = DivergenceOptimizer.project_to_simplex(p)

        return p

    @staticmethod
    def optimize_js(
        target: np.ndarray,
        initial: Optional[np.ndarray] = None,
        learning_rate: float = 0.1,
        iterations: int = 100,
    ) -> np.ndarray:
        """
        Optimize distribution to minimize Jensen-Shannon divergence.
        """
        n = len(target)
        p = initial if initial is not None else np.ones(n) / n

        for _ in range(iterations):
            # Compute gradient of JSD
            p = np.clip(p, 1e-10, 1.0)
            m = (p + target) / 2
            gradient = 0.5 * (np.log(p) - np.log(m)) + 0.5 * (np.log(target) - np.log(m))

            # Update
            p = p - learning_rate * gradient
            p = DivergenceOptimizer.project_to_simplex(p)

        return p
