"""
Wasserstein Distance Module
===========================
Computes Wasserstein distance between probability distributions.
"""

import numpy as np
from typing import Optional, List, Tuple
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


class WassersteinDistance:
    """
    Computes Wasserstein (Earth Mover's) distance.
    """

    @staticmethod
    def compute_1d(p: np.ndarray, q: np.ndarray, support: Optional[np.ndarray] = None) -> float:
        """
        Compute 1D Wasserstein distance.

        W_1(P, Q) = ∫ |F_P(x) - F_Q(x)| dx
        """
        p = np.clip(p, 0, 1)
        q = np.clip(q, 0, 1)

        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)

        if support is None:
            support = np.arange(len(p))

        # Compute cumulative distributions
        cdf_p = np.cumsum(p)
        cdf_q = np.cumsum(q)

        # Compute 1D Wasserstein
        dx = support[1] - support[0] if len(support) > 1 else 1.0
        return np.sum(np.abs(cdf_p - cdf_q)) * dx

    @staticmethod
    def compute_2d(
        p: np.ndarray,
        q: np.ndarray,
        support_p: Optional[np.ndarray] = None,
        support_q: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute 2D Wasserstein distance using linear programming.
        """
        p = np.clip(p, 0, 1)
        q = np.clip(q, 0, 1)

        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)

        if support_p is None:
            support_p = np.array([[i, 0] for i in range(len(p))])
        if support_q is None:
            support_q = np.array([[i, 0] for i in range(len(q))])

        # Compute cost matrix
        cost_matrix = cdist(support_p, support_q, metric="euclidean")

        # Solve transportation problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Compute Wasserstein distance
        distance = np.sum(cost_matrix[row_ind, col_ind] * p[row_ind])
        return distance

    @staticmethod
    def compute_sinkhorn(
        p: np.ndarray, q: np.ndarray, epsilon: float = 0.1, max_iterations: int = 100
    ) -> float:
        """
        Compute Sinkhorn distance (entropy-regularized Wasserstein).
        """
        p = np.clip(p, 0, 1)
        q = np.clip(q, 0, 1)

        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)

        n, m = len(p), len(q)

        # Cost matrix
        C = np.abs(np.arange(n)[:, None] - np.arange(m)[None, :])

        # Sinkhorn algorithm
        K = np.exp(-C / epsilon)

        # Initialize
        u = np.ones(n) / n
        v = np.ones(m) / m

        for _ in range(max_iterations):
            u = p / (K @ v)
            v = q / (K.T @ u)

        # Compute transport plan
        P = np.diag(u) @ K @ np.diag(v)

        # Compute Sinkhorn distance
        distance = np.sum(P * C)
        return distance

    @staticmethod
    def compute_matrix(distributions: List[np.ndarray]) -> np.ndarray:
        """
        Compute pairwise Wasserstein distance matrix.
        """
        n = len(distributions)
        w_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    w_matrix[i, j] = WassersteinDistance.compute_1d(
                        distributions[i], distributions[j]
                    )
                else:
                    w_matrix[i, j] = 0.0

        return w_matrix


class WassersteinMonitor:
    """
    Monitors Wasserstein distances over time.
    """

    def __init__(self, reference_distribution: Optional[np.ndarray] = None):
        self.reference = reference_distribution
        self.history = []
        self.distances = []

    def update(self, distribution: np.ndarray):
        """
        Update distance history.
        """
        self.history.append(distribution.copy())

        if self.reference is not None:
            distance = WassersteinDistance.compute_1d(distribution, self.reference)
            self.distances.append(distance)

    def get_current_distance(self) -> float:
        """
        Get current distance from reference.
        """
        if not self.distances:
            return 0.0
        return self.distances[-1]

    def get_average_distance(self) -> float:
        """
        Get average distance.
        """
        if not self.distances:
            return 0.0
        return np.mean(self.distances)

    def get_distance_trend(self) -> float:
        """
        Get distance trend.
        """
        if len(self.distances) < 2:
            return 0.0

        x = np.arange(len(self.distances))
        slope = np.polyfit(x, self.distances, 1)[0]
        return slope

    def get_wasserstein_similarity(self) -> float:
        """
        Get similarity based on Wasserstein distance.

        Similarity = 1 / (1 + distance)
        """
        distance = self.get_current_distance()
        return 1.0 / (1.0 + distance)

    def get_statistics(self) -> dict:
        """
        Get Wasserstein statistics.
        """
        return {
            "current_distance": self.get_current_distance(),
            "average_distance": self.get_average_distance(),
            "distance_trend": self.get_distance_trend(),
            "similarity": self.get_wasserstein_similarity(),
            "history_length": len(self.history),
        }


class EarthMoverOptimizer:
    """
    Optimizes distribution to minimize Earth Mover's distance.
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
    def optimize_wasserstein(
        target: np.ndarray,
        initial: Optional[np.ndarray] = None,
        learning_rate: float = 0.01,
        iterations: int = 100,
    ) -> np.ndarray:
        """
        Optimize distribution to minimize Wasserstein distance from target.
        """
        n = len(target)
        p = initial if initial is not None else np.ones(n) / n

        target = np.clip(target, 0, 1)
        target = target / np.sum(target)

        for _ in range(iterations):
            # Compute gradient (approximate)
            p = np.clip(p, 1e-10, 1.0)
            cdf_p = np.cumsum(p)
            cdf_target = np.cumsum(target)

            gradient = np.sign(cdf_p - cdf_target)

            # Update
            p = p - learning_rate * gradient
            p = EarthMoverOptimizer.project_to_simplex(p)

        return p
