"""
Similarity Module for Gower Layer
================================
Provides similarity computation functions for hypotheses.
"""

import numpy as np
from typing import Optional, List, Union, Dict
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform


@dataclass
class SimilarityConfig:
    """Configuration for similarity computation."""

    method: str = "gower"  # gower, cosine, euclidean, manhattan, custom
    normalize: bool = True
    weight_type: str = "uniform"  # uniform, inverse_variance, custom
    custom_weights: Optional[np.ndarray] = None


class SimilarityComputer:
    """
    Computes similarity between hypotheses using various metrics.
    """

    def __init__(self, config: Optional[SimilarityConfig] = None):
        self.config = config or SimilarityConfig()
        self.weights = None
        self.feature_scales = None

    def fit(self, features: np.ndarray):
        """
        Fit similarity computer to data.
        """
        n_features = features.shape[1]

        # Compute feature scales for normalization
        if self.config.normalize:
            self.feature_scales = np.std(features, axis=0)
            self.feature_scales[self.feature_scales == 0] = 1.0

        # Compute weights
        if self.config.weight_type == "uniform":
            self.weights = np.ones(n_features) / n_features
        elif self.config.weight_type == "inverse_variance":
            var = np.var(features, axis=0)
            self.weights = 1.0 / (var + 1e-10)
            self.weights /= np.sum(self.weights)
        elif self.config.weight_type == "custom" and self.config.custom_weights is not None:
            self.weights = self.config.custom_weights
            self.weights /= np.sum(self.weights)
        else:
            self.weights = np.ones(n_features) / n_features

        return self

    def compute_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute similarity between two vectors.
        """
        if self.config.method == "gower":
            return self._gower_similarity(x, y)
        elif self.config.method == "cosine":
            return self._cosine_similarity(x, y)
        elif self.config.method == "euclidean":
            return self._euclidean_similarity(x, y)
        elif self.config.method == "manhattan":
            return self._manhattan_similarity(x, y)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")

    def compute_similarity_matrix(self, features: np.ndarray) -> np.ndarray:
        """
        Compute pairwise similarity matrix.
        """
        n_samples = len(features)
        S = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i, n_samples):
                sim = self.compute_similarity(features[i], features[j])
                S[i, j] = sim
                S[j, i] = sim

        return S

    def _gower_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Gower similarity coefficient."""
        if self.feature_scales is not None:
            x = x / self.feature_scales
            y = y / self.feature_scales

        # Compute distance
        diff = np.abs(x - y)
        distance = np.sum(self.weights * diff)

        # Convert to similarity
        return 1.0 - distance

    def _cosine_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Cosine similarity."""
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        if norm_x == 0 or norm_y == 0:
            return 0.0
        return np.dot(x, y) / (norm_x * norm_y)

    def _euclidean_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Euclidean distance converted to similarity."""
        distance = np.linalg.norm(x - y)
        return 1.0 / (1.0 + distance)

    def _manhattan_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Manhattan distance converted to similarity."""
        distance = np.sum(np.abs(x - y))
        return 1.0 / (1.0 + distance)


class SimilarityKernel:
    """
    Kernel functions for similarity computation.
    """

    @staticmethod
    def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
        """RBF (Gaussian) kernel."""
        distance = np.linalg.norm(x - y) ** 2
        return np.exp(-gamma * distance)

    @staticmethod
    def polynomial_kernel(
        x: np.ndarray, y: np.ndarray, degree: int = 3, coef: float = 1.0
    ) -> float:
        """Polynomial kernel."""
        return (np.dot(x, y) + coef) ** degree

    @staticmethod
    def sigmoid_kernel(
        x: np.ndarray, y: np.ndarray, alpha: float = 1.0, beta: float = 0.0
    ) -> float:
        """Sigmoid kernel."""
        return np.tanh(alpha * np.dot(x, y) + beta)

    @staticmethod
    def laplacian_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
        """Laplacian kernel."""
        distance = np.linalg.norm(x - y, ord=1)
        return np.exp(-distance / sigma)


def compute_similarity_from_distance(
    distance_matrix: np.ndarray, method: str = "exp"
) -> np.ndarray:
    """
    Convert distance matrix to similarity matrix.

    Args:
        distance_matrix: Pairwise distances
        method: Conversion method ('exp', 'linear', 'gaussian')

    Returns:
        np.ndarray: Similarity matrix
    """
    if method == "exp":
        return np.exp(-distance_matrix)
    elif method == "linear":
        max_dist = np.max(distance_matrix)
        if max_dist > 0:
            return 1.0 - distance_matrix / max_dist
        return np.ones_like(distance_matrix)
    elif method == "gaussian":
        sigma = np.std(distance_matrix)
        if sigma > 0:
            return np.exp(-(distance_matrix**2) / (2 * sigma**2))
        return np.ones_like(distance_matrix)
    else:
        raise ValueError(f"Unknown conversion method: {method}")
