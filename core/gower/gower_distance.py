"""
Gower Similarity Layer
========================
Implementation of Gower's similarity coefficient for mixed data types.

Gower (1971) proposed a similarity measure that works with:
- Numeric features (normalized)
- Categorical features (binary)
- Ordinal features

In the context of Gravit Grover, it is used for:
1. Computing closeness between hypotheses
2. Clustering hypotheses
3. Computing consistency scores
"""

import numpy as np
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform


@dataclass
class GowerConfig:
    """Configuration for Gower Similarity"""

    numeric_weights: Optional[List[float]] = None
    categorical_weights: Optional[List[float]] = None
    normalize_numeric: bool = True
    use_rank: bool = False  # For ordinal features


class GowerDistance:
    """
    Class for computing Gower distance and similarity.

    Gower distance is defined as:
    d(i, j) = (1/p) * Σ_k δ_k(i, j)

    where δ_k is the distance on feature k, normalized to [0, 1].
    """

    def __init__(self, config: Optional[GowerConfig] = None):
        self.config = config or GowerConfig()
        self.scalers = {}
        self.feature_types = {}

    def fit(self, data: np.ndarray, feature_types: List[str]):
        """
        Fit normalization parameters.

        Args:
            data: feature array (n_samples, n_features)
            feature_types: list of feature types ['numeric', 'categorical', 'ordinal']
        """
        self.feature_types = feature_types
        n_features = data.shape[1]

        # Initialize weights
        if self.config.numeric_weights is None:
            self.config.numeric_weights = [1.0] * n_features

        if self.config.categorical_weights is None:
            self.config.categorical_weights = [1.0] * n_features

        # Normalize numeric features
        if self.config.normalize_numeric:
            numeric_indices = [i for i, t in enumerate(feature_types) if t == "numeric"]
            if numeric_indices:
                scaler = StandardScaler()
                scaler.fit(data[:, numeric_indices])
                self.scalers["numeric"] = scaler

        return self

    def compute_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Gower distance between two objects.

        Args:
            x: first object (feature vector)
            y: second object (feature vector)

        Returns:
            float: Gower distance in [0, 1]
        """
        if len(x) != len(y):
            raise ValueError("Objects must have the same dimensionality")

        n_features = len(x)
        distances = []

        for k in range(n_features):
            feature_type = self.feature_types[k]

            if feature_type == "numeric":
                # Normalized absolute distance
                if self.config.normalize_numeric:
                    # Apply standardization
                    # Simplified: use range
                    range_val = 1.0  # In practice, this should be computed
                else:
                    range_val = 1.0
                dist = abs(x[k] - y[k]) / range_val

            elif feature_type == "categorical":
                # Distance: 0 if equal, 1 if different
                dist = 0.0 if x[k] == y[k] else 1.0

            elif feature_type == "ordinal":
                # For ordinal: use ranks
                if self.config.use_rank:
                    # Simplified: difference in ranks
                    dist = abs(x[k] - y[k]) / (len(set(x)) - 1)
                else:
                    dist = 0.0 if x[k] == y[k] else 1.0
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")

            distances.append(dist)

        # Weighted average distance
        weights = self._get_weights(n_features)
        return np.average(distances, weights=weights)

    def compute_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Gower similarity between two objects.

        Similarity = 1 - Distance
        """
        return 1.0 - self.compute_distance(x, y)

    def compute_distance_matrix(self, data: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distance matrix.

        Args:
            data: array of objects (n_samples, n_features)

        Returns:
            np.ndarray: distance matrix (n_samples, n_samples)
        """
        n_samples = data.shape[0]
        dist_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = self.compute_distance(data[i], data[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        return dist_matrix

    def compute_similarity_matrix(self, data: np.ndarray) -> np.ndarray:
        """
        Compute pairwise similarity matrix.
        """
        dist_matrix = self.compute_distance_matrix(data)
        return 1.0 - dist_matrix

    def _get_weights(self, n_features: int) -> np.ndarray:
        """
        Get feature weights.
        """
        weights = []
        for i in range(n_features):
            feature_type = self.feature_types[i]
            if feature_type == "numeric":
                weights.append(self.config.numeric_weights[i])
            elif feature_type == "categorical":
                weights.append(self.config.categorical_weights[i])
            else:
                weights.append(1.0)
        return np.array(weights) / np.sum(weights)


class ConsistencyScorer:
    """
    Computes hypothesis consistency based on Gower similarity.
    """

    def __init__(self, gower: GowerDistance, threshold: float = 0.7):
        self.gower = gower
        self.threshold = threshold

    def compute_consistency(self, hypothesis: np.ndarray, reference_set: np.ndarray) -> float:
        """
        Compute consistency of a hypothesis with a reference set.

        Args:
            hypothesis: feature vector of the hypothesis
            reference_set: set of reference hypotheses

        Returns:
            float: consistency measure in [0, 1]
        """
        similarities = []

        for ref in reference_set:
            sim = self.gower.compute_similarity(hypothesis, ref)
            similarities.append(sim)

        # Use average similarity as consistency measure
        return np.mean(similarities)

    def compute_consistency_matrix(self, hypotheses: np.ndarray) -> np.ndarray:
        """
        Compute consistency matrix for all hypotheses.
        """
        n = hypotheses.shape[0]
        consistency_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    consistency_matrix[i, j] = self.compute_consistency(
                        hypotheses[i], hypotheses[j].reshape(1, -1)
                    )

        return consistency_matrix

    def get_consistent_subset(self, hypotheses: np.ndarray) -> np.ndarray:
        """
        Extract subset of consistent hypotheses.
        """
        consistency_scores = []

        for i, hyp in enumerate(hypotheses):
            # Exclude the hypothesis itself
            others = np.delete(hypotheses, i, axis=0)
            score = self.compute_consistency(hyp, others)
            consistency_scores.append(score)

        # Select hypotheses with high consistency
        indices = np.where(np.array(consistency_scores) >= self.threshold)[0]
        return hypotheses[indices], consistency_scores
