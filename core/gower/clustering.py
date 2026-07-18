"""
Clustering Module for Gower Layer
=================================
Implements hierarchical clustering of hypotheses based on similarity.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt


@dataclass
class ClusteringConfig:
    """Configuration for clustering."""

    method: str = "average"  # single, complete, average, ward
    metric: str = "precomputed"  # euclidean, cosine, precomputed
    threshold: float = 0.7
    min_cluster_size: int = 2
    max_clusters: int = 10


class HierarchicalClusterer:
    """
    Hierarchical clustering of hypotheses using similarity matrix.
    """

    def __init__(self, config: Optional[ClusteringConfig] = None):
        self.config = config or ClusteringConfig()
        self.labels = None
        self.n_clusters = None
        self.linkage_matrix = None

    def fit(self, similarity_matrix: np.ndarray) -> "HierarchicalClusterer":
        """
        Fit clustering to similarity matrix.
        """
        # Convert similarity to distance
        distance_matrix = 1.0 - similarity_matrix

        # Ensure symmetric and valid
        n = len(distance_matrix)
        if distance_matrix.shape != (n, n):
            raise ValueError("Similarity matrix must be square")

        # Convert to condensed matrix for linkage
        condensed = squareform(distance_matrix, checks=True)

        # Compute linkage
        self.linkage_matrix = linkage(
            condensed, method=self.config.method, metric=self.config.metric
        )

        return self

    def get_clusters(
        self, n_clusters: Optional[int] = None, threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Get cluster labels.
        """
        if n_clusters is None:
            n_clusters = self.config.max_clusters

        if threshold is None:
            threshold = self.config.threshold

        # Determine clusters
        self.labels = fcluster(
            self.linkage_matrix,
            t=threshold if threshold < 1 else n_clusters,
            criterion="distance" if threshold < 1 else "maxclust",
        )

        self.n_clusters = len(np.unique(self.labels))

        # Filter small clusters
        if self.config.min_cluster_size > 1:
            self._filter_small_clusters()

        return self.labels

    def _filter_small_clusters(self):
        """Filter out clusters smaller than min_cluster_size."""
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        small_clusters = unique_labels[counts < self.config.min_cluster_size]

        for cluster_id in small_clusters:
            # Assign to nearest cluster
            indices = np.where(self.labels == cluster_id)[0]
            for idx in indices:
                nearest = self._find_nearest_cluster(idx, cluster_id)
                if nearest is not None:
                    self.labels[idx] = nearest

        # Re-assign any remaining
        unique_labels = np.unique(self.labels)
        for i, label in enumerate(unique_labels):
            self.labels[self.labels == label] = i + 1

        self.n_clusters = len(unique_labels)

    def _find_nearest_cluster(self, idx: int, exclude: int) -> Optional[int]:
        """Find nearest cluster for a point."""
        # This is a simplified implementation
        # In practice, would compute cluster centroids and distances
        unique_labels = np.unique(self.labels)
        unique_labels = unique_labels[unique_labels != exclude]

        if len(unique_labels) == 0:
            return None

        # Return the first available cluster
        return unique_labels[0]

    def get_cluster_sizes(self) -> Dict[int, int]:
        """Get sizes of each cluster."""
        if self.labels is None:
            return {}
        unique, counts = np.unique(self.labels, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}

    def get_cluster_representatives(
        self, hypotheses: List[str], similarity_matrix: np.ndarray
    ) -> Dict[int, str]:
        """
        Get representative hypothesis for each cluster.
        """
        if self.labels is None:
            self.get_clusters()

        representatives = {}
        unique_labels = np.unique(self.labels)

        for label in unique_labels:
            indices = np.where(self.labels == label)[0]

            # Find point with highest average similarity within cluster
            cluster_sim = similarity_matrix[np.ix_(indices, indices)]
            avg_sim = np.mean(cluster_sim, axis=1)
            rep_idx = indices[np.argmax(avg_sim)]
            representatives[int(label)] = hypotheses[rep_idx]

        return representatives

    def plot_dendrogram(self, labels: Optional[List[str]] = None, figsize: tuple = (10, 6)):
        """
        Plot dendrogram of hierarchical clustering.
        """
        plt.figure(figsize=figsize)
        dendrogram(self.linkage_matrix, labels=labels, leaf_rotation=90, leaf_font_size=10)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Hypotheses")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.show()

    def get_cluster_statistics(self, similarity_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Get statistics about clusters.
        """
        if self.labels is None:
            self.get_clusters()

        stats = {
            "n_clusters": self.n_clusters,
            "cluster_sizes": self.get_cluster_sizes(),
            "intra_cluster_similarity": {},
            "inter_cluster_similarity": {},
        }

        # Compute intra-cluster similarity
        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            indices = np.where(self.labels == label)[0]
            if len(indices) > 1:
                cluster_sim = similarity_matrix[np.ix_(indices, indices)]
                avg_sim = np.mean(cluster_sim[np.triu_indices_from(cluster_sim, k=1)])
                stats["intra_cluster_similarity"][int(label)] = float(avg_sim)

        # Compute inter-cluster similarity
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i + 1 :]:
                indices1 = np.where(self.labels == label1)[0]
                indices2 = np.where(self.labels == label2)[0]
                cluster_sim = np.mean(similarity_matrix[np.ix_(indices1, indices2)])
                key = f"{int(label1)}_{int(label2)}"
                stats["inter_cluster_similarity"][key] = float(cluster_sim)

        return stats


class ClusterEvaluator:
    """
    Evaluate clustering quality using various metrics.
    """

    @staticmethod
    def silhouette_score(similarity_matrix: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute silhouette score for clustering.
        """
        n = len(labels)
        if n < 2 or len(np.unique(labels)) < 2:
            return 0.0

        silhouette_scores = []

        for i in range(n):
            a_i = 0.0  # Mean similarity within cluster
            b_i = float("inf")  # Mean similarity to nearest cluster

            cluster_i = labels[i]
            cluster_indices = np.where(labels == cluster_i)[0]

            if len(cluster_indices) > 1:
                other_indices = cluster_indices[cluster_indices != i]
                a_i = np.mean(similarity_matrix[i][other_indices])

            for cluster_j in np.unique(labels):
                if cluster_j == cluster_i:
                    continue
                indices_j = np.where(labels == cluster_j)[0]
                b_j = np.mean(similarity_matrix[i][indices_j])
                b_i = min(b_i, b_j)

            if len(np.unique(labels)) > 1:
                s_i = (b_i - a_i) / max(a_i, b_i)
                silhouette_scores.append(s_i)

        return np.mean(silhouette_scores)

    @staticmethod
    def davies_bouldin_index(similarity_matrix: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute Davies-Bouldin index (lower is better).
        """
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return float("inf")

        # Compute cluster centroids in similarity space
        centroids = {}
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            centroids[label] = np.mean(similarity_matrix[indices], axis=0)

        # Compute intra-cluster distances
        intra_dist = {}
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            if len(indices) > 0:
                distances = 1.0 - similarity_matrix[np.ix_(indices, indices)]
                intra_dist[label] = np.mean(distances)

        # Compute Davies-Bouldin index
        db_index = 0.0
        for i in unique_labels:
            max_val = 0.0
            for j in unique_labels:
                if i != j:
                    centroid_dist = 1.0 - np.mean(
                        similarity_matrix[
                            np.ix_(np.where(labels == i)[0], np.where(labels == j)[0])
                        ]
                    )
                    if centroid_dist > 0:
                        val = (intra_dist[i] + intra_dist[j]) / centroid_dist
                        max_val = max(max_val, val)
            db_index += max_val

        return db_index / len(unique_labels)
