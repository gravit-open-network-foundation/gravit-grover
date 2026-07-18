"""
Unit tests for Gower Similarity Layer.
"""

import unittest
import numpy as np
from core.gower.gower_distance import GowerDistance, GowerConfig, ConsistencyScorer
from core.gower.similarity import SimilarityComputer, SimilarityConfig
from core.gower.clustering import HierarchicalClusterer, ClusteringConfig
from core.gower.consistency import ConsistencyChecker, ConsistencyConfig


class TestGowerDistance(unittest.TestCase):
    """Test GowerDistance class."""

    def setUp(self):
        """Set up test data."""
        self.features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        self.feature_types = ["numeric", "numeric", "numeric"]
        self.config = GowerConfig(normalize_numeric=True)
        self.gower = GowerDistance(self.config)
        self.gower.fit(self.features, self.feature_types)

    def test_distance_computation(self):
        """Test distance computation."""
        dist = self.gower.compute_distance(self.features[0], self.features[1])
        self.assertGreaterEqual(dist, 0)
        self.assertLessEqual(dist, 1)

    def test_similarity_computation(self):
        """Test similarity computation."""
        sim = self.gower.compute_similarity(self.features[0], self.features[1])
        self.assertGreaterEqual(sim, 0)
        self.assertLessEqual(sim, 1)
        self.assertEqual(sim, 1.0 - self.gower.compute_distance(self.features[0], self.features[1]))

    def test_distance_matrix(self):
        """Test distance matrix computation."""
        matrix = self.gower.compute_distance_matrix(self.features)
        self.assertEqual(matrix.shape, (3, 3))
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertTrue(np.allclose(np.diag(matrix), 0))

    def test_similarity_matrix(self):
        """Test similarity matrix computation."""
        matrix = self.gower.compute_similarity_matrix(self.features)
        self.assertEqual(matrix.shape, (3, 3))
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertTrue(np.allclose(np.diag(matrix), 1.0))

    def test_consistency_scorer(self):
        """Test consistency scorer."""
        scorer = ConsistencyScorer(self.gower, threshold=0.5)
        subset, scores = scorer.get_consistent_subset(self.features)
        self.assertEqual(len(subset), len(scores))
        self.assertGreaterEqual(len(subset), 0)
        self.assertLessEqual(len(subset), len(self.features))


class TestSimilarityComputer(unittest.TestCase):
    """Test SimilarityComputer class."""

    def setUp(self):
        """Set up test data."""
        self.features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.config = SimilarityConfig(method="cosine")
        self.computer = SimilarityComputer(self.config)
        self.computer.fit(self.features)

    def test_similarity_computation(self):
        """Test similarity computation."""
        sim = self.computer.compute_similarity(self.features[0], self.features[1])
        self.assertGreaterEqual(sim, -1)
        self.assertLessEqual(sim, 1)

    def test_similarity_matrix(self):
        """Test similarity matrix computation."""
        matrix = self.computer.compute_similarity_matrix(self.features)
        self.assertEqual(matrix.shape, (3, 3))
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertTrue(np.allclose(np.diag(matrix), 1.0))


class TestHierarchicalClusterer(unittest.TestCase):
    """Test HierarchicalClusterer class."""

    def setUp(self):
        """Set up test data."""
        self.similarity_matrix = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]])
        self.config = ClusteringConfig(method="average", max_clusters=2)
        self.clusterer = HierarchicalClusterer(self.config)

    def test_fit(self):
        """Test fitting."""
        self.clusterer.fit(self.similarity_matrix)
        self.assertIsNotNone(self.clusterer.linkage_matrix)

    def test_get_clusters(self):
        """Test cluster assignment."""
        self.clusterer.fit(self.similarity_matrix)
        labels = self.clusterer.get_clusters(n_clusters=2)
        self.assertEqual(len(labels), 3)
        self.assertEqual(len(np.unique(labels)), 2)

    def test_cluster_sizes(self):
        """Test cluster sizes."""
        self.clusterer.fit(self.similarity_matrix)
        self.clusterer.get_clusters(n_clusters=2)
        sizes = self.clusterer.get_cluster_sizes()
        self.assertEqual(sum(sizes.values()), 3)


class TestConsistencyChecker(unittest.TestCase):
    """Test ConsistencyChecker class."""

    def setUp(self):
        """Set up test data."""
        self.hypotheses = ["H1", "H2", "H3", "H4", "H5"]
        self.similarity_matrix = np.array(
            [
                [1.0, 0.9, 0.1, 0.2, 0.3],
                [0.9, 1.0, 0.2, 0.3, 0.4],
                [0.1, 0.2, 1.0, 0.8, 0.7],
                [0.2, 0.3, 0.8, 1.0, 0.9],
                [0.3, 0.4, 0.7, 0.9, 1.0],
            ]
        )
        self.config = ConsistencyConfig(threshold=0.5)
        self.checker = ConsistencyChecker(self.config)

    def test_get_consistent_subset(self):
        """Test getting consistent subset."""
        subset, scores = self.checker.get_consistent_subset(self.hypotheses, self.similarity_matrix)
        self.assertGreaterEqual(len(subset), 0)
        self.assertLessEqual(len(subset), len(self.hypotheses))
        self.assertEqual(len(scores), len(self.hypotheses))

    def test_detect_outliers(self):
        """Test outlier detection."""
        inliers, outliers = self.checker.detect_outliers(self.hypotheses, self.similarity_matrix)
        self.assertGreaterEqual(len(inliers), 0)
        self.assertGreaterEqual(len(outliers), 0)
        self.assertEqual(len(inliers) + len(outliers), len(self.hypotheses))


if __name__ == "__main__":
    unittest.main()
