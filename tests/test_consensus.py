"""
Unit tests for Consensus Layer.
"""

import unittest
import numpy as np
from core.consensus.gossip import GossipProtocol, GossipSimulator
from core.consensus.mixing import MixingMatrix
from core.consensus.spectral_gap import SpectralAnalyzer
from core.consensus.security import SignatureManager, IdentityVerifier, SecureMessageHandler


class TestGossipProtocol(unittest.TestCase):
    """Test GossipProtocol class."""

    def setUp(self):
        """Set up test data."""
        self.node_id = "test_node"
        self.protocol = GossipProtocol(self.node_id)

    def test_initialization(self):
        """Test initialization."""
        self.assertEqual(self.protocol.node_id, self.node_id)
        self.assertEqual(len(self.protocol.peers), 0)
        self.assertEqual(self.protocol.state.version, 0)

    def test_add_peer(self):
        """Test adding peer."""
        self.protocol.add_peer("peer1")
        self.assertIn("peer1", self.protocol.peers)
        self.assertEqual(len(self.protocol.peers), 1)

    def test_remove_peer(self):
        """Test removing peer."""
        self.protocol.add_peer("peer1")
        self.protocol.remove_peer("peer1")
        self.assertNotIn("peer1", self.protocol.peers)

    def test_update_state(self):
        """Test state update."""
        self.protocol.update_state("key", "value")
        self.assertEqual(self.protocol.get_state("key"), "value")
        self.assertEqual(self.protocol.state.version, 1)

    def test_get_statistics(self):
        """Test statistics."""
        stats = self.protocol.get_statistics()
        self.assertEqual(stats["node_id"], self.node_id)
        self.assertEqual(stats["peers"], 0)
        self.assertEqual(stats["version"], 0)


class TestMixingMatrix(unittest.TestCase):
    """Test MixingMatrix class."""

    def setUp(self):
        """Set up test data."""
        self.adjacency = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        self.mixing = MixingMatrix(self.adjacency)

    def test_create_metropolis_hastings(self):
        """Test Metropolis-Hastings matrix creation."""
        W = self.mixing.create_metropolis_hastings(self.adjacency)
        self.assertEqual(W.shape, (3, 3))
        self.assertTrue(np.allclose(np.sum(W, axis=1), 1.0))

    def test_create_symmetric(self):
        """Test symmetric matrix creation."""
        W = self.mixing.create_symmetric(self.adjacency)
        self.assertEqual(W.shape, (3, 3))
        self.assertTrue(np.allclose(np.sum(W, axis=1), 1.0))

    def test_spectral_gap(self):
        """Test spectral gap computation."""
        self.mixing.create_metropolis_hastings(self.adjacency)
        gap = self.mixing.compute_spectral_gap()
        self.assertGreater(gap, 0)
        self.assertLess(gap, 1)

    def test_apply_mixing(self):
        """Test applying mixing matrix."""
        self.mixing.create_metropolis_hastings(self.adjacency)
        state = np.array([1.0, 0.0, 0.0])
        new_state = self.mixing.apply_mixing(state)
        self.assertEqual(len(new_state), 3)
        self.assertTrue(np.allclose(np.sum(new_state), 1.0))


class TestSpectralAnalyzer(unittest.TestCase):
    """Test SpectralAnalyzer class."""

    def setUp(self):
        """Set up test data."""
        self.matrix = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
        self.analyzer = SpectralAnalyzer(self.matrix)

    def test_compute_eigenvalues(self):
        """Test eigenvalue computation."""
        eigenvalues, _ = self.analyzer.compute_full_spectrum()
        self.assertEqual(len(eigenvalues), 3)
        self.assertTrue(np.allclose(np.max(eigenvalues), 1.0))

    def test_compute_spectral_gap(self):
        """Test spectral gap computation."""
        gap = self.analyzer.compute_spectral_gap()
        self.assertGreater(gap, 0)
        self.assertLess(gap, 1)

    def test_compute_fiedler_value(self):
        """Test Fiedler value computation."""
        fiedler = self.analyzer.compute_fiedler_value()
        self.assertGreater(fiedler, 0)

    def test_analyze_connectivity(self):
        """Test connectivity analysis."""
        analysis = self.analyzer.analyze_connectivity()
        self.assertIn("is_connected", analysis)
        self.assertTrue(analysis["is_connected"])


class TestSecurity(unittest.TestCase):
    """Test security components."""

    def setUp(self):
        """Set up test data."""
        self.signature_manager = SignatureManager()
        self.identity_verifier = IdentityVerifier()

    def test_key_generation(self):
        """Test key pair generation."""
        private, public = self.signature_manager.generate_keypair()
        self.assertIsNotNone(private)
        self.assertIsNotNone(public)

    def test_signing_and_verification(self):
        """Test message signing and verification."""
        self.signature_manager.generate_keypair()
        message = {"test": "data"}
        signature = self.signature_manager.sign_message(message)
        result = self.signature_manager.verify_message(message, signature)
        self.assertTrue(result)

    def test_identity_verification(self):
        """Test identity verification."""
        self.identity_verifier.register_identity("node1", {"id": "node1"})
        status = self.identity_verifier.get_identity_status("node1")
        self.assertEqual(status["status"], "active")

    def test_revocation(self):
        """Test identity revocation."""
        self.identity_verifier.register_identity("node1", {"id": "node1"})
        self.identity_verifier.revoke_identity("node1", "test")
        status = self.identity_verifier.get_identity_status("node1")
        self.assertEqual(status["status"], "revoked")


if __name__ == "__main__":
    unittest.main()
