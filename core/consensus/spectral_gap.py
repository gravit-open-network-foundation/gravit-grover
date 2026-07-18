"""
Spectral Gap Module
===================
Computes and analyzes spectral properties for convergence analysis.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from scipy.linalg import eigvalsh, eigh
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix


class SpectralAnalyzer:
    """
    Analyzes spectral properties of consensus matrices.
    """

    def __init__(self, matrix: Optional[np.ndarray] = None):
        self.matrix = matrix
        self.eigenvalues = None
        self.eigenvectors = None
        self.spectral_gap = None
        self.condition_number = None

    def set_matrix(self, matrix: np.ndarray):
        """Set matrix for analysis."""
        self.matrix = matrix
        self.eigenvalues = None
        self.eigenvectors = None
        self.spectral_gap = None
        self.condition_number = None

    def compute_full_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute full eigenvalue decomposition.
        """
        if self.matrix is None:
            raise ValueError("Matrix not set")

        self.eigenvalues, self.eigenvectors = eigh(self.matrix)
        return self.eigenvalues, self.eigenvectors

    def compute_partial_spectrum(self, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute partial spectrum (largest eigenvalues).
        """
        if self.matrix is None:
            raise ValueError("Matrix not set")

        # Convert to sparse if large
        if len(self.matrix) > 1000:
            matrix_sparse = csr_matrix(self.matrix)
            self.eigenvalues, self.eigenvectors = eigs(matrix_sparse, k=k, which="LR")
        else:
            self.eigenvalues, self.eigenvectors = eigh(self.matrix)
            # Sort descending and take top k
            idx = np.argsort(self.eigenvalues)[::-1][:k]
            self.eigenvalues = self.eigenvalues[idx]
            self.eigenvectors = self.eigenvectors[:, idx]

        return self.eigenvalues, self.eigenvectors

    def compute_spectral_gap(self) -> float:
        """
        Compute spectral gap = 1 - λ₂.
        """
        if self.eigenvalues is None:
            self.compute_full_spectrum()

        eigenvalues = np.sort(self.eigenvalues)[::-1]
        if len(eigenvalues) == 1:
            self.spectral_gap = 1.0
            return 1.0

        # Spectral gap = 1 - λ₂ (second largest eigenvalue)
        self.spectral_gap = 1.0 - eigenvalues[1]

        # Ensure non-negative
        self.spectral_gap = max(0.0, self.spectral_gap)
        return self.spectral_gap

    def compute_condition_number(self) -> float:
        """
        Compute condition number = λ_max / λ_min.
        """
        if self.eigenvalues is None:
            self.compute_full_spectrum()

        if len(self.eigenvalues) == 0:
            self.condition_number = 1.0
            return 1.0

        eigenvalues = np.sort(self.eigenvalues)
        self.condition_number = (
            eigenvalues[-1] / eigenvalues[0] if eigenvalues[0] != 0 else float("inf")
        )
        return self.condition_number

    def compute_fiedler_value(self) -> float:
        """
        Compute Fiedler value (second smallest eigenvalue of Laplacian).
        """
        if self.matrix is None:
            raise ValueError("Matrix not set")

        # For a mixing matrix, need Laplacian
        # L = I - P where P is mixing matrix
        P = self.matrix
        L = np.eye(len(P)) - P
        eigenvalues = eigvalsh(L)
        eigenvalues = np.sort(eigenvalues)

        if len(eigenvalues) > 1:
            return eigenvalues[1]  # Fiedler value
        return 0.0

    def compute_convergence_time(self, epsilon: float = 1e-3) -> float:
        """
        Compute convergence time.
        """
        spectral_gap = self.compute_spectral_gap()
        if spectral_gap == 0:
            return float("inf")
        return np.log(1.0 / epsilon) / spectral_gap

    def compute_mixing_time(self, epsilon: float = 1e-3) -> float:
        """
        Compute mixing time.
        """
        return self.compute_convergence_time(epsilon)

    def compute_algebraic_connectivity(self) -> float:
        """
        Compute algebraic connectivity (λ₂ of Laplacian).
        """
        return self.compute_fiedler_value()

    def analyze_connectivity(self) -> Dict[str, Any]:
        """
        Analyze graph connectivity from spectrum.
        """
        if self.matrix is None:
            raise ValueError("Matrix not set")

        # For a connected graph, λ₂ > 0
        eigenvalues = eigvalsh(self.matrix)
        eigenvalues = np.sort(eigenvalues)

        # Check connectivity
        is_connected = eigenvalues[1] < 1.0  # λ₂ < 1 for connected

        # Check regularity
        degrees = np.sum(self.matrix, axis=1)
        is_regular = np.allclose(degrees, degrees[0])

        return {
            "is_connected": bool(is_connected),
            "is_regular": bool(is_regular),
            "degree_variation": float(np.std(degrees)),
            "spectral_gap": self.compute_spectral_gap(),
            "algebraic_connectivity": self.compute_algebraic_connectivity(),
            "n_components": len(eigenvalues[eigenvalues < 1e-10]) + 1,
        }

    def get_convergence_rate(self) -> float:
        """
        Get convergence rate = 1 - spectral_gap.
        """
        spectral_gap = self.compute_spectral_gap()
        return 1.0 - spectral_gap

    def estimate_consensus_time(self, n_nodes: int) -> float:
        """
        Estimate consensus time based on network size.

        T_consensus = O(log(n) / λ₂)
        """
        algebraic_connectivity = self.compute_algebraic_connectivity()
        if algebraic_connectivity == 0:
            return float("inf")
        return np.log(n_nodes) / algebraic_connectivity


class SpectralPlotter:
    """
    Utility for plotting spectral properties.
    """

    @staticmethod
    def plot_eigenvalues(
        eigenvalues: np.ndarray, title: str = "Eigenvalues", figsize: tuple = (10, 6)
    ):
        """
        Plot eigenvalue distribution.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=figsize)
        plt.plot(np.sort(eigenvalues), "bo-", alpha=0.6)
        plt.axhline(y=1, color="r", linestyle="--", label="λ=1")
        plt.axhline(y=0, color="k", linestyle="--", label="λ=0")
        plt.xlabel("Index")
        plt.ylabel("Eigenvalue")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_spectral_gap_trace(gaps: List[float], title: str = "Spectral Gap Evolution"):
        """
        Plot spectral gap over time.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(gaps, "g-", linewidth=2)
        plt.axhline(y=0.1, color="r", linestyle="--", label="Threshold")
        plt.xlabel("Time")
        plt.ylabel("Spectral Gap")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
