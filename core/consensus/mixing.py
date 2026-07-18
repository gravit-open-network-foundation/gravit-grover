"""
Mixing Matrix Module
====================
Implements mixing matrices for consensus protocols.
"""

import numpy as np
from typing import Optional, List, Tuple
from scipy.linalg import eigvalsh


class MixingMatrix:
    """
    Creates and manages mixing matrices for consensus.
    """

    def __init__(self, adjacency: Optional[np.ndarray] = None):
        self.adjacency = adjacency
        self.matrix = None
        self.eigenvalues = None
        self.spectral_gap = None

    def create_metropolis_hastings(self, adjacency: np.ndarray) -> np.ndarray:
        """
        Create Metropolis-Hastings mixing matrix.

        W_ij = 1/(1 + max(d_i, d_j)) for (i,j) ∈ E
        W_ii = 1 - Σ_j W_ij
        """
        n = len(adjacency)
        W = np.zeros((n, n))
        degrees = np.sum(adjacency, axis=1)

        for i in range(n):
            for j in range(n):
                if i != j and adjacency[i, j] == 1:
                    W[i, j] = 1.0 / (1.0 + max(degrees[i], degrees[j]))

            W[i, i] = 1.0 - np.sum(W[i])

        self.matrix = W
        return W

    def create_heat_kernel(self, adjacency: np.ndarray, beta: float = 1.0) -> np.ndarray:
        """
        Create heat kernel mixing matrix.

        W = exp(-βL)
        where L is the graph Laplacian.
        """
        n = len(adjacency)
        degrees = np.sum(adjacency, axis=1)
        L = np.diag(degrees) - adjacency

        # Compute matrix exponential
        # For simplicity, approximate using series expansion
        # In practice, use scipy.linalg.expm
        I = np.eye(n)
        W = I - beta * L + (beta**2 / 2) * L @ L

        self.matrix = W
        return W

    def create_symmetric(self, adjacency: np.ndarray) -> np.ndarray:
        """
        Create symmetric mixing matrix.

        W_ij = 1/(d_i + 1) for (i,j) ∈ E
        W_ii = 1 - Σ_j W_ij
        """
        n = len(adjacency)
        W = np.zeros((n, n))
        degrees = np.sum(adjacency, axis=1)

        for i in range(n):
            for j in range(n):
                if i != j and adjacency[i, j] == 1:
                    W[i, j] = 1.0 / (degrees[i] + 1)

            W[i, i] = 1.0 - np.sum(W[i])

        self.matrix = W
        return W

    def create_push_sum(self, adjacency: np.ndarray) -> np.ndarray:
        """
        Create push-sum mixing matrix.

        W_ij = 1/(d_j + 1) for (i,j) ∈ E
        W_ii = 1/(d_i + 1)
        """
        n = len(adjacency)
        W = np.zeros((n, n))
        degrees = np.sum(adjacency, axis=1)

        for i in range(n):
            for j in range(n):
                if i != j and adjacency[i, j] == 1:
                    W[i, j] = 1.0 / (degrees[j] + 1)

            W[i, i] = 1.0 / (degrees[i] + 1)

        self.matrix = W
        return W

    def compute_eigenvalues(self) -> np.ndarray:
        """
        Compute eigenvalues of mixing matrix.
        """
        if self.matrix is None:
            raise ValueError("Mixing matrix not created")

        self.eigenvalues = eigvalsh(self.matrix)
        return self.eigenvalues

    def compute_spectral_gap(self) -> float:
        """
        Compute spectral gap of mixing matrix.
        """
        if self.eigenvalues is None:
            self.compute_eigenvalues()

        # Sort in descending order
        eigenvalues = np.sort(self.eigenvalues)[::-1]

        if len(eigenvalues) == 1:
            self.spectral_gap = 1.0
            return 1.0

        # Spectral gap = 1 - λ₂
        self.spectral_gap = 1.0 - eigenvalues[1]
        return self.spectral_gap

    def compute_convergence_rate(self) -> float:
        """
        Compute convergence rate.
        """
        spectral_gap = self.compute_spectral_gap()
        return 1.0 - spectral_gap

    def compute_mixing_time(self, epsilon: float = 1e-3) -> float:
        """
        Compute mixing time.

        T_mix = log(1/ε) / spectral_gap
        """
        spectral_gap = self.compute_spectral_gap()
        if spectral_gap == 0:
            return float("inf")
        return np.log(1.0 / epsilon) / spectral_gap

    def compute_second_eigenvector(self) -> np.ndarray:
        """
        Compute second eigenvector (Fiedler vector).
        """
        if self.matrix is None:
            raise ValueError("Mixing matrix not created")

        eigenvalues, eigenvectors = np.linalg.eigh(self.matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]

        if len(eigenvalues) > 1:
            return eigenvectors[:, 1]
        return eigenvectors[:, 0]

    def validate_matrix(self) -> Tuple[bool, str]:
        """
        Validate mixing matrix properties.
        """
        if self.matrix is None:
            return False, "Mixing matrix not created"

        # Check non-negativity
        if np.any(self.matrix < 0):
            return False, "Matrix has negative entries"

        # Check row sums
        row_sums = np.sum(self.matrix, axis=1)
        if not np.allclose(row_sums, 1.0):
            return False, f"Row sums not 1: {row_sums}"

        # Check eigenvalues
        eigenvalues = eigvalsh(self.matrix)
        if np.max(eigenvalues) > 1 + 1e-10:
            return False, f"Largest eigenvalue > 1: {np.max(eigenvalues)}"

        return True, "Matrix is valid"

    def apply_mixing(self, state: np.ndarray, iterations: int = 1) -> np.ndarray:
        """
        Apply mixing matrix to state vector.
        """
        if self.matrix is None:
            raise ValueError("Mixing matrix not created")

        current = state.copy()
        for _ in range(iterations):
            current = self.matrix @ current

        return current


class AdaptiveMixing:
    """
    Adaptive mixing matrix with changing graph topology.
    """

    def __init__(self, initial_adjacency: np.ndarray):
        self.adjacency = initial_adjacency.copy()
        self.mixing = MixingMatrix(self.adjacency)
        self.mixing.create_metropolis_hastings(self.adjacency)
        self.history = []

    def update_topology(self, new_adjacency: np.ndarray):
        """
        Update graph topology.
        """
        self.adjacency = new_adjacency.copy()
        self.mixing = MixingMatrix(self.adjacency)
        self.mixing.create_metropolis_hastings(self.adjacency)

        self.history.append({"time": len(self.history), "adjacency": new_adjacency.copy()})

    def add_node(self, node_id: int, connections: List[int]):
        """
        Add new node to topology.
        """
        n = len(self.adjacency) + 1
        new_adj = np.zeros((n, n))
        new_adj[: n - 1, : n - 1] = self.adjacency

        for conn in connections:
            new_adj[conn, n - 1] = 1
            new_adj[n - 1, conn] = 1

        self.update_topology(new_adj)

    def remove_node(self, node_id: int):
        """
        Remove node from topology.
        """
        n = len(self.adjacency) - 1
        mask = np.ones(len(self.adjacency), dtype=bool)
        mask[node_id] = False

        new_adj = self.adjacency[mask][:, mask]
        self.update_topology(new_adj)

    def get_convergence_trajectory(
        self, state: np.ndarray, max_iterations: int = 100
    ) -> List[np.ndarray]:
        """
        Get trajectory of convergence.
        """
        trajectory = [state.copy()]
        current = state.copy()

        for _ in range(max_iterations):
            current = self.mixing.apply_mixing(current)
            trajectory.append(current.copy())

        return trajectory
