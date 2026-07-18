"""
Objective Function Layer
=======================
Defines the optimization objectives for the Gravit Grover Engine.

The objective function determines what the network optimizes:
1. Log-likelihood of hypotheses given evidence
2. Consistency of hypotheses with validators
3. Entropy of the consensus distribution
4. Diversity of the hypothesis set
"""

import numpy as np
from typing import List, Optional, Callable, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ObjectiveConfig:
    """Configuration for objective functions."""

    entropy_weight: float = 0.1
    diversity_weight: float = 0.05
    consistency_weight: float = 1.0
    use_log_likelihood: bool = True
    min_prob: float = 1e-10


class ObjectiveFunction(ABC):
    """Base class for objective functions."""

    @abstractmethod
    def compute(self, hypotheses: np.ndarray, scores: np.ndarray) -> float:
        """Compute objective value."""
        pass

    @abstractmethod
    def gradient(self, hypotheses: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Compute gradient of objective."""
        pass


class LogLikelihoodObjective(ObjectiveFunction):
    """
    Maximize log-likelihood of hypotheses given evidence.

    L = Σ_i log(p_i) * score_i
    """

    def __init__(self, config: Optional[ObjectiveConfig] = None):
        self.config = config or ObjectiveConfig()

    def compute(self, hypotheses: np.ndarray, scores: np.ndarray) -> float:
        """Compute log-likelihood."""
        probs = self._get_probabilities(hypotheses)
        log_probs = np.log(np.clip(probs, self.config.min_prob, 1.0))
        return np.sum(log_probs * scores)

    def gradient(self, hypotheses: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Compute gradient of log-likelihood."""
        probs = self._get_probabilities(hypotheses)
        return scores / np.clip(probs, self.config.min_prob, 1.0)

    def _get_probabilities(self, hypotheses: np.ndarray) -> np.ndarray:
        """Extract probabilities from hypotheses."""
        # Assumes hypotheses contains probability as last dimension
        if hypotheses.shape[-1] == 1:
            return hypotheses.flatten()
        # Otherwise compute from features
        return np.ones(len(hypotheses)) / len(hypotheses)


class ConsistencyObjective(ObjectiveFunction):
    """
    Maximize consistency of hypotheses with validators.

    C = (1/N) Σ_i consistency_score_i
    """

    def __init__(self, consistency_scores: Optional[np.ndarray] = None):
        self.consistency_scores = consistency_scores

    def compute(self, hypotheses: np.ndarray, scores: np.ndarray) -> float:
        """Compute consistency objective."""
        if self.consistency_scores is None:
            # Compute consistency from scores
            return np.mean(scores)
        return np.mean(self.consistency_scores[: len(hypotheses)])

    def gradient(self, hypotheses: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Compute gradient of consistency."""
        if self.consistency_scores is None:
            return np.ones(len(hypotheses)) / len(hypotheses)
        return self.consistency_scores / np.sum(self.consistency_scores)


class EntropyObjective(ObjectiveFunction):
    """
    Minimize entropy of distribution.

    H = -Σ_i p_i * log(p_i)
    """

    def compute(self, hypotheses: np.ndarray, scores: np.ndarray) -> float:
        """Compute entropy objective."""
        probs = self._get_probabilities(hypotheses)
        probs = np.clip(probs, 1e-10, 1.0)
        return -np.sum(probs * np.log(probs))

    def gradient(self, hypotheses: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Compute gradient of entropy."""
        probs = self._get_probabilities(hypotheses)
        return -np.log(np.clip(probs, 1e-10, 1.0)) - 1.0

    def _get_probabilities(self, hypotheses: np.ndarray) -> np.ndarray:
        """Extract probabilities from hypotheses."""
        if hypotheses.shape[-1] == 1:
            return hypotheses.flatten()
        return np.ones(len(hypotheses)) / len(hypotheses)


class DiversityObjective(ObjectiveFunction):
    """
    Maximize diversity of hypothesis set.

    D = 1 - Σ_i Σ_j S_ij / (n*(n-1))
    """

    def __init__(self, similarity_matrix: Optional[np.ndarray] = None):
        self.similarity_matrix = similarity_matrix

    def compute(self, hypotheses: np.ndarray, scores: np.ndarray) -> float:
        """Compute diversity objective."""
        if self.similarity_matrix is None:
            # Compute from hypotheses
            n = len(hypotheses)
            if n <= 1:
                return 1.0
            # Assuming Euclidean distance for simplicity
            diff = hypotheses[:, None] - hypotheses[None, :]
            distances = np.sqrt(np.sum(diff**2, axis=-1))
            avg_distance = np.mean(distances[np.triu_indices_from(distances, k=1)])
            return avg_distance / np.sqrt(hypotheses.shape[1])

        n = len(self.similarity_matrix)
        if n <= 1:
            return 1.0
        upper_tri = self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)]
        return 1.0 - np.mean(upper_tri)

    def gradient(self, hypotheses: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Compute gradient of diversity."""
        # Simplified gradient
        n = len(hypotheses)
        if n <= 1:
            return np.zeros_like(hypotheses)

        grad = np.zeros_like(hypotheses)
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = hypotheses[i] - hypotheses[j]
                    dist = np.sqrt(np.sum(diff**2))
                    if dist > 0:
                        grad[i] += diff / dist
        return grad / (n * (n - 1))


class CombinedObjective:
    """
    Combined objective with multiple components.

    O = α*L + β*C - γ*H + δ*D
    """

    def __init__(self, config: ObjectiveConfig):
        self.config = config
        self.components = []
        self.weights = []

        # Initialize components
        self.add_component(LogLikelihoodObjective(config), 1.0)
        self.add_component(ConsistencyObjective(), config.consistency_weight)
        self.add_component(EntropyObjective(), -config.entropy_weight)
        self.add_component(DiversityObjective(), config.diversity_weight)

    def add_component(self, component: ObjectiveFunction, weight: float):
        """Add objective component with weight."""
        self.components.append(component)
        self.weights.append(weight)

    def compute(self, hypotheses: np.ndarray, scores: np.ndarray) -> float:
        """Compute combined objective."""
        total = 0.0
        for component, weight in zip(self.components, self.weights):
            total += weight * component.compute(hypotheses, scores)
        return total

    def gradient(self, hypotheses: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Compute combined gradient."""
        grad = np.zeros_like(hypotheses)
        for component, weight in zip(self.components, self.weights):
            grad += weight * component.gradient(hypotheses, scores)
        return grad

    def is_converged(
        self, hypotheses: np.ndarray, scores: np.ndarray, tolerance: float = 1e-6
    ) -> bool:
        """
        Check if optimization has converged.
        """
        grad = self.gradient(hypotheses, scores)
        return np.max(np.abs(grad)) < tolerance


class RewardFunction:
    """
    Reward function for hypothesis scoring.

    R = α * consistency + β * confidence + γ * novelty
    """

    def __init__(self, alpha: float = 0.6, beta: float = 0.3, gamma: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_reward(self, consistency: float, confidence: float, novelty: float) -> float:
        """Compute reward for a hypothesis."""
        return self.alpha * consistency + self.beta * confidence + self.gamma * novelty

    def compute_rewards(
        self, consistencies: np.ndarray, confidences: np.ndarray, novelties: np.ndarray
    ) -> np.ndarray:
        """Compute rewards for multiple hypotheses."""
        rewards = np.zeros_like(consistencies)
        for i in range(len(consistencies)):
            rewards[i] = self.compute_reward(consistencies[i], confidences[i], novelties[i])
        return rewards


class LossFunction:
    """
    Loss function for hypothesis evaluation.

    L = MSE between predicted and actual scores
    """

    def __init__(self, use_huber: bool = False, delta: float = 1.0):
        self.use_huber = use_huber
        self.delta = delta

    def compute_loss(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """Compute loss between predicted and actual."""
        if self.use_huber:
            return self._huber_loss(predicted, actual)
        return self._mse_loss(predicted, actual)

    def _mse_loss(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """Mean squared error loss."""
        return np.mean((predicted - actual) ** 2)

    def _huber_loss(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """Huber loss (robust to outliers)."""
        diff = predicted - actual
        abs_diff = np.abs(diff)
        loss = np.where(
            abs_diff <= self.delta, 0.5 * diff**2, self.delta * (abs_diff - 0.5 * self.delta)
        )
        return np.mean(loss)
