"""
Multiplicative Weights Layer
=============================
Implementation of the Multiplicative Weights Update (MWU) algorithm.

MWU is used in Gravit Grover for:
1. Updating hypothesis probabilities based on local scores
2. Adaptively changing the distribution
3. Balancing exploration and exploitation

Formally:
p_i^{t+1} = p_i^t * exp(η * score_i) / Z_t
"""

import numpy as np
from typing import List, Optional, Callable, Union
from dataclasses import dataclass
from scipy.special import softmax


@dataclass
class MWUConfig:
    """Configuration for Multiplicative Weights"""

    learning_rate: float = 1.0  # η (eta) - learning rate
    temperature: float = 1.0  # Temperature for softmax
    min_prob: float = 1e-10  # Minimum probability (for numerical stability)
    max_prob: float = 0.999  # Maximum probability
    use_entropy_regularization: bool = True
    entropy_coefficient: float = 0.1


class MultiplicativeWeights:
    """
    Class for implementing Multiplicative Weights Update.
    """

    def __init__(self, config: Optional[MWUConfig] = None):
        self.config = config or MWUConfig()
        self.history = []

    def update(
        self, probabilities: np.ndarray, scores: np.ndarray, return_entropy: bool = False
    ) -> Union[np.ndarray, tuple]:
        """
        Update probabilities based on received scores.

        Args:
            probabilities: current probability distribution (n,)
            scores: scores for each hypothesis (n,)
            return_entropy: also return entropy

        Returns:
            np.ndarray: updated probability distribution
            or tuple: (probabilities, entropy)
        """
        # Check dimensions
        if len(probabilities) != len(scores):
            raise ValueError("Probability and score dimensions must match")

        # 1. Compute exponential weights
        exp_weights = np.exp(self.config.learning_rate * scores)

        # 2. Multiply by current probabilities
        unnormalized = probabilities * exp_weights

        # 3. Normalize
        Z = np.sum(unnormalized)
        if Z == 0:
            # Numerical stability: if sum is 0, use uniform distribution
            new_probs = np.ones_like(probabilities) / len(probabilities)
        else:
            new_probs = unnormalized / Z

        # 4. Clip for numerical stability
        new_probs = np.clip(new_probs, self.config.min_prob, self.config.max_prob)

        # 5. Entropy regularization (if enabled)
        if self.config.use_entropy_regularization:
            entropy = self.compute_entropy(new_probs)
            # Mix with uniform distribution
            uniform = np.ones_like(new_probs) / len(new_probs)
            alpha = self.config.entropy_coefficient * entropy
            new_probs = (1 - alpha) * new_probs + alpha * uniform

        # Save history
        self.history.append(
            {
                "probabilities": new_probs.copy(),
                "scores": scores.copy(),
                "entropy": self.compute_entropy(new_probs),
            }
        )

        if return_entropy:
            return new_probs, self.compute_entropy(new_probs)
        return new_probs

    def update_with_temperature(
        self, probabilities: np.ndarray, scores: np.ndarray, temperature: Optional[float] = None
    ) -> np.ndarray:
        """
        Update using temperature scaling (softmax approach).
        """
        temp = temperature or self.config.temperature

        # Compute logits with temperature
        scaled_scores = scores / temp

        # Use softmax
        logits = np.log(probabilities + 1e-10) + self.config.learning_rate * scaled_scores
        new_probs = softmax(logits)

        # Clip
        new_probs = np.clip(new_probs, self.config.min_prob, self.config.max_prob)

        return new_probs

    def compute_entropy(self, probabilities: np.ndarray) -> float:
        """
        Compute Shannon entropy.
        """
        # Protect against log(0)
        probs = np.clip(probabilities, 1e-10, 1.0)
        return -np.sum(probs * np.log(probs))

    def compute_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute KL divergence: D_KL(P || Q)
        """
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        return np.sum(p * np.log(p / q))

    def get_best_hypothesis(self, probabilities: np.ndarray, hypotheses: List[str]) -> str:
        """
        Get the best hypothesis from the distribution.
        """
        best_idx = np.argmax(probabilities)
        return hypotheses[best_idx]

    def get_top_k(
        self, probabilities: np.ndarray, hypotheses: List[str], k: int = 3
    ) -> List[tuple]:
        """
        Get top-K hypotheses with their probabilities.
        """
        indices = np.argsort(probabilities)[-k:][::-1]
        return [(hypotheses[i], probabilities[i]) for i in indices]


class AdaptiveMultiplicativeWeights(MultiplicativeWeights):
    """
    Adaptive version of MWU with changing learning rate.
    """

    def __init__(
        self, config: Optional[MWUConfig] = None, initial_lr: float = 1.0, lr_decay: float = 0.99
    ):
        super().__init__(config)
        self.initial_lr = initial_lr
        self.lr_decay = lr_decay
        self.current_lr = initial_lr
        self.step_count = 0

    def update(
        self, probabilities: np.ndarray, scores: np.ndarray, adaptive: bool = True
    ) -> np.ndarray:
        """
        Update with adaptive learning rate.
        """
        if adaptive:
            # Update learning rate
            self.current_lr = self.initial_lr * (self.lr_decay**self.step_count)
            self.config.learning_rate = self.current_lr
            self.step_count += 1

        return super().update(probabilities, scores)

    def reset(self):
        """Reset adaptive state."""
        self.current_lr = self.initial_lr
        self.step_count = 0
        self.history = []


class EnsembleWeights:
    """
    Ensemble of multiple MWU experts.
    """

    def __init__(self, n_experts: int = 3, learning_rates: Optional[List[float]] = None):
        self.n_experts = n_experts
        self.learning_rates = learning_rates or [0.5, 1.0, 2.0]
        self.experts = [
            MultiplicativeWeights(MWUConfig(learning_rate=lr)) for lr in self.learning_rates
        ]

    def update_ensemble(self, probabilities: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """
        Update all experts and average results.
        """
        results = []

        for expert in self.experts:
            result = expert.update(probabilities.copy(), scores)
            results.append(result)

        # Average probabilities
        ensemble_prob = np.mean(results, axis=0)

        # Normalize
        return ensemble_prob / np.sum(ensemble_prob)

    def get_expert_agreement(self) -> float:
        """
        Compute agreement between experts.
        """
        # Can implement agreement measure here
        # e.g., average pairwise KL divergence
        pass
