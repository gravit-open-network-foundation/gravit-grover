"""
Grover Engine
=============

One round of the Gravit Grover protocol over a population of validator nodes:

1. **Local update** - multiplicative weights on each node's belief over the
   hypotheses (``core.probability.multiplicative_weights``).
2. **Gossip mixing** - each node moves a fraction ``epsilon`` of its belief
   towards the stake-weighted average of its neighbours.
3. **Amplification** - every node's stake is rescaled by how well its belief
   agrees with the network-wide consensus, raised to the power ``gamma``.

Naming note: "Grover" refers to the *inspiration* - amplitude amplification in
Grover's quantum search algorithm (Grover, 1996). This engine is an entirely
classical analogue: it sharpens the weight of globally consistent hypotheses
with an exponent ``gamma > 1``. It makes **no claim of quantum speed-up**.

All randomness comes from one ``numpy.random.Generator`` created from ``seed``,
so a run with the same seed is reproducible bit for bit.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from core.probability.multiplicative_weights import MultiplicativeWeights, MWUConfig


@dataclass
class GroverConfig:
    """Protocol parameters (defaults: the 10x Sybil experiment)."""

    eta: float = 0.2  # learning rate of the local multiplicative-weights update
    gamma: float = 1.5  # amplification exponent (gamma > 1 sharpens)
    epsilon: float = 0.1  # gossip mixing fraction
    fanout: int = 10  # neighbours per node in the gossip graph


@dataclass
class GroverState:
    """Beliefs, stakes and fixed gossip graph of the node population."""

    probs: np.ndarray  # (n_nodes, n_hypotheses), rows sum to 1
    scores: np.ndarray  # (n_nodes, n_hypotheses), each node's local evidence
    stake: np.ndarray  # (n_nodes,), positive
    neighbors: np.ndarray  # (n_nodes, fanout) neighbour indices


class GroverEngine:
    """Classical, Grover-inspired consensus engine over validator nodes."""

    def __init__(self, config: Optional[GroverConfig] = None, seed: Optional[int] = None):
        self.config = config or GroverConfig()
        self.rng = np.random.default_rng(seed)
        # The local step *is* the library MWU (no clipping, no entropy mixing);
        # local_update() is its vectorised form and is tested against it.
        self.mwu = MultiplicativeWeights(
            MWUConfig(
                learning_rate=self.config.eta,
                min_prob=0.0,
                max_prob=1.0,
                use_entropy_regularization=False,
            )
        )

    # -- building blocks ----------------------------------------------------

    def gossip_graph(self, n_nodes: int) -> np.ndarray:
        """Random fixed-fanout graph without self-loops, drawn from the engine RNG."""
        fanout = self.config.fanout
        if fanout >= n_nodes:
            raise ValueError("fanout must be smaller than the number of nodes")
        neighbors = np.empty((n_nodes, fanout), dtype=int)
        for i in range(n_nodes):
            others = np.delete(np.arange(n_nodes), i)
            neighbors[i] = self.rng.choice(others, size=fanout, replace=False)
        return neighbors

    def local_update(self, probs: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Multiplicative weights for every node at once: p_i <- p_i * exp(eta * s_i) / Z_i."""
        updated = probs * np.exp(self.config.eta * scores)
        return updated / updated.sum(axis=1, keepdims=True)

    def mix(self, probs: np.ndarray, stake: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
        """Stake-weighted gossip: p_i <- (1 - eps) p_i + eps * avg_{j in N(i)} p_j (weights = stake_j)."""
        eps = self.config.epsilon
        nb_stake = stake[neighbors]  # (n, fanout)
        nb_probs = probs[neighbors]  # (n, fanout, h)
        totals = nb_stake.sum(axis=1, keepdims=True)
        weights = np.where(
            totals > 0, nb_stake / np.where(totals > 0, totals, 1.0), 1.0 / neighbors.shape[1]
        )
        nb_avg = np.einsum("nf,nfh->nh", weights, nb_probs)
        return (1 - eps) * probs + eps * nb_avg

    def amplify(self, probs: np.ndarray, stake: np.ndarray) -> np.ndarray:
        """Rescale stake by agreement with the stake-weighted consensus, amplified by gamma."""
        consensus = (probs * stake[:, None]).sum(axis=0) / stake.sum()
        factors = (consensus / (consensus.mean() + 1e-10)) ** self.config.gamma
        return stake * (probs * factors).sum(axis=1)

    # -- protocol -----------------------------------------------------------

    def step(self, state: GroverState) -> GroverState:
        """One full round: local update -> gossip mixing -> amplification."""
        probs = self.local_update(state.probs, state.scores)
        probs = self.mix(probs, state.stake, state.neighbors)
        stake = self.amplify(probs, state.stake)
        return GroverState(probs=probs, scores=state.scores, stake=stake, neighbors=state.neighbors)

    @staticmethod
    def confidence(state: GroverState) -> np.ndarray:
        """Stake-weighted network confidence in each hypothesis."""
        return (state.probs * state.stake[:, None]).sum(axis=0) / state.stake.sum()

    def run(
        self, state: GroverState, rounds: int, honest: Optional[np.ndarray] = None
    ) -> List[Dict[str, float]]:
        """Run ``rounds`` rounds; record confidence in hypotheses 0 and 1 and, if an
        ``honest`` mask is given, the stake held by honest vs. other nodes."""
        history: List[Dict[str, float]] = []
        for r in range(rounds):
            state = self.step(state)
            conf = self.confidence(state)
            row: Dict[str, float] = {
                "round": r + 1,
                "confidence_true": float(conf[0]),
                "confidence_false": float(conf[1]),
            }
            if honest is not None:
                hw = float(state.stake[honest].sum())
                sw = float(state.stake[~honest].sum())
                row.update(
                    {"honest_weight": hw, "sybil_weight": sw, "sybil_fraction": sw / (hw + sw)}
                )
            history.append(row)
        self.final_state = state
        return history
