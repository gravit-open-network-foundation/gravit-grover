"""
Memory Layer
============
Manages persistent state for Gravit Grover nodes.

Memory stores:
1. Hypothesis history
2. Node state at different time steps
3. Consensus results
4. Learned parameters
5. Model checkpoints
"""

import json
import pickle
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from pathlib import Path


@dataclass
class MemoryEntry:
    """Single entry in memory."""

    timestamp: datetime
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"


@dataclass
class HypothesisMemory:
    """Memory entry for a hypothesis."""

    hypothesis_id: str
    content: str
    features: np.ndarray
    weights: List[float]
    consistency: float
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusMemory:
    """Memory entry for consensus round."""

    round_id: str
    start_time: datetime
    end_time: datetime
    hypotheses: List[str]
    final_distribution: np.ndarray
    convergence: float
    entropy: float
    participants: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateMemory:
    """Memory entry for node state."""

    node_id: str
    iteration: int
    probabilities: np.ndarray
    scores: np.ndarray
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class Memory:
    """
    Persistent memory manager for Gravit Grover nodes.
    """

    def __init__(self, storage_path: Optional[str] = None, max_size: int = 10000):
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_size = max_size
        self.hypotheses: List[HypothesisMemory] = []
        self.consensus_rounds: List[ConsensusMemory] = []
        self.state_history: List[StateMemory] = []
        self.parameters: Dict[str, Any] = {}
        self.initialized = False

        if self.storage_path:
            self._initialize_storage()

    def _initialize_storage(self):
        """Initialize storage directory."""
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            (self.storage_path / "hypotheses").mkdir(exist_ok=True)
            (self.storage_path / "consensus").mkdir(exist_ok=True)
            (self.storage_path / "states").mkdir(exist_ok=True)
            (self.storage_path / "parameters").mkdir(exist_ok=True)
        self.initialized = True

    def add_hypothesis(self, hypothesis: HypothesisMemory):
        """Add hypothesis to memory."""
        self.hypotheses.append(hypothesis)
        if len(self.hypotheses) > self.max_size:
            self.hypotheses.pop(0)

        if self.storage_path:
            self._save_hypothesis(hypothesis)

    def add_consensus_round(self, round_data: ConsensusMemory):
        """Add consensus round to memory."""
        self.consensus_rounds.append(round_data)
        if len(self.consensus_rounds) > self.max_size // 10:
            self.consensus_rounds.pop(0)

        if self.storage_path:
            self._save_consensus(round_data)

    def add_state(self, state: StateMemory):
        """Add state to memory."""
        self.state_history.append(state)
        if len(self.state_history) > self.max_size:
            self.state_history.pop(0)

        if self.storage_path:
            self._save_state(state)

    def get_hypothesis(self, hypothesis_id: str) -> Optional[HypothesisMemory]:
        """Get hypothesis by ID."""
        for h in self.hypotheses:
            if h.hypothesis_id == hypothesis_id:
                return h
        return None

    def get_consensus_round(self, round_id: str) -> Optional[ConsensusMemory]:
        """Get consensus round by ID."""
        for r in self.consensus_rounds:
            if r.round_id == round_id:
                return r
        return None

    def get_best_hypothesis(self) -> Optional[HypothesisMemory]:
        """Get hypothesis with highest weight."""
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda h: h.weights[-1] if h.weights else 0)

    def get_state_at_iteration(self, iteration: int) -> Optional[StateMemory]:
        """Get state at specific iteration."""
        for state in self.state_history:
            if state.iteration == iteration:
                return state
        return None

    def get_recent_states(self, n: int = 10) -> List[StateMemory]:
        """Get n most recent states."""
        return self.state_history[-n:]

    def save_parameters(self, params: Dict[str, Any], name: str):
        """Save parameters to memory."""
        self.parameters[name] = params
        if self.storage_path:
            path = self.storage_path / "parameters" / f"{name}.json"
            with open(path, "w") as f:
                json.dump(
                    {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in params.items()},
                    f,
                )

    def load_parameters(self, name: str) -> Optional[Dict[str, Any]]:
        """Load parameters from memory."""
        if name in self.parameters:
            return self.parameters[name]

        if self.storage_path:
            path = self.storage_path / "parameters" / f"{name}.json"
            if path.exists():
                with open(path, "r") as f:
                    return json.load(f)
        return None

    def _save_hypothesis(self, hypothesis: HypothesisMemory):
        """Save hypothesis to disk."""
        path = self.storage_path / "hypotheses" / f"{hypothesis.hypothesis_id}.pkl"
        with open(path, "wb") as f:
            pickle.dump(hypothesis, f)

    def _save_consensus(self, round_data: ConsensusMemory):
        """Save consensus round to disk."""
        path = self.storage_path / "consensus" / f"{round_data.round_id}.pkl"
        with open(path, "wb") as f:
            pickle.dump(round_data, f)

    def _save_state(self, state: StateMemory):
        """Save state to disk."""
        path = self.storage_path / "states" / f"{state.node_id}_{state.iteration}.pkl"
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load_all_hypotheses(self) -> List[HypothesisMemory]:
        """Load all hypotheses from disk."""
        if not self.storage_path:
            return self.hypotheses

        loaded = []
        path = self.storage_path / "hypotheses"
        for file in path.glob("*.pkl"):
            with open(file, "rb") as f:
                loaded.append(pickle.load(f))
        return loaded

    def clear(self):
        """Clear memory."""
        self.hypotheses.clear()
        self.consensus_rounds.clear()
        self.state_history.clear()
        self.parameters.clear()

        if self.storage_path:
            import shutil

            shutil.rmtree(self.storage_path / "hypotheses")
            shutil.rmtree(self.storage_path / "consensus")
            shutil.rmtree(self.storage_path / "states")
            shutil.rmtree(self.storage_path / "parameters")
            self._initialize_storage()

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_hypotheses": len(self.hypotheses),
            "total_consensus_rounds": len(self.consensus_rounds),
            "total_states": len(self.state_history),
            "parameters_count": len(self.parameters),
            "max_size": self.max_size,
            "storage_path": str(self.storage_path) if self.storage_path else None,
            "initialized": self.initialized,
        }


class CheckpointManager:
    """
    Manages model checkpoints for persistence.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        step: int,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Save model checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step}_{timestamp}.pt"

        checkpoint = {
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "step": step,
            "timestamp": timestamp,
            "metadata": metadata or {},
        }

        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get latest checkpoint path."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if checkpoints:
            return str(checkpoints[-1])
        return None

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints with metadata."""
        checkpoints = []
        for path in sorted(self.checkpoint_dir.glob("checkpoint_*.pt")):
            with open(path, "rb") as f:
                data = pickle.load(f)
                checkpoints.append(
                    {
                        "path": str(path),
                        "step": data["step"],
                        "timestamp": data["timestamp"],
                        "metadata": data.get("metadata", {}),
                    }
                )
        return checkpoints

    def delete_checkpoint(self, checkpoint_path: str):
        """Delete checkpoint."""
        path = Path(checkpoint_path)
        if path.exists():
            path.unlink()

    def cleanup_old_checkpoints(self, keep_n: int = 5):
        """Delete old checkpoints keeping only n most recent."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        for checkpoint in checkpoints[:-keep_n]:
            checkpoint.unlink()
