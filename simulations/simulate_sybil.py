#!/usr/bin/env python3
"""
GQRVP 10x Sybil attack simulation
=================================

100 honest nodes (stake 1.0, believe the true claim) against 1000 Sybil nodes
(stake 0.01, believe the false claim), 30 rounds of the Gravit Grover protocol
(eta=0.2, gamma=1.5, epsilon=0.1, fanout=10).

Success criteria: Confidence(true) > 0.90 and Confidence(false) < 0.10 after
30 rounds, Sybil stake share < 5% at round 9.

Reproducible: fixed seed, values written with 12 significant digits (float
sums differ in the last bits between CPUs), output in this repository
(experiments/results/sybil-10x-g1.5.json). CI regenerates the file and fails
if it differs by a single byte.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from core.engine.grover_engine import GroverConfig, GroverEngine, GroverState

SEED = 20260709
N_HONEST, N_SYBIL, ROUNDS = 100, 1000, 30
RESULTS = Path(__file__).resolve().parent.parent / "experiments" / "results" / "sybil-10x-g1.5.json"


def initial_state(
    engine: GroverEngine, n_honest: int = N_HONEST, n_sybil: int = N_SYBIL
) -> GroverState:
    n = n_honest + n_sybil
    probs = np.vstack([np.tile([0.9, 0.1], (n_honest, 1)), np.tile([0.1, 0.9], (n_sybil, 1))])
    scores = np.vstack([np.tile([1.0, 0.0], (n_honest, 1)), np.tile([0.0, 1.0], (n_sybil, 1))])
    stake = np.concatenate([np.ones(n_honest), np.full(n_sybil, 0.01)])
    return GroverState(probs=probs, scores=scores, stake=stake, neighbors=engine.gossip_graph(n))


def run_simulation(seed: int = SEED) -> List[Dict[str, float]]:
    engine = GroverEngine(GroverConfig(eta=0.2, gamma=1.5, epsilon=0.1, fanout=10), seed=seed)
    honest = np.arange(N_HONEST + N_SYBIL) < N_HONEST
    return engine.run(initial_state(engine), ROUNDS, honest=honest)


def rounded(history: List[Dict[str, float]], digits: int = 12) -> List[Dict[str, float]]:
    """12 significant digits: float sums differ in the last bits between CPUs and
    BLAS builds (macOS vs. Linux); this keeps the file identical across platforms."""
    return [
        {k: (float(f"{v:.{digits}g}") if isinstance(v, float) else v) for k, v in row.items()}
        for row in history
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args()

    history = run_simulation(a.seed)
    final, r9 = history[-1], history[8]
    checks = [
        ("Confidence(true) > 0.90", final["confidence_true"] > 0.90, final["confidence_true"]),
        ("Confidence(false) < 0.10", final["confidence_false"] < 0.10, final["confidence_false"]),
        ("Sybil stake share at round 9 < 5%", r9["sybil_fraction"] < 0.05, r9["sybil_fraction"]),
    ]
    for name, ok, value in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {value:.4f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rounded(history), indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {a.out.relative_to(Path.cwd()) if a.out.is_relative_to(Path.cwd()) else a.out} (seed {a.seed})"
    )
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
