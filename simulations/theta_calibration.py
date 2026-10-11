#!/usr/bin/env python3
"""
theta_critical calibration trace (sybil-10x-g1.5)
=================================================

Regenerates, from the real ``GroverEngine``, the 3000-record trace used to
calibrate the gateway threshold ``theta_critical`` (GEVP: a gateway MUST reject
an action whose basis has confidence below theta_critical).

Model (all assumptions are stated here and fixed before the run):

* 100 runs, 30 claims per run = 3000 records. Runs 0-49 record the network's
  confidence in a *true* claim (label ``honest``); runs 50-99 record its
  confidence in the *false* claim pushed by the Sybils (label ``sybil``).
* Each run: 100 honest nodes (stake 1.0) and 1000 Sybil nodes (stake 0.01),
  i.e. 10x Sybil amplification by node count; one fixed gossip graph per run.
* Each claim has a difficulty q ~ Uniform(0.6, 1.0): every honest node sees
  correct local evidence with probability q, wrong evidence otherwise. Sybil
  nodes always report evidence for the false claim.
* All nodes start undecided (belief 0.5 / 0.5); the gateway reads the
  stake-weighted network confidence after 30 rounds (as in simulate_sybil.py).
* Protocol parameters: eta=0.2, gamma=1.5, epsilon=0.1, fanout=10.

Outputs (in experiments/results/):

* ``theta-calibration.jsonl`` - one record per claim, with the SHA-256 of its
  canonical JSON (``record_hash``);
* ``theta-calibration-summary.json`` - per-label statistics, TPR/FPR at
  theta = 0.73, the full ROC curve, ROC AUC, the thresholds maximising Youden's
  J (as a range), and the Merkle root over all record hashes (recomputable by anyone).

Reproducible: fixed seed, numbers written with 12 significant digits; CI
regenerates both files and fails if they differ by a single byte.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Dict, List

import numpy as np

from core.engine.grover_engine import GroverConfig, GroverEngine, GroverState

SEED = 20261011
N_HONEST, N_SYBIL, ROUNDS = 100, 1000, 30
RUNS, CLAIMS = 100, 30
Q_LOW, Q_HIGH = 0.6, 1.0
THETA = 0.73
NAME = "sybil-10x-g1.5"
PARAMS = {"eta": 0.2, "gamma": 1.5, "eps": 0.1, "fanout": 10, "sybil_factor": 10, "rounds": ROUNDS}
RESULTS = Path(__file__).resolve().parent.parent / "experiments" / "results"


def sig(x: float, digits: int = 12) -> float:
    """12 significant digits, identical on macOS and Linux."""
    return float(f"{x:.{digits}g}")


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def merkle_root(leaves: List[str]) -> str:
    """Binary SHA-256 Merkle tree over hex leaf hashes; an odd node is paired with itself."""
    level = [bytes.fromhex(h) for h in leaves]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
    return level[0].hex()


def claim_confidence(engine: GroverEngine, neighbors: np.ndarray, q: float) -> np.ndarray:
    """Network confidence [true claim, false claim] after ROUNDS rounds."""
    honest_ok = engine.rng.random(N_HONEST) < q
    scores = np.vstack(
        [
            np.where(honest_ok[:, None], [1.0, 0.0], [0.0, 1.0]),
            np.tile([0.0, 1.0], (N_SYBIL, 1)),
        ]
    )
    stake = np.concatenate([np.ones(N_HONEST), np.full(N_SYBIL, 0.01)])
    state = GroverState(
        probs=np.full((N_HONEST + N_SYBIL, 2), 0.5), scores=scores, stake=stake, neighbors=neighbors
    )
    for _ in range(ROUNDS):
        state = engine.step(state)
    return engine.confidence(state)


def generate(seed: int = SEED) -> List[Dict]:
    records = []
    for run in range(RUNS):
        label = "honest" if run < RUNS // 2 else "sybil"
        engine = GroverEngine(
            GroverConfig(eta=0.2, gamma=1.5, epsilon=0.1, fanout=10), seed=[seed, run]
        )
        neighbors = engine.gossip_graph(N_HONEST + N_SYBIL)
        for c in range(CLAIMS):
            q = float(engine.rng.uniform(Q_LOW, Q_HIGH))
            conf = claim_confidence(engine, neighbors, q)
            value = sig(float(conf[0] if label == "honest" else conf[1]))
            rec = {
                "run_id": f"{NAME}-run-{run:03d}",
                "claim_id": f"{NAME}-run-{run:03d}-claim-{c:02d}",
                "label": label,
                "difficulty_q": sig(q),
                "confidence": value,
                "passes_theta": value >= THETA,
            }
            rec["record_hash"] = sha256(canonical(rec))
            records.append(rec)
    return records


def roc(records: List[Dict]) -> Dict:
    pos = np.array([r["confidence"] for r in records if r["label"] == "honest"])
    neg = np.array([r["confidence"] for r in records if r["label"] == "sybil"])
    thresholds = np.round(np.arange(0.0, 1.0001, 0.01), 2)
    curve = [
        {
            "theta": float(t),
            "tpr": sig(float((pos >= t).mean())),
            "fpr": sig(float((neg >= t).mean())),
        }
        for t in thresholds
    ]
    # AUC = P(conf_honest > conf_sybil) + 0.5 P(tie), exact over all pairs
    neg_sorted = np.sort(neg)
    less = np.searchsorted(neg_sorted, pos, side="left")
    leq = np.searchsorted(neg_sorted, pos, side="right")
    auc = float((less + 0.5 * (leq - less)).sum() / (len(pos) * len(neg)))
    # Youden's J = TPR - FPR; report every threshold that reaches the maximum, not one picked point
    j = [round(p["tpr"] - p["fpr"], 9) for p in curve]
    best = [p["theta"] for p, v in zip(curve, j) if v == max(j)]
    return {
        "curve": curve,
        "auc": sig(auc),
        "youden": {"max_j": sig(max(j)), "theta_min": best[0], "theta_max": best[-1]},
    }


def stats(values: List[float]) -> Dict:
    return {
        "n": len(values),
        "mean": sig(statistics.mean(values)),
        "sd": sig(statistics.pstdev(values)),
        "min": sig(min(values)),
        "max": sig(max(values)),
    }


def summarize(records: List[Dict], seed: int) -> Dict:
    honest = [r for r in records if r["label"] == "honest"]
    sybil = [r for r in records if r["label"] == "sybil"]
    tp = sum(r["passes_theta"] for r in honest)
    fp = sum(r["passes_theta"] for r in sybil)
    r = roc(records)
    return {
        "trace": NAME,
        "generator": "simulations/theta_calibration.py",
        "seed": seed,
        "params": PARAMS,
        "model": {
            "runs": RUNS,
            "claims_per_run": CLAIMS,
            "honest_nodes": N_HONEST,
            "sybil_nodes": N_SYBIL,
            "stake_honest": 1.0,
            "stake_sybil": 0.01,
            "difficulty_q": [Q_LOW, Q_HIGH],
            "initial_belief": 0.5,
        },
        "honest": stats([x["confidence"] for x in honest]),
        "sybil": stats([x["confidence"] for x in sybil]),
        "at_theta": {
            "theta": THETA,
            "tp": tp,
            "fn": len(honest) - tp,
            "fp": fp,
            "tn": len(sybil) - fp,
            "tpr": sig(tp / len(honest)),
            "fpr": sig(fp / len(sybil)),
        },
        "roc_auc": r["auc"],
        "youden": r["youden"],
        "roc_curve": r["curve"],
        "merkle_root": "sha256:" + merkle_root([x["record_hash"] for x in records]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args()

    records = generate(a.seed)
    summary = summarize(records, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "theta-calibration.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8"
    )
    (a.out / "theta-calibration-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    t = summary["at_theta"]
    print(f"honest: mean {summary['honest']['mean']:.3f} sd {summary['honest']['sd']:.3f}")
    print(f"sybil:  mean {summary['sybil']['mean']:.3f} sd {summary['sybil']['sd']:.3f}")
    print(
        f"theta {THETA}: TPR {t['tp']}/{t['tp'] + t['fn']} ({100 * t['tpr']:.1f}%), "
        f"FPR {t['fp']}/{t['fp'] + t['tn']} ({100 * t['fpr']:.1f}%)"
    )
    print(
        f"ROC AUC {summary['roc_auc']:.4f}; max Youden J {summary['youden']['max_j']:.3f} for theta in [{summary['youden']['theta_min']}, {summary['youden']['theta_max']}]"
    )
    print(f"merkle_root {summary['merkle_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
