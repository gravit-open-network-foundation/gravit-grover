"""GroverEngine: equivalence with the library MWU, reproducibility, Sybil resistance."""

import numpy as np
import pytest

from core.engine.grover_engine import GroverConfig, GroverEngine, GroverState
from simulations.simulate_sybil import SEED, initial_state, run_simulation


def test_local_update_equals_library_multiplicative_weights():
    engine = GroverEngine(seed=1)
    rng = np.random.default_rng(7)
    probs = rng.dirichlet([1, 1, 1], size=20)
    scores = rng.random((20, 3))
    vectorised = engine.local_update(probs, scores)
    for i in range(20):
        expected = engine.mwu.update(probs[i], scores[i])
        np.testing.assert_allclose(vectorised[i], expected, rtol=1e-12)


def test_mix_is_convex_and_keeps_rows_normalised():
    engine = GroverEngine(GroverConfig(fanout=3), seed=2)
    rng = np.random.default_rng(3)
    probs = rng.dirichlet([1, 1], size=10)
    stake = rng.random(10) + 0.1
    mixed = engine.mix(probs, stake, engine.gossip_graph(10))
    np.testing.assert_allclose(mixed.sum(axis=1), 1.0)
    assert (mixed >= 0).all()


def test_gossip_graph_has_no_self_loops_or_duplicates():
    neighbors = GroverEngine(GroverConfig(fanout=5), seed=4).gossip_graph(30)
    for i, row in enumerate(neighbors):
        assert i not in row
        assert len(set(row)) == 5


def test_fanout_must_be_smaller_than_population():
    with pytest.raises(ValueError):
        GroverEngine(GroverConfig(fanout=5), seed=0).gossip_graph(5)


def test_same_seed_same_history_different_seed_different_graph():
    a = run_simulation(seed=SEED)
    b = run_simulation(seed=SEED)
    c = run_simulation(seed=SEED + 1)
    assert a == b
    assert a != c


def test_sybil_10x_success_criteria():
    history = run_simulation(seed=SEED)
    assert history[-1]["confidence_true"] > 0.90
    assert history[-1]["confidence_false"] < 0.10
    assert history[8]["sybil_fraction"] < 0.05  # round 9


def test_amplification_shifts_stake_towards_consensus():
    engine = GroverEngine(seed=5)
    state = initial_state(engine, n_honest=10, n_sybil=10)
    before = state.stake[:10].sum() / state.stake.sum()
    after_state = engine.step(state)
    after = after_state.stake[:10].sum() / after_state.stake.sum()
    assert after > before
