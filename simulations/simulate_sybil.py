#!/usr/bin/env python3
"""
GQRVP 10x Sybil Attack Model Simulation
======================================
This script validates that under VCP parameters (eta=0.2, gamma=1.5, epsilon=0.1),
an honest supermajority of stake-weighted attention converges on the true claim
and decays the influence of a 10x Sybil attack within 30 gossip rounds.
"""

import os
import json
import numpy as np

def run_simulation():
    # Parameters
    eta = 0.2
    gamma = 1.5
    epsilon = 0.1
    n_honest = 100
    n_sybil = 1000
    n_nodes = n_honest + n_sybil
    rounds = 30
    fanout = 10

    print(f"Initializing simulation: {n_honest} honest nodes, {n_sybil} Sybil nodes (10x Sybil).")
    print(f"Parameters: eta={eta}, gamma={gamma}, epsilon={epsilon}")

    # Initial stakes: honest = 1.0, sybil = 0.01
    stake = np.concatenate([np.ones(n_honest) * 1.0, np.ones(n_sybil) * 0.01])

    # Initial probability distributions over [True Claim, False Claim]
    # Honest nodes are confident in True Claim, Sybil nodes are confident in False Claim
    probs = np.zeros((n_nodes, 2))
    probs[:n_honest] = [0.9, 0.1]
    probs[n_honest:] = [0.1, 0.9]

    # Node scoring outputs:
    # Honest nodes score [1.0, 0.0], Sybils score [0.0, 1.0]
    scores = np.zeros((n_nodes, 2))
    scores[:n_honest] = [1.0, 0.0]
    scores[n_honest:] = [0.0, 1.0]

    # Generate random regular-like gossip graph
    neighbors = []
    for i in range(n_nodes):
        choices = list(range(n_nodes))
        choices.remove(i)
        node_neighbors = np.random.choice(choices, fanout, replace=False)
        neighbors.append(node_neighbors)

    history = []

    print("\nStarting simulation rounds...")
    print(f"{'Round':<6} | {'Confidence (True)':<17} | {'Confidence (False)':<18} | {'Sybil Weight Ratio':<18}")
    print("-" * 70)

    for r in range(rounds):
        # 1. Local Multiplicative Weights Update (MWU)
        new_probs = np.zeros_like(probs)
        for i in range(n_nodes):
            mwu_factor = np.exp(eta * scores[i])
            updated = probs[i] * mwu_factor
            new_probs[i] = updated / np.sum(updated)

        # 2. Gossip Mixing Layer
        gossip_probs = np.zeros_like(new_probs)
        for i in range(n_nodes):
            nb_avg = np.mean(new_probs[neighbors[i]], axis=0)
            gossip_probs[i] = (1 - epsilon) * new_probs[i] + epsilon * nb_avg
        probs = gossip_probs

        # 3. Grover Amplification on Stake/Weights
        # Calculate global weighted score for each hypothesis
        total_stake = np.sum(stake)
        weighted_score = np.sum(probs * stake[:, np.newaxis], axis=0) / total_stake
        avg_score = np.mean(weighted_score)

        # Update stake based on Grover amplification exponent
        amp_factors = (weighted_score / (avg_score + 1e-10)) ** gamma
        stake_update = np.sum(probs * amp_factors, axis=1)
        stake = stake * stake_update

        # Track metrics
        total_honest_weight = np.sum(stake[:n_honest])
        total_sybil_weight = np.sum(stake[n_honest:])
        sybil_fraction = total_sybil_weight / (total_honest_weight + total_sybil_weight)

        confidence_true = np.sum(probs[:, 0] * stake) / np.sum(stake)
        confidence_false = np.sum(probs[:, 1] * stake) / np.sum(stake)

        # Print progress
        if (r + 1) % 2 == 0 or r == 0 or r == rounds - 1:
            print(f"{r+1:<6} | {confidence_true:<17.4f} | {confidence_false:<18.4f} | {sybil_fraction:<18.4%}")

        history.append({
            "round": r + 1,
            "confidence_true": float(confidence_true),
            "confidence_false": float(confidence_false),
            "honest_weight": float(total_honest_weight),
            "sybil_weight": float(total_sybil_weight),
            "sybil_fraction": float(sybil_fraction)
        })

    # Validate success criteria
    final_conf_true = history[-1]["confidence_true"]
    final_conf_false = history[-1]["confidence_false"]
    sybil_decay_r9 = history[8]["sybil_fraction"] # Round 9 (1-indexed index 8)

    print("\n" + "="*50)
    print("Simulation Completed. Evaluating Success Criteria:")
    print(f"1. Confidence(True) = {final_conf_true:.4f} (>0.90): {'PASSED' if final_conf_true > 0.90 else 'FAILED'}")
    print(f"2. Confidence(False) = {final_conf_false:.4f} (<0.10): {'PASSED' if final_conf_false < 0.10 else 'FAILED'}")
    print(f"3. Sybil Weight at Round 9 = {sybil_decay_r9:.4%} (<5%): {'PASSED' if sybil_decay_r9 < 0.05 else 'FAILED'}")
    print("="*50)

    # Save to experiments/results/
    out_dir = "/root/gravit/gravit-grover/experiments/results"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "sybil-10x-g1.5.json")
    with open(out_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved results trace to: {out_file}")

if __name__ == "__main__":
    run_simulation()
