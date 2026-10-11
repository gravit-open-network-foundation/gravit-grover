import hashlib

from simulations.theta_calibration import canonical, merkle_root, roc, sha256


def test_merkle_root_matches_manual_tree():
    leaves = [sha256(canonical({"i": i})) for i in range(3)]
    h = [bytes.fromhex(x) for x in leaves]
    left = hashlib.sha256(h[0] + h[1]).digest()
    right = hashlib.sha256(h[2] + h[2]).digest()
    assert merkle_root(leaves) == hashlib.sha256(left + right).hexdigest()


def test_roc_on_separable_scores():
    recs = [{"label": "honest", "confidence": 0.9}] * 4 + [
        {"label": "sybil", "confidence": 0.1}
    ] * 4
    r = roc(recs)
    assert r["auc"] == 1.0
    assert r["youden"]["max_j"] == 1.0
    assert r["youden"]["theta_min"] <= 0.73 <= r["youden"]["theta_max"]
