"""
cost_matrix.py — Asymmetric misclassification cost scoring

Business rationale:
    A C1 claim routed to STP (Straight-Through Processing) is the worst
    possible outcome in this system. STP means automated settlement with
    no human review. A C1 claim involves reserves >$500K, fatalities,
    permanent injury, or CAT linkage. Routing it to STP means:
      - No senior examiner assignment → reserve is not set correctly
      - Regulatory exposure (state insurance dept. requires timely reserve)
      - Bad faith lawsuit risk if claim is underpaid
      - Potential reinsurance reporting failure

    Contrast with C3 → LLU: a standard claim sent to the Large Loss Unit
    wastes a senior examiner's time (maybe 30 minutes) but causes no
    regulatory or legal harm. The cost is operational inefficiency only.

    Therefore: C1 → STP penalty = 100, C3 → LLU penalty = 10 (10x ratio as required).

    The full matrix encodes severity × routing combinations.
    Values represent relative business cost units (not dollars).
"""

import numpy as np

# Routing order: LLU, LMU, STP, SIU
ROUTING_LABELS = ["LLU", "LMU", "STP", "SIU"]

# Severity tier order: C1, C2, C3
SEVERITY_LABELS = ["C1", "C2", "C3"]

# ── Routing cost matrix (true → predicted) ───────────────────────────────────
# Rows = true routing, Cols = predicted routing
# Diagonal = correct (cost 0)
#
# Cost rationale:
#   C1 misrouted to STP: catastrophic (reserve failure, bad faith) → 100
#   C1 misrouted to LMU: serious but senior LMU examiner will catch it → 20
#   C1 misrouted to SIU: delays claim, but SIU will review and escalate → 15
#   C2 misrouted to STP: litigation claim auto-settled, attorney cost exposure → 40
#   C2 misrouted to LLU: minor inefficiency, LLU examiner escalates down → 5
#   C3 misrouted to LLU: operational waste, 30-min senior examiner time → 10
#   C3 misrouted to SIU: insured flagged as fraud incorrectly, bad experience → 25
#   SIU misrouted to STP: fraud claim auto-settled → 80
#   SIU misrouted to LLU: fraud reviewed by wrong unit, likely caught → 15

# routing_cost_matrix[true_idx][pred_idx]
ROUTING_COST_MATRIX = np.array([
    #  LLU   LMU   STP   SIU   ← predicted
    [   0,    20,  100,   15],  # true=LLU (C1 should go here)
    [   5,     0,   40,   10],  # true=LMU (C2 should go here)
    [  10,     8,    0,   25],  # true=STP (C3 should go here)
    [  15,    10,   80,    0],  # true=SIU
])

# ── Severity tier cost matrix ─────────────────────────────────────────────────
# Misclassifying severity is also penalized separately from routing.
# Rows = true severity, Cols = predicted severity.

SEVERITY_COST_MATRIX = np.array([
    #  C1    C2    C3   ← predicted
    [   0,   30,   75],  # true=C1 — calling C1 a C3 is catastrophic
    [  10,    0,   20],  # true=C2 — downgrading to C3 misses litigation signal
    [   5,    3,    0],  # true=C3 — upgrading wastes resources only
])


def routing_cost(y_true: list[str], y_pred: list[str]) -> float:
    """
    Compute mean routing misclassification cost over a set of predictions.

    Args:
        y_true: list of true routing labels
        y_pred: list of predicted routing labels

    Returns:
        Mean cost per prediction (lower is better)
    """
    label_to_idx = {label: i for i, label in enumerate(ROUTING_LABELS)}
    costs = []
    for true, pred in zip(y_true, y_pred):
        if true not in label_to_idx or pred not in label_to_idx:
            costs.append(0)  # unknown label — skip
            continue
        costs.append(ROUTING_COST_MATRIX[label_to_idx[true]][label_to_idx[pred]])
    return float(np.mean(costs))


def severity_cost(y_true: list[str], y_pred: list[str]) -> float:
    """Compute mean severity misclassification cost."""
    label_to_idx = {label: i for i, label in enumerate(SEVERITY_LABELS)}
    costs = []
    for true, pred in zip(y_true, y_pred):
        if true not in label_to_idx or pred not in label_to_idx:
            costs.append(0)
            continue
        costs.append(SEVERITY_COST_MATRIX[label_to_idx[true]][label_to_idx[pred]])
    return float(np.mean(costs))


def perfect_cost_baseline() -> dict:
    """Returns cost for a perfect predictor (all zeros, for reference)."""
    return {"routing_cost": 0.0, "severity_cost": 0.0}


def random_cost_baseline(label_dist: dict[str, float]) -> dict:
    """
    Estimate expected cost of a random predictor given label distribution.
    Useful as a sanity check lower bound.
    """
    labels = list(label_dist.keys())
    probs = list(label_dist.values())
    n = 1000
    y_true = random_choices(labels, probs, n)
    y_pred = random_choices(labels, probs, n)
    return {
        "routing_cost": routing_cost(y_true, y_pred),
        "severity_cost": severity_cost(y_true, y_pred),
    }


def random_choices(labels, probs, n):
    import random
    return random.choices(labels, weights=probs, k=n)


if __name__ == "__main__":
    # Quick sanity check
    print("Cost matrix sanity check:")
    print(f"  C1→STP: {ROUTING_COST_MATRIX[0][2]} (should be 100)")
    print(f"  C3→LLU: {ROUTING_COST_MATRIX[2][0]} (should be 10)")
    print(f"  Ratio:  {ROUTING_COST_MATRIX[0][2] / ROUTING_COST_MATRIX[2][0]}x (spec requires ≥10x)")
    assert ROUTING_COST_MATRIX[0][2] / ROUTING_COST_MATRIX[2][0] >= 10, "10x ratio violated!"
    print("  ✓ 10x constraint satisfied")
