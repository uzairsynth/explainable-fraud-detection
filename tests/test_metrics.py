import numpy as np

from fraud.metrics import best_f1_threshold, evaluate, slice_report


def test_best_threshold_separates_clean_scores():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    threshold = best_f1_threshold(y, scores)
    assert 0.3 < threshold <= 0.8


def test_evaluate_counts_and_fraud_value():
    y = np.array([1, 1, 0, 0, 1])
    scores = np.array([0.9, 0.2, 0.7, 0.1, 0.6])
    amounts = np.array([100.0, 50.0, 10.0, 10.0, 50.0])
    m = evaluate(y, scores, threshold=0.5, amounts=amounts)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["precision"] == 2 / 3
    assert m["recall"] == 2 / 3
    assert m["fraud_amount_caught_pct"] == 150 / 200


def test_slice_report_covers_every_transaction():
    rng = np.random.default_rng(0)
    amounts = rng.uniform(0, 1000, 500)
    y = rng.integers(0, 2, 500)
    rows = slice_report(y, rng.random(500), 0.5, amounts)
    assert sum(r["transactions"] for r in rows) == 500
