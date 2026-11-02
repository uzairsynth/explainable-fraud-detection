"""Train a logistic regression baseline and an XGBoost model, then report on the held-out test period.

Usage:
    python -m fraud.train
    python -m fraud.train --data path/to/creditcard.csv
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fraud.data import DEFAULT_PATH, FEATURE_COLUMNS, TARGET, add_features, load_raw, time_split
from fraud.explain import plot_global_importance, shap_values
from fraud.metrics import best_f1_threshold, evaluate, slice_report

ROOT = Path(__file__).resolve().parent.parent


def train_logreg(X, y, seed: int):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
    )
    return model.fit(X, y)


def train_xgb(X_train, y_train, X_val, y_val, seed: int) -> XGBClassifier:
    # Up-weight frauds so the trees do not ignore a 0.2% minority class. sqrt(class ratio) beat both
    # no weighting and the full ratio on validation PR-AUC (scripts/tune_class_weight.py).
    pos_weight = np.sqrt((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    return model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)


def single_transaction_latency_ms(model, X: pd.DataFrame, repeats: int = 300) -> float:
    """Median time to score one transaction, as a real-time payment check would."""
    rows = [X.iloc[[i % len(X)]] for i in range(repeats)]
    timings = []
    for row in rows:
        start = time.perf_counter()
        model.predict_proba(row)
        timings.append(time.perf_counter() - start)
    return float(np.median(timings) * 1000)


def plot_pr_curves(y_true, scores_by_model: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, scores in scores_by_model.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.step(recall, precision, where="post", label=f"{name} (PR-AUC {ap:.3f})")
    ax.set_xlabel("Recall (share of frauds caught)")
    ax.set_ylabel("Precision (share of alerts that are fraud)")
    ax.set_title("Precision-recall on the held-out test period")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(df: pd.DataFrame, out_dir: Path, model_dir: Path, seed: int = 42) -> dict:
    splits = time_split(add_features(df))
    X = {name: part[FEATURE_COLUMNS] for name, part in splits.items()}
    y = {name: part[TARGET].to_numpy() for name, part in splits.items()}
    test_amounts = splits["test"]["Amount"].to_numpy()

    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "dataset": {
            name: {"transactions": len(part), "frauds": int(part[TARGET].sum())}
            for name, part in splits.items()
        },
        "models": {},
    }
    trainers = {
        "logistic_regression": lambda: train_logreg(X["train"], y["train"], seed),
        "xgboost": lambda: train_xgb(X["train"], y["train"], X["val"], y["val"], seed),
    }
    trained, test_scores = {}, {}
    for name, fit in trainers.items():
        start = time.perf_counter()
        model = fit()
        train_seconds = time.perf_counter() - start
        # The threshold is chosen on validation only, so the test numbers stay honest.
        threshold = best_f1_threshold(y["val"], model.predict_proba(X["val"])[:, 1])
        scores = model.predict_proba(X["test"])[:, 1]
        metrics = evaluate(y["test"], scores, threshold, test_amounts)
        metrics["train_seconds"] = round(train_seconds, 2)
        metrics["latency_ms"] = round(single_transaction_latency_ms(model, X["test"]), 3)
        results["models"][name] = metrics
        trained[name] = model
        test_scores[name] = scores

    xgb_model = trained["xgboost"]
    xgb_threshold = results["models"]["xgboost"]["threshold"]
    results["models"]["xgboost"]["trees_used"] = int(xgb_model.best_iteration + 1)
    results["xgboost_amount_slices"] = slice_report(y["test"], test_scores["xgboost"], xgb_threshold, test_amounts)

    shap_df, _ = shap_values(xgb_model, X["test"])
    plot_global_importance(shap_df, figures / "shap_importance.png")
    plot_pr_curves(y["test"], test_scores, figures / "pr_curve.png")

    xgb_model.save_model(model_dir / "xgboost.json")
    joblib.dump(trained["logistic_regression"], model_dir / "logistic_regression.joblib")
    (model_dir / "config.json").write_text(
        json.dumps({"features": FEATURE_COLUMNS, "xgboost_threshold": xgb_threshold}, indent=2)
    )
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


def print_summary(results: dict) -> None:
    for split, info in results["dataset"].items():
        print(f"{split:>5}: {info['transactions']:>7,} transactions, {info['frauds']:>4} frauds")
    print(f"\n{'model':<20}{'PR-AUC':>8}{'ROC-AUC':>9}{'prec':>7}{'recall':>8}{'F1':>7}{'FP':>6}{'FN':>5}{'value caught':>14}")
    for name, m in results["models"].items():
        print(
            f"{name:<20}{m['pr_auc']:>8.3f}{m['roc_auc']:>9.3f}{m['precision']:>7.3f}{m['recall']:>8.3f}"
            f"{m['f1']:>7.3f}{m['fp']:>6}{m['fn']:>5}{m['fraud_amount_caught_pct']:>13.1%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run(load_raw(args.data), args.out, args.models, args.seed)
    print_summary(results)
    print(f"\nSaved metrics and figures to {args.out}")


if __name__ == "__main__":
    main()
