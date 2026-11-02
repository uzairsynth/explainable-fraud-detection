"""Choose XGBoost's fraud class weight using the validation period only (the test period is never touched).

Usage:
    python scripts/tune_class_weight.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fraud.data import FEATURE_COLUMNS, TARGET, add_features, load_raw, time_split  # noqa: E402


def main() -> None:
    splits = time_split(add_features(load_raw()))
    X_train, y_train = splits["train"][FEATURE_COLUMNS], splits["train"][TARGET].to_numpy()
    X_val, y_val = splits["val"][FEATURE_COLUMNS], splits["val"][TARGET].to_numpy()
    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    results = {}
    for name, weight in {"none (1)": 1.0, "sqrt(ratio)": float(np.sqrt(ratio)), "ratio": float(ratio)}.items():
        model = XGBClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=weight, eval_metric="aucpr", early_stopping_rounds=50,
            tree_method="hist", random_state=42, n_jobs=-1,
        ).fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_pr_auc = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
        results[name] = {"scale_pos_weight": round(weight, 1), "val_pr_auc": round(float(val_pr_auc), 4)}
        print(f"{name:<12} weight {weight:>7.1f}  validation PR-AUC {val_pr_auc:.4f}")

    out = ROOT / "reports" / "class_weight_tuning.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
