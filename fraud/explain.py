"""Per-transaction explanations with SHAP values.

XGBoost computes exact TreeSHAP values itself (pred_contribs=True), so no extra
library is needed. Values are in log-odds: positive pushes towards fraud.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


def shap_values(model: xgb.XGBClassifier, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return one SHAP value per feature per row, plus the base value (last column)."""
    booster = model.get_booster()
    iteration_range = (0, model.best_iteration + 1) if hasattr(model, "best_iteration") else (0, 0)
    contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True, iteration_range=iteration_range)
    return pd.DataFrame(contribs[:, :-1], columns=X.columns, index=X.index), contribs[:, -1]


def top_reasons(row_shap: pd.Series, k: int = 5) -> list[tuple[str, float]]:
    """The k features that moved this transaction's score the most, largest first."""
    order = row_shap.abs().sort_values(ascending=False).index[:k]
    return [(feature, float(row_shap[feature])) for feature in order]


def plot_global_importance(shap_df: pd.DataFrame, path, top: int = 15) -> None:
    importance = shap_df.abs().mean().sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(importance.index, importance.values, color="#2a6f97")
    ax.set_xlabel("Mean |SHAP value| (log-odds)")
    ax.set_title("Which features drive fraud scores (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
