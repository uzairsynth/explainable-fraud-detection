"""End-to-end run on synthetic data, so CI does not need the 150 MB dataset."""

import json

import numpy as np
import xgboost as xgb

from fraud.data import FEATURE_COLUMNS, add_features, make_synthetic
from fraud.explain import shap_values, top_reasons
from fraud.train import run


def test_pipeline_end_to_end(tmp_path):
    results = run(make_synthetic(n=6000), tmp_path / "reports", tmp_path / "models")

    assert set(results["models"]) == {"logistic_regression", "xgboost"}
    assert results["models"]["xgboost"]["pr_auc"] > 0.5
    for name in ("metrics.json", "figures/pr_curve.png", "figures/shap_importance.png"):
        assert (tmp_path / "reports" / name).exists()
    config = json.loads((tmp_path / "models" / "config.json").read_text())
    assert config["features"] == FEATURE_COLUMNS


def test_shap_values_add_up_to_model_output(tmp_path):
    run(make_synthetic(n=3000), tmp_path / "reports", tmp_path / "models")
    model = xgb.XGBClassifier()
    model.load_model(tmp_path / "models" / "xgboost.json")
    X = add_features(make_synthetic(n=20, seed=1))[FEATURE_COLUMNS]

    shap_df, base = shap_values(model, X)
    margin = model.predict(X, output_margin=True)
    assert np.allclose(shap_df.sum(axis=1) + base, margin, atol=1e-4)

    reasons = top_reasons(shap_df.iloc[0], k=3)
    assert len(reasons) == 3
    assert abs(reasons[0][1]) >= abs(reasons[1][1]) >= abs(reasons[2][1])
