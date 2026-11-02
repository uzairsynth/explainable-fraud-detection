import numpy as np
import pytest

from fraud.data import add_features, make_synthetic, time_split, validate


def test_add_features():
    df = make_synthetic(n=10)
    df.loc[0, ["Time", "Amount"]] = [3600 * 25 + 5, 0.0]
    out = add_features(df)
    assert out.loc[0, "hour"] == 1
    assert out.loc[0, "log_amount"] == 0.0
    assert np.allclose(out["log_amount"], np.log1p(df["Amount"]))


def test_time_split_is_ordered_and_complete():
    df = make_synthetic(n=1000)
    splits = time_split(df, val_frac=0.2, test_frac=0.2)
    assert [len(splits[k]) for k in ("train", "val", "test")] == [600, 200, 200]
    assert splits["train"]["Time"].max() <= splits["val"]["Time"].min()
    assert splits["val"]["Time"].max() <= splits["test"]["Time"].min()


def test_validate_rejects_bad_files():
    df = make_synthetic(n=50)
    with pytest.raises(ValueError, match="Missing columns"):
        validate(df.drop(columns="V14"))
    bad = df.copy()
    bad.loc[0, "Class"] = 2
    with pytest.raises(ValueError, match="Class"):
        validate(bad)
