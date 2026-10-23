import numpy as np
import pandas as pd
import pytest

import analytics as A
import ml


@pytest.fixture(scope="module")
def ds():
    rng = np.random.RandomState(0)
    n = 500
    df = pd.DataFrame({"id": range(n), "tenure": rng.randint(1, 72, n), "monthly": rng.normal(70, 20, n),
                       "contract": rng.choice(["Month", "Year"], n)})
    p = 1 / (1 + np.exp(-(-0.06 * df.tenure + 1.5 * (df.contract == "Month") + 1)))
    df["Churn"] = np.where(rng.rand(n) < p, "Yes", "No")
    df["revenue"] = df.monthly * df.tenure
    return A.Dataset(df, "t.csv", "t.csv")


def test_classification_beats_baseline(ds):
    r = ml.train(ds, "Churn")
    assert r["task"] == "classification"
    assert r["importance"][0][0] in ("tenure", "contract")
    assert r["metrics"]["accuracy"] >= r["baseline"]["metrics"]["accuracy"] - 0.02


def test_regression_and_whatif(ds):
    r = ml.train(ds, "revenue")
    assert r["task"] == "regression" and r["metrics"]["r2"] > 0.8
    low, high = ml.predict(r, {"tenure": 5})["value"], ml.predict(r, {"tenure": 65})["value"]
    assert high > low


def test_cluster_and_anomaly(ds):
    c = ml.cluster(ds, k=3)
    assert c["k"] == 3 and sum(g["size"] for g in c["groups"]) == c["rows"]
    assert len(ml.anomalies(ds)["rows"]) == 10


def test_tests_pick_the_right_test(ds):
    assert ml.test(ds, "tenure", "monthly")["kind"] == "correlation"
    assert ml.test(ds, "monthly", "contract")["kind"] == "ttest"
    assert ml.test(ds, "contract", "Churn")["kind"] == "chi2"


def test_unknown_target_is_a_friendly_error(ds):
    with pytest.raises(ml.MLError):
        ml.train(ds, "nope")
