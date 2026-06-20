"""Tests for tsplab.baselines module."""

import numpy as np
from tsplab.baselines import (
    NaiveBaseline, SeasonalNaive, DriftBaseline,
    DLinear, TreeBaseline, run_all_baselines, compare_with_baselines,
)


def test_naive_baseline():
    """Naive baseline should predict last value."""
    y = np.array([1, 2, 3, 4, 5], dtype=float)
    model = NaiveBaseline()
    model.fit(y)
    pred = model.predict(3)
    assert np.allclose(pred, [5, 5, 5])


def test_seasonal_naive():
    """Seasonal naive should repeat pattern."""
    y = np.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=float)
    model = SeasonalNaive(seasonal_length=4)
    model.fit(y)
    pred = model.predict(4)
    assert np.allclose(pred, [1, 2, 3, 4])


def test_drift_baseline():
    """Drift should extrapolate trend."""
    y = np.array([0, 1, 2, 3, 4], dtype=float)
    model = DriftBaseline()
    model.fit(y)
    pred = model.predict(2)
    # Slope = 1, last = 4, so pred = [5, 6]
    assert np.allclose(pred, [5, 6])


def test_tree_baseline():
    """Tree should learn simple pattern."""
    y = np.sin(np.arange(100) * 0.1) + 1.0
    model = TreeBaseline()
    model.fit(y, lookback=10)
    pred = model.predict(y, horizon=5)
    assert len(pred) == 5
    assert np.all(np.isfinite(pred))


def test_dlinear():
    """DLinear should fit and predict."""
    np.random.seed(42)
    y = np.cumsum(np.random.randn(200)) + 10
    lookback, horizon = 24, 1
    n = len(y)
    X = np.array([y[i:i+lookback] for i in range(n - lookback - horizon)])
    Y = np.array([y[i+lookback:i+lookback+horizon] for i in range(n - lookback - horizon)])

    model = DLinear(lookback=lookback)
    model.fit(X[:100], Y[:100])
    pred = model.predict(X[100:101])
    assert pred.shape == (1, horizon)
    assert np.all(np.isfinite(pred))


def test_run_all_baselines():
    """run_all_baselines should return results for each model."""
    np.random.seed(42)
    y_train = np.sin(np.arange(300) * 0.1) + np.random.randn(300) * 0.1
    y_test = np.sin(np.arange(300, 350) * 0.1) + np.random.randn(50) * 0.1
    results = run_all_baselines(y_train, y_test, horizon=50, lookback=24, seasonal_length=24)

    assert "Naive" in results
    assert "SeasonalNaive" in results
    assert "Drift" in results
    assert "DecisionTree" in results

    for name, result in results.items():
        if "error" not in result:
            assert "mae" in result
            assert result["mae"] >= 0


def test_compare_with_baselines():
    """compare_with_baselines should print comparison."""
    np.random.seed(42)
    y_train = np.sin(np.arange(300) * 0.1)
    y_test = np.sin(np.arange(300, 350) * 0.1)
    results = run_all_baselines(y_train, y_test, horizon=50, lookback=24)

    model_metrics = {"mae": 0.1, "smape": 5.0, "rmse": 0.15}
    # Should not raise
    compare_with_baselines(model_metrics, results, metric="mae")
