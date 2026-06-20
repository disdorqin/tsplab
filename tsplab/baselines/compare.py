"""Run all baselines and compare with your model."""

from __future__ import annotations

import numpy as np

from tsplab.baselines.naive import NaiveBaseline, SeasonalNaive, DriftBaseline
from tsplab.baselines.linear import DLinear
from tsplab.baselines.tree import TreeBaseline


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    return 100.0 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)
    )


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def run_all_baselines(
    y_train: np.ndarray,
    y_test: np.ndarray,
    horizon: int | None = None,
    lookback: int = 24,
    seasonal_length: int | None = None,
) -> dict[str, dict[str, float]]:
    """Run all baseline models and return their metrics.

    Parameters
    ----------
    y_train : training time series
    y_test : test time series (ground truth)
    horizon : forecast horizon (defaults to len(y_test))
    lookback : lookback window for DLinear and Tree
    seasonal_length : season period for SeasonalNaive

    Returns
    -------
    dict: {model_name: {"mae": ..., "smape": ..., "rmse": ..., "predictions": ...}}
    """
    if horizon is None:
        horizon = len(y_test)

    if seasonal_length is None:
        seasonal_length = lookback

    y_train = np.asarray(y_train, dtype=float)
    y_test = np.asarray(y_test, dtype=float)

    results: dict[str, dict] = {}

    # 1. Naive (Persistence)
    naive = NaiveBaseline()
    naive.fit(y_train)
    pred = naive.predict(horizon)
    results["Naive"] = {
        "mae": _mae(y_test[:horizon], pred),
        "smape": _smape(y_test[:horizon], pred),
        "rmse": _rmse(y_test[:horizon], pred),
        "predictions": pred,
    }

    # 2. Seasonal Naive
    snaive = SeasonalNaive(seasonal_length=seasonal_length)
    snaive.fit(y_train)
    pred = snaive.predict(horizon)
    results["SeasonalNaive"] = {
        "mae": _mae(y_test[:horizon], pred),
        "smape": _smape(y_test[:horizon], pred),
        "rmse": _rmse(y_test[:horizon], pred),
        "predictions": pred,
    }

    # 3. Drift
    drift = DriftBaseline()
    drift.fit(y_train)
    pred = drift.predict(horizon)
    results["Drift"] = {
        "mae": _mae(y_test[:horizon], pred),
        "smape": _smape(y_test[:horizon], pred),
        "rmse": _rmse(y_test[:horizon], pred),
        "predictions": pred,
    }

    # 4. DLinear (needs windowed data)
    try:
        if len(y_train) > lookback + horizon:
            X, Y = _make_windows(y_train, lookback, horizon)
            dlinear = DLinear(lookback=lookback)
            dlinear.fit(X, Y)

            # Predict: use last lookback window
            X_test = y_train[-lookback:].reshape(1, -1)
            pred = dlinear.predict(X_test)[0]
            results["DLinear"] = {
                "mae": _mae(y_test[:horizon], pred),
                "smape": _smape(y_test[:horizon], pred),
                "rmse": _rmse(y_test[:horizon], pred),
                "predictions": pred,
            }
    except Exception as e:
        results["DLinear"] = {"error": str(e)}

    # 5. Decision Tree
    try:
        tree = TreeBaseline(tree_type="decision_tree")
        tree.fit(y_train, lookback=lookback)
        pred = tree.predict(y_train, horizon)
        results["DecisionTree"] = {
            "mae": _mae(y_test[:horizon], pred),
            "smape": _smape(y_test[:horizon], pred),
            "rmse": _rmse(y_test[:horizon], pred),
            "predictions": pred,
        }
    except Exception as e:
        results["DecisionTree"] = {"error": str(e)}

    return results


def _make_windows(y: np.ndarray, lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Create (X, Y) sliding windows from 1D series."""
    n = len(y)
    X, Y = [], []
    for i in range(n - lookback - horizon + 1):
        X.append(y[i:i + lookback])
        Y.append(y[i + lookback:i + lookback + horizon])
    return np.array(X), np.array(Y)


def compare_with_baselines(
    model_metrics: dict[str, float],
    baseline_results: dict[str, dict],
    metric: str = "mae",
) -> None:
    """Compare your model's metrics with baselines and print warnings.

    Parameters
    ----------
    model_metrics : {"mae": 0.32, "smape": 12.5, "rmse": 0.45}
    baseline_results : output of run_all_baselines()
    metric : which metric to compare ("mae", "smape", "rmse")
    """
    print("=" * 65)
    print(f"  TSPLab Baseline Comparison (metric: {metric.upper()})")
    print("=" * 65)
    print()

    # Collect all scores
    scores = []
    for name, result in baseline_results.items():
        if metric in result:
            scores.append((name, result[metric]))

    model_score = model_metrics.get(metric, None)
    if model_score is not None:
        scores.append(("Your Model", model_score))

    # Sort (lower is better)
    scores.sort(key=lambda x: x[1])

    # Print leaderboard
    print(f"  {'Rank':<6} {'Model':<20} {metric.upper():<12} {'Status'}")
    print(f"  {'-'*6} {'-'*20} {'-'*12} {'-'*20}")

    for rank, (name, score) in enumerate(scores, 1):
        if name == "Your Model":
            # Check if worse than any baseline
            worse_than = []
            for bname, bscore in scores:
                if bname != "Your Model" and bscore < model_score:
                    worse_than.append(bname)

            if worse_than:
                if any(b in ["Naive", "Drift"] for b in worse_than):
                    status = "[X] WORSE THAN NAIVE!"
                elif "DLinear" in worse_than:
                    status = "[!] Worse than DLinear"
                else:
                    status = f"[!] Worse than: {', '.join(worse_than)}"
            else:
                status = "[OK] Best model"
            print(f"  {rank:<6} {name:<20} {score:<12.4f} {status}")
        else:
            print(f"  {rank:<6} {name:<20} {score:<12.4f}")

    print()
    if model_score is not None:
        # Find rank of your model
        model_rank = next(i for i, (n, _) in enumerate(scores, 1) if n == "Your Model")
        if model_rank == 1:
            print("  [OK] Your model beats all baselines. Good work!")
        elif model_rank <= 3:
            print(f"  [~] Your model ranks #{model_rank}. Room for improvement.")
        else:
            print(f"  [X] Your model ranks #{model_rank} out of {len(scores)}.")
            print("      Consider checking for data leakage or model bugs.")
    print("=" * 65)
