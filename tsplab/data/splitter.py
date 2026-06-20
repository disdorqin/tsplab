"""Time-series-safe data splitting and windowing utilities.

These functions ensure temporal ordering is respected — no random shuffling,
no future data leaking into training windows.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def time_series_split(
    data: np.ndarray,
    test_ratio: float = 0.2,
    gap: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split time series data chronologically.

    Parameters
    ----------
    data : 1D or 2D time series data
    test_ratio : fraction of data for testing
    gap : number of samples to skip between train and test (prevents leakage)

    Returns
    -------
    train, test : split data
    """
    n = len(data)
    split_idx = int(n * (1 - test_ratio))
    train = data[:split_idx - gap]
    test = data[split_idx:]
    return train, test


def create_windows(
    data: np.ndarray,
    lookback: int,
    horizon: int,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows from a time series.

    IMPORTANT: Call this AFTER splitting, not before.
    Calling it on the full dataset before splitting causes leakage.

    Parameters
    ----------
    data : 1D time series
    lookback : input window size
    horizon : forecast horizon
    step : sliding step (1 = no overlap between consecutive targets)

    Returns
    -------
    X : shape (n_windows, lookback)
    Y : shape (n_windows, horizon)
    """
    data = np.asarray(data)
    n = len(data)
    n_windows = (n - lookback - horizon + 1) // step

    X = np.zeros((n_windows, lookback))
    Y = np.zeros((n_windows, horizon))

    for i in range(n_windows):
        start = i * step
        X[i] = data[start:start + lookback]
        Y[i] = data[start + lookback:start + lookback + horizon]

    return X, Y


def create_cv_splits(
    n_samples: int,
    n_windows: int = 5,
    test_size: int | None = None,
    gap: int = 0,
    strategy: str = "expanding",
) -> list[Tuple[np.ndarray, np.ndarray]]:
    """Create time series cross-validation splits.

    Parameters
    ----------
    n_samples : total number of samples
    n_windows : number of CV folds
    test_size : samples per test fold (auto if None)
    gap : samples to skip between train and test
    strategy : "expanding" (train grows) or "rolling" (train slides)

    Returns
    -------
    list of (train_indices, test_indices) tuples
    """
    if test_size is None:
        test_size = max(n_samples // (n_windows + 2), 1)

    splits = []
    for i in range(n_windows):
        test_start = n_samples - (n_windows - i) * test_size
        test_end = min(test_start + test_size, n_samples)

        if strategy == "rolling":
            train_start = max(0, test_start - test_size - gap)
        else:  # expanding
            train_start = 0

        train_end = test_start - gap

        if train_end <= train_start:
            continue

        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    return splits
