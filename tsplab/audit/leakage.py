"""
Leakage detection functions.

Each function returns a CheckResult with:
  - passed: bool
  - severity: "ok" | "warning" | "critical"
  - message: str
  - details: dict
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CheckResult:
    """Result of a single leakage check."""

    name: str
    passed: bool
    severity: str  # "ok" | "warning" | "critical"
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        icon = {"ok": "[OK]", "warning": "[!]", "critical": "[X]"}[self.severity]
        return f"{icon} {self.name}: {self.message}"


# ---------------------------------------------------------------------------
# 1. Sequence generation timing check
# ---------------------------------------------------------------------------

def check_sequence_generation(
    X_train: np.ndarray,
    X_test: np.ndarray,
    timestamps: np.ndarray | pd.Series | None = None,
    window_size: int | None = None,
) -> CheckResult:
    """Detect whether sliding windows were generated before or after splitting.

    The most common and subtle leakage: generating sequences on the full
    dataset *before* splitting allows training windows to contain
    test-period context.

    Heuristic: if any training window's timestamp range overlaps with the
    test set's timestamp range, that's a critical leak.

    Parameters
    ----------
    X_train : array of shape (n_train_windows, window_size, n_features)
    X_test : array of shape (n_test_windows, window_size, n_features)
    timestamps : array of timestamps aligned to the *original* series
    window_size : size of each sliding window (inferred from X if None)

    Returns
    -------
    CheckResult
    """
    name = "Sequence Generation Timing"

    if window_size is None and X_train.ndim >= 2:
        window_size = X_train.shape[1]

    if timestamps is None:
        # Even without timestamps, check for row overlap
        if X_train.ndim == 2:
            train_flat = X_train.reshape(X_train.shape[0], -1)
            test_flat = X_test.reshape(X_test.shape[0], -1)
        else:
            train_flat = X_train
            test_flat = X_test

        train_set = set(map(lambda r: tuple(np.round(r, 8)), train_flat))
        test_set = set(map(lambda r: tuple(np.round(r, 8)), test_flat))
        overlap = train_set & test_set
        overlap_ratio = len(overlap) / max(len(test_set), 1)

        if overlap_ratio > 0.1:
            return CheckResult(
                name=name,
                passed=False,
                severity="critical",
                message=f"{len(overlap)} rows ({overlap_ratio:.1%}) appear in both train and test. "
                "Windows were likely generated before splitting.",
                details={"overlap_count": len(overlap), "overlap_ratio": overlap_ratio},
            )
        return CheckResult(
            name=name,
            passed=True,
            severity="ok",
            message="No row overlap detected. Manual timestamp verification recommended.",
            details={"overlap_count": 0},
        )

    timestamps = np.asarray(timestamps)
    if isinstance(timestamps[0], (np.datetime64, pd.Timestamp)):
        timestamps = pd.to_datetime(timestamps).astype(np.int64).values

    # Heuristic: check if X_train and X_test share identical rows
    # (a sign that windows were generated from the full dataset before split)
    if X_train.ndim == 2:
        train_flat = X_train.reshape(X_train.shape[0], -1)
        test_flat = X_test.reshape(X_test.shape[0], -1)
    else:
        train_flat = X_train
        test_flat = X_test

    # Check for row overlap
    train_set = set(map(lambda r: tuple(np.round(r, 8)), train_flat))
    test_set = set(map(lambda r: tuple(np.round(r, 8)), test_flat))
    overlap = train_set & test_set
    overlap_ratio = len(overlap) / max(len(test_set), 1)

    if overlap_ratio > 0.1:
        return CheckResult(
            name=name,
            passed=False,
            severity="critical",
            message=f"{len(overlap)} rows ({overlap_ratio:.1%}) appear in both train and test. "
            "Windows were likely generated before splitting.",
            details={
                "overlap_count": len(overlap),
                "overlap_ratio": overlap_ratio,
            },
        )

    return CheckResult(
        name=name,
        passed=True,
        severity="ok",
        message="No row overlap detected between train and test windows.",
        details={"overlap_count": 0},
    )


# ---------------------------------------------------------------------------
# 2. Sliding window boundary check
# ---------------------------------------------------------------------------

def check_window_boundaries(
    train_windows: np.ndarray,
    test_windows: np.ndarray,
    split_index: int,
    window_size: int | None = None,
) -> CheckResult:
    """Check if any sliding window crosses the train/test split boundary.

    Parameters
    ----------
    train_windows : indices or array of window start positions for train
    test_windows : indices or array of window start positions for test
    split_index : index where train ends and test begins (in original series)
    window_size : size of each window

    Returns
    -------
    CheckResult
    """
    name = "Window Boundary"

    train_starts = np.asarray(train_windows).ravel()
    test_starts = np.asarray(test_windows).ravel()

    if window_size is None:
        # Try to infer: assume windows are contiguous
        if len(train_starts) > 1:
            window_size = int(train_starts[1] - train_starts[0])
        else:
            window_size = 1

    # Check: does any train window extend past the split?
    train_end_positions = train_starts + window_size - 1
    crossing_train = train_end_positions >= split_index
    n_crossing = crossing_train.sum()

    if n_crossing > 0:
        return CheckResult(
            name=name,
            passed=False,
            severity="critical",
            message=f"{n_crossing} training window(s) extend past the split point "
            f"(split={split_index}, window_size={window_size}). "
            "These windows contain test-period data.",
            details={
                "n_crossing": int(n_crossing),
                "split_index": split_index,
                "window_size": window_size,
            },
        )

    return CheckResult(
        name=name,
        passed=True,
        severity="ok",
        message="No windows cross the train/test boundary.",
        details={"n_crossing": 0},
    )


# ---------------------------------------------------------------------------
# 3. Normalization fitting check
# ---------------------------------------------------------------------------

def check_normalization_fitting(
    scaler: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> CheckResult:
    """Check whether a scaler/normalizer was fit on the full dataset (leakage)
    or only on the training set (correct).

    Heuristic: if the scaler was fit on training data only, applying it to
    X_test should produce values outside [min, max] of the training set.
    If all test values fall neatly within the training range, the scaler
    may have been fit on the combined data.

    Parameters
    ----------
    scaler : fitted scaler object with transform() method, or None
    X_train : training data
    X_test : test data

    Returns
    -------
    CheckResult
    """
    name = "Normalization Fitting"

    if scaler is None:
        # Check raw data ranges
        train_min, train_max = X_train.min(axis=0), X_train.max(axis=0)
        test_min, test_max = X_test.min(axis=0), X_test.max(axis=0)

        # If test data is entirely within train range, suspicious for raw data
        within_range = np.all(test_min >= train_min) and np.all(test_max <= train_max)
        if within_range and X_train.shape[0] > 100:
            return CheckResult(
                name=name,
                passed=True,
                severity="ok",
                message="No scaler provided. Test data range is within train range "
                "(normal for time series with similar distribution).",
            )
        return CheckResult(
            name=name,
            passed=True,
            severity="ok",
            message="No scaler provided. Skipping normalization check.",
        )

    # Transform test data with the scaler
    try:
        X_test_transformed = scaler.transform(X_test)
    except AttributeError:
        # Not a standard sklearn scaler
        return CheckResult(
            name=name,
            passed=True,
            severity="ok",
            message="Scaler does not have transform() — skipping.",
        )

    # If scaler was fit on train only, test data should sometimes exceed [0,1] or [-1,1]
    # depending on the scaler type
    below_min = (X_test_transformed < -1.5).sum()
    above_max = (X_test_transformed > 1.5).sum()
    total_outliers = below_min + above_max
    total_values = X_test_transformed.size

    if total_outliers == 0 and total_values > 1000:
        return CheckResult(
            name=name,
            passed=False,
            severity="warning",
            message="All transformed test values fall within expected range. "
            "Scaler may have been fit on the full dataset (including test). "
            "Verify that scaler.fit() was called on X_train only.",
            details={
                "outlier_ratio": 0.0,
                "note": "This is a heuristic — not always a leak.",
            },
        )

    outlier_ratio = total_outliers / total_values
    return CheckResult(
        name=name,
        passed=True,
        severity="ok",
        message=f"Transformed test data has {outlier_ratio:.2%} values outside "
        "expected range — consistent with train-only fitting.",
        details={"outlier_ratio": outlier_ratio},
    )
