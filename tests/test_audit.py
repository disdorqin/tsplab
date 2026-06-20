"""Tests for tsplab.audit module."""

import numpy as np
from tsplab.audit import AuditReport
from tsplab.audit.leakage import check_sequence_generation, check_window_boundaries
from tsplab.audit.fold_check import check_fold_overlap
from tsplab.audit.rmse_gain import quantify_rmse_gain


def test_sequence_generation_no_overlap():
    """Clean split should pass."""
    X_train = np.random.randn(100, 24)
    X_test = np.random.randn(20, 24) + 100  # Different distribution
    result = check_sequence_generation(X_train, X_test)
    assert result.passed
    assert result.severity == "ok"


def test_sequence_generation_with_overlap():
    """Overlapping data should fail."""
    data = np.random.randn(100, 24)
    X_train = data[:80]
    X_test = data[70:]  # 10 overlapping rows
    result = check_sequence_generation(X_train, X_test)
    assert not result.passed
    assert result.severity == "critical"


def test_window_boundaries_clean():
    """Windows that don't cross split should pass."""
    train_windows = np.array([0, 24, 48, 72])
    test_windows = np.array([96, 120])
    result = check_window_boundaries(train_windows, test_windows, split_index=96, window_size=24)
    assert result.passed


def test_window_boundaries_crossing():
    """Windows crossing the split should fail."""
    train_windows = np.array([0, 24, 48, 72, 80])
    test_windows = np.array([96, 120])
    # Window starting at 80 with size 24 ends at 103, past split at 96
    result = check_window_boundaries(train_windows, test_windows, split_index=96, window_size=24)
    assert not result.passed
    assert result.severity == "critical"


def test_fold_overlap_clean():
    """Non-overlapping folds should pass."""
    cv_splits = [
        (np.arange(0, 100), np.arange(100, 120)),
        (np.arange(0, 120), np.arange(120, 140)),
        (np.arange(0, 140), np.arange(140, 160)),
    ]
    result = check_fold_overlap(cv_splits, gap=0)
    assert result.passed


def test_fold_overlap_leaky():
    """Overlapping folds should fail."""
    cv_splits = [
        (np.arange(0, 100), np.arange(90, 110)),  # 90-99 in both
        (np.arange(0, 110), np.arange(100, 120)),
    ]
    result = check_fold_overlap(cv_splits)
    assert not result.passed


def test_rmse_gain_no_leak():
    """Small RMSE gain should pass."""
    result = quantify_rmse_gain(rmse_clean=0.50, rmse_leaky=0.49)
    assert result.passed
    assert result.severity == "ok"


def test_rmse_gain_severe():
    """Large RMSE gain should fail critically."""
    result = quantify_rmse_gain(rmse_clean=0.50, rmse_leaky=0.35)
    assert not result.passed
    assert result.severity == "critical"


def test_audit_report_summary():
    """AuditReport should aggregate results."""
    report = AuditReport()
    report.check_sequence_generation(
        np.random.randn(50, 10),
        np.random.randn(10, 10) + 50,
    )
    report.summary()
    assert len(report.results) == 1
