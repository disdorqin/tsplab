"""CV fold overlap detection for time series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tsplab.audit.leakage import CheckResult


def check_fold_overlap(
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    gap: int = 0,
) -> CheckResult:
    """Check if time series CV folds have temporal overlap.

    In proper time series CV, each fold's training data should end *before*
    the test data starts (with an optional gap). Overlap means a sample
    appears in both training and test across different folds, which is
    a form of leakage.

    Parameters
    ----------
    cv_splits : list of (train_indices, test_indices) tuples
    gap : minimum number of samples between train end and test start

    Returns
    -------
    CheckResult
    """
    name = "Fold Overlap"

    n_splits = len(cv_splits)
    issues = []

    for i, (train_idx, test_idx) in enumerate(cv_splits):
        train_set = set(train_idx.tolist()) if hasattr(train_idx, "tolist") else set(train_idx)
        test_set = set(test_idx.tolist()) if hasattr(test_idx, "tolist") else set(test_idx)

        # Direct overlap within this fold
        direct_overlap = train_set & test_set
        if direct_overlap:
            issues.append(
                f"Fold {i}: {len(direct_overlap)} samples in both train and test"
            )

        # Cross-fold: test data of this fold in test data of another fold (real leakage)
        for j in range(i + 1, n_splits):
            later_test = set(cv_splits[j][1].tolist()) if hasattr(cv_splits[j][1], "tolist") else set(cv_splits[j][1])
            cross_overlap = test_set & later_test
            if cross_overlap:
                issues.append(
                    f"Fold {i} test data also appears in fold {j} test data "
                    f"({len(cross_overlap)} samples) — this is data leakage"
                )

        # Gap check: is there enough buffer between train end and test start?
        train_end = max(train_set) if train_set else 0
        test_start = min(test_set) if test_set else 0
        if gap > 0 and test_start - train_end < gap:
            issues.append(
                f"Fold {i}: gap between train end ({train_end}) and test start "
                f"({test_start}) is {test_start - train_end}, less than required {gap}"
            )

    if issues:
        severity = "critical" if any("both train and test" in s for s in issues) else "warning"
        return CheckResult(
            name=name,
            passed=False,
            severity=severity,
            message=f"{len(issues)} overlap issue(s) found across {n_splits} folds.",
            details={"issues": issues},
        )

    return CheckResult(
        name=name,
        passed=True,
        severity="ok",
        message=f"No overlap detected across {n_splits} folds (gap={gap}).",
    )
