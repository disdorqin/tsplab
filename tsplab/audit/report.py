"""Unified audit report that aggregates all leakage checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tsplab.audit.leakage import CheckResult


class AuditReport:
    """Aggregate and display results from all leakage checks.

    Usage::

        report = AuditReport()
        report.check_sequence_generation(X_train, X_test, timestamps)
        report.check_window_boundaries(train_windows, test_windows, split_idx)
        report.check_normalization_fitting(scaler, X_train, X_test)
        report.check_fold_overlap(cv_splits)
        report.check_covariate_availability(covariates, timestamps, horizon)
        report.summary()
    """

    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def add(self, result: CheckResult) -> None:
        """Add a check result to the report."""
        self.results.append(result)

    # -- Convenience wrappers for each check --

    def check_sequence_generation(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        timestamps: np.ndarray | pd.Series | None = None,
        window_size: int | None = None,
    ) -> CheckResult:
        from tsplab.audit.leakage import check_sequence_generation as _check

        result = _check(X_train, X_test, timestamps, window_size)
        self.add(result)
        return result

    def check_window_boundaries(
        self,
        train_windows: np.ndarray,
        test_windows: np.ndarray,
        split_index: int,
        window_size: int | None = None,
    ) -> CheckResult:
        from tsplab.audit.leakage import check_window_boundaries as _check

        result = _check(train_windows, test_windows, split_index, window_size)
        self.add(result)
        return result

    def check_normalization_fitting(
        self,
        scaler: Any,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> CheckResult:
        from tsplab.audit.leakage import check_normalization_fitting as _check

        result = _check(scaler, X_train, X_test)
        self.add(result)
        return result

    def check_fold_overlap(
        self,
        cv_splits: list[tuple[np.ndarray, np.ndarray]],
        gap: int = 0,
    ) -> CheckResult:
        from tsplab.audit.fold_check import check_fold_overlap as _check

        result = _check(cv_splits, gap)
        self.add(result)
        return result

    def check_covariate_availability(
        self,
        covariates: pd.DataFrame | np.ndarray,
        timestamps: np.ndarray | pd.Series,
        horizon: int,
        covariate_names: list[str] | None = None,
        known_future: list[str] | None = None,
    ) -> CheckResult:
        from tsplab.audit.covariate_check import check_covariate_availability as _check

        result = _check(covariates, timestamps, horizon, covariate_names, known_future)
        self.add(result)
        return result

    def quantify_rmse_gain(
        self,
        rmse_clean: float,
        rmse_leaky: float,
    ) -> CheckResult:
        from tsplab.audit.rmse_gain import quantify_rmse_gain as _check

        result = _check(rmse_clean, rmse_leaky)
        self.add(result)
        return result

    # -- Summary --

    def summary(self) -> None:
        """Print a formatted summary of all checks."""
        print("=" * 70)
        print("  TSPLab Data Audit Report")
        print("=" * 70)
        print()

        if not self.results:
            print("  No checks performed yet.")
            return

        n_critical = 0
        n_warning = 0
        n_ok = 0

        for r in self.results:
            print(f"  {r}")
            if r.details:
                for k, v in r.details.items():
                    if isinstance(v, list) and len(v) > 3:
                        print(f"      {k}: [{len(v)} items]")
                    elif isinstance(v, dict):
                        for dk, dv in v.items():
                            print(f"      {k}.{dk}: {dv}")
                    else:
                        print(f"      {k}: {v}")
            print()

            if r.severity == "critical":
                n_critical += 1
            elif r.severity == "warning":
                n_warning += 1
            else:
                n_ok += 1

        print("-" * 70)
        print(f"  Total: {len(self.results)} checks | "
              f"[OK] {n_ok}  [!] {n_warning}  [X] {n_critical}")
        print()

        if n_critical > 0:
            print("  [X] CRITICAL issues found — fix these before trusting your results!")
        elif n_warning > 0:
            print("  [!] Warnings found — review recommended.")
        else:
            print("  All checks passed. Your data pipeline looks clean.")
        print("=" * 70)
