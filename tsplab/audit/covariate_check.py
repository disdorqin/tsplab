"""Covariate availability checker.

Ensures that 'future' covariates used during training are actually
obtainable at prediction time in a real deployment scenario.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsplab.audit.leakage import CheckResult


def check_covariate_availability(
    covariates: pd.DataFrame | np.ndarray,
    timestamps: np.ndarray | pd.Series,
    horizon: int,
    covariate_names: list[str] | None = None,
    known_future: list[str] | None = None,
) -> CheckResult:
    """Check if future covariates will be available at prediction time.

    In time series forecasting, some covariates are:
      - *past-observed*: known up to the current time (e.g., historical weather)
      - *future-known*: known ahead of time (e.g., calendar features, scheduled events)
      - *static*: constant across time (e.g., location ID)

    Using a covariate as "future-known" when it's actually "past-observed"
    is a form of look-ahead leakage.

    Parameters
    ----------
    covariates : DataFrame or array of covariate values
    timestamps : time index for the covariates
    horizon : forecast horizon (how many steps ahead you predict)
    covariate_names : names of covariate columns
    known_future : list of covariate names that are genuinely known in the future

    Returns
    -------
    CheckResult
    """
    name = "Covariate Availability"

    if isinstance(covariates, np.ndarray):
        n_features = covariates.shape[1] if covariates.ndim > 1 else 1
        covariate_names = covariate_names or [f"feature_{i}" for i in range(n_features)]
    elif isinstance(covariates, pd.DataFrame):
        covariate_names = covariate_names or list(covariates.columns)
        n_features = len(covariate_names)
    else:
        return CheckResult(
            name=name,
            passed=True,
            severity="ok",
            message="Unsupported covariate type — skipping.",
        )

    known_future = known_future or []

    # Classify each covariate
    classifications = {}
    suspicious = []

    for i, col_name in enumerate(covariate_names):
        if isinstance(covariates, pd.DataFrame):
            col_data = covariates[col_name].values
        else:
            col_data = covariates[:, i] if covariates.ndim > 1 else covariates

        # Check if constant (static covariate)
        if np.nanstd(col_data) < 1e-10:
            classifications[col_name] = "static"
            continue

        # Check if it looks like calendar features (weekday, month, etc.)
        # These are genuinely future-known
        if any(kw in col_name.lower() for kw in ["day", "week", "month", "year", "hour", "holiday", "season"]):
            classifications[col_name] = "calendar (future-known)"
            continue

        # If declared as known_future, trust the user
        if col_name in known_future:
            classifications[col_name] = "declared future-known"
            continue

        # Otherwise, treat as past-observed
        classifications[col_name] = "past-observed"
        suspicious.append(col_name)

    if not suspicious:
        return CheckResult(
            name=name,
            passed=True,
            severity="ok",
            message=f"All {n_features} covariate(s) are calendar/static/declared-future-known. "
            f"No look-ahead risk detected.",
            details={"classifications": classifications},
        )

    return CheckResult(
        name=name,
        passed=True,
        severity="warning",
        message=f"{len(suspicious)} covariate(s) classified as past-observed: {suspicious}. "
        f"Ensure these are NOT used as future covariates in your model. "
        f"If they are, you have look-ahead leakage.",
        details={
            "classifications": classifications,
            "suspicious": suspicious,
            "horizon": horizon,
        },
    )
