"""RMSE Gain quantification for leakage impact assessment.

RMSE Gain (%) = (RMSE_clean - RMSE_leaky) / RMSE_clean * 100

Positive value = leakage causes optimistic bias (performance looks better than reality).
>5% is significant, >10% is severe.
"""

from __future__ import annotations

from tsplab.audit.leakage import CheckResult


def quantify_rmse_gain(
    rmse_clean: float,
    rmse_leaky: float,
) -> CheckResult:
    """Quantify the impact of data leakage via RMSE Gain.

    Parameters
    ----------
    rmse_clean : RMSE from a properly cleaned (no-leak) pipeline
    rmse_leaky : RMSE from a potentially leaky pipeline

    Returns
    -------
    CheckResult
    """
    name = "RMSE Gain"

    if rmse_clean == 0:
        return CheckResult(
            name=name,
            passed=True,
            severity="ok",
            message="Cannot compute RMSE Gain (clean RMSE is 0).",
        )

    gain = (rmse_clean - rmse_leaky) / rmse_clean * 100

    if gain > 10:
        severity = "critical"
        message = (
            f"RMSE Gain = {gain:.1f}% — SEVERE leakage. "
            f"The leaky pipeline's RMSE ({rmse_leaky:.4f}) is {gain:.1f}% lower than "
            f"the clean pipeline ({rmse_clean:.4f}). "
            f"Your model's reported performance is likely inflated."
        )
    elif gain > 5:
        severity = "warning"
        message = (
            f"RMSE Gain = {gain:.1f}% — moderate leakage detected. "
            f"Clean RMSE: {rmse_clean:.4f}, Leaky RMSE: {rmse_leaky:.4f}."
        )
    elif gain > 1:
        severity = "ok"
        message = (
            f"RMSE Gain = {gain:.1f}% — minor difference, likely within noise. "
            f"Clean: {rmse_clean:.4f}, Leaky: {rmse_leaky:.4f}."
        )
    else:
        severity = "ok"
        message = (
            f"RMSE Gain = {gain:.1f}% — no significant leakage impact. "
            f"Clean: {rmse_clean:.4f}, Leaky: {rmse_leaky:.4f}."
        )

    return CheckResult(
        name=name,
        passed=gain <= 5,
        severity=severity,
        message=message,
        details={
            "rmse_clean": rmse_clean,
            "rmse_leaky": rmse_leaky,
            "gain_percent": gain,
        },
    )
