"""
tsplab.audit — Data leakage detection for time series forecasting.

Six-check matrix:
  1. Sequence generation timing
  2. Sliding window boundary
  3. Normalization fitting
  4. CV fold overlap
  5. Covariate availability
  6. RMSE Gain quantification
"""

from tsplab.audit.report import AuditReport
from tsplab.audit.leakage import (
    check_sequence_generation,
    check_window_boundaries,
    check_normalization_fitting,
)
from tsplab.audit.fold_check import check_fold_overlap
from tsplab.audit.covariate_check import check_covariate_availability
from tsplab.audit.rmse_gain import quantify_rmse_gain

__all__ = [
    "AuditReport",
    "check_sequence_generation",
    "check_window_boundaries",
    "check_normalization_fitting",
    "check_fold_overlap",
    "check_covariate_availability",
    "quantify_rmse_gain",
]
