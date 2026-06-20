"""Forecasting metrics: MAE, SMAPE, MASE, RMSE, MAPE, WAPE."""

from __future__ import annotations

import numpy as np


class Metrics:
    """Compute and compare forecasting metrics.

    Usage::

        m = Metrics(y_true, y_pred)
        m.mae()    # -> 0.321
        m.smape()  # -> 12.5
        m.summary()  # -> all metrics
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self.y_true = np.asarray(y_true, dtype=float)
        self.y_pred = np.asarray(y_pred, dtype=float)

    def mae(self) -> float:
        """Mean Absolute Error."""
        return float(np.mean(np.abs(self.y_pred - self.y_true)))

    def rmse(self) -> float:
        """Root Mean Squared Error."""
        return float(np.sqrt(np.mean((self.y_pred - self.y_true) ** 2)))

    def mape(self) -> float:
        """Mean Absolute Percentage Error."""
        return 100.0 * np.mean(
            np.abs(self.y_pred - self.y_true) / (np.abs(self.y_true) + 1e-10)
        )

    def smape(self) -> float:
        """Symmetric MAPE."""
        return 100.0 * np.mean(
            2 * np.abs(self.y_pred - self.y_true)
            / (np.abs(self.y_true) + np.abs(self.y_pred) + 1e-10)
        )

    def wape(self) -> float:
        """Weighted Absolute Percentage Error."""
        total = np.sum(np.abs(self.y_true))
        if total == 0:
            return 0.0
        return 100.0 * np.sum(np.abs(self.y_pred - self.y_true)) / total

    def mase(self, y_train: np.ndarray | None = None, seasonal_length: int = 1) -> float:
        """Mean Absolute Scaled Error.

        Scales MAE by the MAE of the naive seasonal forecast on training data.
        MASE < 1 means your model beats seasonal naive; MASE > 1 means it's worse.

        Parameters
        ----------
        y_train : training data (for computing the scaling factor)
        seasonal_length : seasonal period for naive forecast
        """
        if y_train is None:
            return self.mae()

        y_train = np.asarray(y_train, dtype=float)
        naive_errors = np.abs(
            y_train[seasonal_length:] - y_train[:-seasonal_length]
        )
        scaling = np.mean(naive_errors)
        if scaling == 0:
            return float("inf")
        return self.mae() / scaling

    def summary(self, y_train: np.ndarray | None = None) -> dict[str, float]:
        """Compute all metrics and return as dict."""
        result = {
            "MAE": self.mae(),
            "RMSE": self.rmse(),
            "MAPE": self.mape(),
            "SMAPE": self.smape(),
            "WAPE": self.wape(),
        }
        if y_train is not None:
            result["MASE"] = self.mase(y_train)
        return result

    def __repr__(self) -> str:
        return f"Metrics(MAE={self.mae():.4f}, RMSE={self.rmse():.4f}, SMAPE={self.smape():.2f}%)"
