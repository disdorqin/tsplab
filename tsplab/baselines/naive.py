"""Naive baselines: Persistence, Seasonal Naive, Drift."""

from __future__ import annotations

import numpy as np


class NaiveBaseline:
    """Persistence forecast: predict last observed value.

    The simplest possible baseline. If your DL model can't beat this,
    something is seriously wrong.
    """

    def __init__(self) -> None:
        self.last_value: float | np.ndarray | None = None

    def fit(self, y: np.ndarray) -> "NaiveBaseline":
        self.last_value = y[-1]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.last_value is None:
            raise ValueError("Model not fitted.")
        return np.full(horizon, self.last_value)


class SeasonalNaive:
    """Seasonal naive: predict value from one season ago.

    Parameters
    ----------
    seasonal_length : period of the seasonality (e.g., 24 for hourly, 7 for daily)
    """

    def __init__(self, seasonal_length: int = 1) -> None:
        self.seasonal_length = seasonal_length
        self.history: np.ndarray | None = None

    def fit(self, y: np.ndarray) -> "SeasonalNaive":
        self.history = np.asarray(y)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.history is None:
            raise ValueError("Model not fitted.")

        predictions = np.zeros(horizon)
        for i in range(horizon):
            idx = len(self.history) - self.seasonal_length + (i % self.seasonal_length)
            if idx < 0:
                idx = -1
            predictions[i] = self.history[idx]
        return predictions


class DriftBaseline:
    """Drift forecast: linear extrapolation of the trend.

    Predict = last_value + slope * (t+1), where slope is the
    average change over the entire history.
    """

    def __init__(self) -> None:
        self.last_value: float = 0.0
        self.slope: float = 0.0

    def fit(self, y: np.ndarray) -> "DriftBaseline":
        y = np.asarray(y, dtype=float)
        self.last_value = y[-1]
        if len(y) > 1:
            self.slope = (y[-1] - y[0]) / (len(y) - 1)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        predictions = np.array([
            self.last_value + self.slope * (i + 1)
            for i in range(horizon)
        ])
        return predictions
