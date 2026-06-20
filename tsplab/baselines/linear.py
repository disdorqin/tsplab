"""DLinear and NLinear baselines (AAAI 2023).

Single-layer linear models that surprisingly outperform Transformers
on many time series datasets.

Reference: Zeng et al., "Are Transformers Effective for Time Series Forecasting?"
"""

from __future__ import annotations

import numpy as np


class _LinearBase:
    """Base for DLinear/NLinear: decompose trend + seasonal, then linear regression."""

    def __init__(
        self,
        lookback: int = 96,
        kernel_size: int = 25,
    ) -> None:
        self.lookback = lookback
        self.kernel_size = kernel_size
        self._fitted = False

    def _decompose(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decompose into trend (moving average) and seasonal (residual)."""
        from scipy.ndimage import uniform_filter1d

        trend = uniform_filter1d(x, size=self.kernel_size, mode="nearest")
        seasonal = x - trend
        return trend, seasonal

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LinearBase":
        """Fit linear models on decomposed trend and seasonal components.

        Parameters
        ----------
        X : shape (n_samples, lookback) — input windows
        y : shape (n_samples, horizon) — target windows
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Decompose each window
        n_samples = X.shape[0]
        X_trend = np.zeros_like(X)
        X_seasonal = np.zeros_like(X)

        for i in range(n_samples):
            t, s = self._decompose(X[i])
            X_trend[i] = t
            X_seasonal[i] = s

        # Fit linear regression: y = W_trend @ trend + W_seasonal @ seasonal
        # Using least squares
        X_combined = np.hstack([X_trend, X_seasonal])
        # Add bias
        X_with_bias = np.column_stack([X_combined, np.ones(n_samples)])

        # Solve: W = (X^T X)^{-1} X^T y
        self.weights, _, _, _ = np.linalg.lstsq(X_with_bias, y, rcond=None)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Model not fitted.")

        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        X_trend = np.zeros_like(X)
        X_seasonal = np.zeros_like(X)

        for i in range(n_samples):
            t, s = self._decompose(X[i])
            X_trend[i] = t
            X_seasonal[i] = s

        X_combined = np.hstack([X_trend, X_seasonal])
        X_with_bias = np.column_stack([X_combined, np.ones(n_samples)])

        return X_with_bias @ self.weights


class DLinear(_LinearBase):
    """DLinear: Decomposition + Linear.

    Decomposes input into trend and seasonal, then applies
    separate linear layers to each component.
    """

    def __repr__(self) -> str:
        return f"DLinear(lookback={self.lookback}, kernel_size={self.kernel_size})"


class NLinear(_LinearBase):
    """NLinear: Normalization + Linear.

    Subtracts the last value of the input window before prediction,
    then adds it back. A simple normalization trick.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NLinear":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Normalize: subtract last value
        self.last_values = X[:, -1:]
        X_norm = X - self.last_values
        y_norm = y - self.last_values

        super().fit(X_norm, y_norm)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        last_values = X[:, -1:]
        X_norm = X - last_values

        pred_norm = super().predict(X_norm)
        return pred_norm + last_values

    def __repr__(self) -> str:
        return f"NLinear(lookback={self.lookback}, kernel_size={self.kernel_size})"
