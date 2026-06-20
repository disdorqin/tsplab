"""Tree-based baseline (Decision Tree / XGBoost).

Uses a simple sliding-window-to-tabular approach for tree models.
This is often the strong baseline that DL models struggle to beat.
"""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeRegressor


class TreeBaseline:
    """Decision tree baseline for time series.

    Converts sliding windows to tabular format and fits a tree regressor.

    Parameters
    ----------
    tree_type : "decision_tree" or "xgboost"
    max_depth : tree max depth
    """

    def __init__(
        self,
        tree_type: str = "decision_tree",
        max_depth: int = 8,
    ) -> None:
        self.tree_type = tree_type
        self.max_depth = max_depth
        self.model = None
        self.lookback: int | None = None

    def _make_features(self, y: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
        """Convert 1D series to (X, y) tabular format using sliding windows."""
        n = len(y)
        X, targets = [], []
        for i in range(n - lookback):
            X.append(y[i:i + lookback])
            targets.append(y[i + lookback])
        return np.array(X), np.array(targets)

    def fit(self, y: np.ndarray, lookback: int = 24) -> "TreeBaseline":
        """Fit the tree model on the series.

        Parameters
        ----------
        y : 1D time series
        lookback : number of past steps used as features
        """
        self.lookback = lookback
        X, targets = self._make_features(y, lookback)

        if self.tree_type == "xgboost":
            try:
                from xgboost import XGBRegressor
                self.model = XGBRegressor(
                    max_depth=self.max_depth,
                    n_estimators=100,
                    learning_rate=0.1,
                )
            except ImportError:
                print("[tsplab] xgboost not installed, falling back to DecisionTree")
                self.model = DecisionTreeRegressor(max_depth=self.max_depth)
        else:
            self.model = DecisionTreeRegressor(max_depth=self.max_depth)

        self.model.fit(X, targets)
        return self

    def predict(self, y_history: np.ndarray, horizon: int) -> np.ndarray:
        """Predict future values using autoregressive approach."""
        if self.model is None or self.lookback is None:
            raise ValueError("Model not fitted.")

        predictions = []
        window = list(y_history[-self.lookback:])

        for _ in range(horizon):
            x = np.array(window[-self.lookback:]).reshape(1, -1)
            pred = self.model.predict(x)[0]
            predictions.append(pred)
            window.append(pred)

        return np.array(predictions)
