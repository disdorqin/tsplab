"""Time series-aware hyperparameter tuner using Optuna.

Key features:
  - Time series cross-validation (expanding/rolling window)
  - Optuna pruners for early termination of bad trials
  - gap between train and validation to prevent leakage
  - Hyperparameter importance analysis
"""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TimeSeriesTuner:
    """Hyperparameter tuner for time series models.

    Usage::

        def model_factory(trial):
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            hidden = trial.suggest_int("hidden", 32, 256)
            return MyModel(lr=lr, hidden=hidden)

        def train_eval(model, train_data, val_data):
            model.fit(train_data)
            return model.score(val_data)  # return metric (lower=better)

        tuner = TimeSeriesTuner(
            model_factory=model_factory,
            train_eval_fn=train_eval,
            cv_strategy="expanding",
            n_windows=5,
            gap=1,
        )

        study = tuner.optimize(data, timestamps)
        print(tuner.best_params)
    """

    def __init__(
        self,
        model_factory: Callable,
        train_eval_fn: Callable | None = None,
        cv_strategy: str = "expanding",
        n_windows: int = 5,
        gap: int = 0,
        refit_every: int | None = None,
        pruner: str = "hyperband",
        timeout: int = 1800,
        metric_name: str = "val_loss",
        direction: str = "minimize",
    ) -> None:
        """
        Parameters
        ----------
        model_factory : function(trial) -> model instance
        train_eval_fn : function(model, train_idx, val_idx, data) -> float
                        If None, uses default: calls model.fit() and model.score()
        cv_strategy : "expanding" or "rolling"
        n_windows : number of CV windows
        gap : samples to skip between train and val (prevents leakage)
        refit_every : refit model every N windows (None = refit every window)
        pruner : "hyperband", "median", or "none"
        timeout : max seconds for optimization
        metric_name : name of the metric being optimized
        direction : "minimize" or "maximize"
        """
        self.model_factory = model_factory
        self.train_eval_fn = train_eval_fn
        self.cv_strategy = cv_strategy
        self.n_windows = n_windows
        self.gap = gap
        self.refit_every = refit_every
        self.pruner_name = pruner
        self.timeout = timeout
        self.metric_name = metric_name
        self.direction = direction

        self.study = None
        self.best_params: dict | None = None
        self.best_value: float | None = None

    def _make_cv_splits(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate time series CV splits."""
        # Reserve last n_windows * test_size for validation
        test_size = max(n_samples // (self.n_windows + 2), 1)
        splits = []

        for i in range(self.n_windows):
            # Test window
            test_start = n_samples - (self.n_windows - i) * test_size
            test_end = test_start + test_size

            # Train window
            if self.cv_strategy == "rolling":
                train_start = max(0, test_start - test_size - self.gap)
            else:  # expanding
                train_start = 0

            train_end = test_start - self.gap

            if train_end <= train_start:
                continue

            train_idx = np.arange(train_start, train_end)
            val_idx = np.arange(test_start, min(test_end, n_samples))
            splits.append((train_idx, val_idx))

        return splits

    def optimize(self, data: np.ndarray, timestamps: np.ndarray | None = None) -> Any:
        """Run hyperparameter optimization.

        Parameters
        ----------
        data : time series data (1D or 2D)
        timestamps : optional time index

        Returns
        -------
        Optuna Study object
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            raise ImportError(
                "optuna is required for tuning. Install with: pip install tsplab[tune]"
            )

        # Setup pruner
        if self.pruner_name == "hyperband":
            pruner = optuna.pruners.HyperbandPruner()
        elif self.pruner_name == "median":
            pruner = optuna.pruners.MedianPruner()
        else:
            pruner = optuna.pruners.NopPruner()

        # Create study
        sampler = TPESampler()
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
        )

        n_samples = len(data) if hasattr(data, "__len__") else 0
        cv_splits = self._make_cv_splits(n_samples)

        def objective(trial):
            model = self.model_factory(trial)

            scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
                if self.train_eval_fn is not None:
                    score = self.train_eval_fn(model, train_idx, val_idx, data)
                else:
                    # Default: assume model has fit/score interface
                    train_data = data[train_idx] if hasattr(data, "__getitem__") else data
                    val_data = data[val_idx] if hasattr(data, "__getitem__") else data
                    model.fit(train_data)
                    score = model.score(val_data)

                scores.append(score)

                # Report intermediate value for pruning
                trial.report(np.mean(scores), fold_idx)

                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(scores)

        # Run optimization
        self.study.optimize(objective, timeout=self.timeout)

        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        print(f"\nBest {self.metric_name}: {self.best_value:.6f}")
        print(f"Best params: {self.best_params}")

        return self.study

    def plot_optimization_history(self, save_path: str | None = None) -> None:
        """Plot optimization history."""
        if self.study is None:
            print("Run optimize() first.")
            return

        try:
            import optuna
        except ImportError:
            return

        fig, ax = plt.subplots(figsize=(10, 5))

        trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        values = [t.value for t in trials]
        best_values = []
        best = float("inf") if self.direction == "minimize" else float("-inf")
        for v in values:
            if self.direction == "minimize":
                best = min(best, v)
            else:
                best = max(best, v)
            best_values.append(best)

        ax.plot(range(len(values)), values, "o-", color="#2196F3", alpha=0.5, label="Trial value")
        ax.plot(range(len(best_values)), best_values, color="#FF5722", linewidth=2, label="Best value")
        ax.set_xlabel("Trial")
        ax.set_ylabel(self.metric_name)
        ax.set_title("Optimization History")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[saved] {save_path}")
        plt.close(fig)

    def plot_param_importances(self, save_path: str | None = None) -> None:
        """Plot hyperparameter importances."""
        if self.study is None:
            print("Run optimize() first.")
            return

        try:
            import optuna
        except ImportError:
            return

        try:
            importances = optuna.importance.get_param_importances(self.study)
        except Exception:
            print("Cannot compute importances (need more completed trials).")
            return

        fig, ax = plt.subplots(figsize=(8, 4))

        names = list(importances.keys())
        values = list(importances.values())

        ax.barh(names, values, color="#4CAF50")
        ax.set_xlabel("Importance")
        ax.set_title("Hyperparameter Importances")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[saved] {save_path}")
        plt.close(fig)
