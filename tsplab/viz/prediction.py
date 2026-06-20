"""Prediction vs ground truth visualization."""

from __future__ import annotations

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_history: np.ndarray | None = None,
    title: str = "Predictions vs Ground Truth",
    save_path: str | None = None,
    confidence_interval: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Plot prediction vs ground truth.

    Parameters
    ----------
    y_true : ground truth values
    y_pred : predicted values
    y_history : optional history (plotted before predictions)
    title : plot title
    save_path : if provided, save; otherwise display
    confidence_interval : (lower_bound, upper_bound) for prediction interval
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    n_pred = len(y_pred)
    pred_x = range(len(y_history) if y_history is not None else 0,
                   len(y_history) if y_history is not None else 0 + n_pred)

    if y_history is not None:
        hist_x = range(len(y_history))
        ax.plot(hist_x, y_history, color="#9E9E9E", linewidth=1, label="History")
        ax.plot(pred_x, y_true, color="#4CAF50", linewidth=1.5, label="Ground Truth")
        ax.plot(pred_x, y_pred, color="#FF5722", linewidth=1.5, linestyle="--", label="Prediction")
    else:
        ax.plot(range(n_pred), y_true, color="#4CAF50", linewidth=1.5, label="Ground Truth")
        ax.plot(range(n_pred), y_pred, color="#FF5722", linewidth=1.5, linestyle="--", label="Prediction")

    # Confidence interval
    if confidence_interval is not None:
        lower, upper = confidence_interval
        ax.fill_between(pred_x, lower, upper, alpha=0.2, color="#FF5722", label="Confidence")

    ax.axvline(x=len(y_history) - 0.5 if y_history is not None else -0.5,
               color="#2196F3", linestyle=":", alpha=0.5, label="Forecast Start")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[saved] {save_path}")
    else:
        plt.show()
    plt.close(fig)
