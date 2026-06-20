"""Residual analysis plots."""

from __future__ import annotations

import os
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Analysis",
    save_path: str | None = None,
) -> None:
    """Plot residual analysis: time series, histogram, and Q-Q plot.

    Parameters
    ----------
    y_true : ground truth
    y_pred : predictions
    title : overall title
    save_path : if provided, save; otherwise display
    """
    residuals = np.array(y_true) - np.array(y_pred)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Residuals over time
    ax = axes[0]
    ax.plot(range(len(residuals)), residuals, color="#FF5722", linewidth=1)
    ax.axhline(y=0, color="#9E9E9E", linestyle="--")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals Over Time")
    ax.grid(True, alpha=0.3)

    # 2. Residual histogram
    ax = axes[1]
    ax.hist(residuals, bins=30, color="#2196F3", alpha=0.7, density=True, edgecolor="white")
    # Overlay normal distribution
    mu, sigma = np.mean(residuals), np.std(residuals)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color="#FF5722", linewidth=2, label="Normal fit")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.set_title(f"Residual Distribution (mu={mu:.4f}, sigma={sigma:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Q-Q plot
    ax = axes[2]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot")
    ax.grid(True, alpha=0.3)

    # Adjust line colors
    for line in ax.get_lines():
        if line.get_marker() == "o":
            line.set_markerfacecolor("#2196F3")
            line.set_markeredgecolor("#2196F3")
        else:
            line.set_color("#FF5722")

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[saved] {save_path}")
    else:
        plt.show()
    plt.close(fig)
