"""Training curve visualization."""

from __future__ import annotations

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curves(
    train_losses: list[float] | np.ndarray,
    val_losses: list[float] | np.ndarray,
    best_epoch: int | None = None,
    title: str = "Training Curves",
    save_path: str | None = None,
    log_scale: bool = False,
) -> None:
    """Plot training and validation loss curves.

    Parameters
    ----------
    train_losses : training loss per epoch
    val_losses : validation loss per epoch
    best_epoch : epoch with best validation loss (marked with vertical line)
    title : plot title
    save_path : if provided, save to this path; otherwise display
    log_scale : use log scale for y-axis
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss", color="#2196F3", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Val Loss", color="#FF5722", linewidth=1.5)

    if best_epoch is not None and 0 < best_epoch <= len(val_losses):
        ax.axvline(x=best_epoch, color="#4CAF50", linestyle="--", alpha=0.7,
                   label=f"Best (epoch {best_epoch})")

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[saved] {save_path}")
    else:
        plt.show()
    plt.close(fig)
