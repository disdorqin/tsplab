"""
Training curve monitor with automatic diagnosis.

Diagnoses 7 training patterns (from Google tuning_playbook):
  1. Healthy Convergence
  2. Overfitting
  3. Instability
  4. Slow Convergence
  5. Still Improving
  6. Early Plateau
  7. High Variance
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


class StopReason(Enum):
    """Why training was stopped (inspired by PyTorch Lightning)."""

    NOT_STOPPED = "training continues"
    PLATEAU = "loss plateaued — no improvement expected"
    OVERFITTING = "validation loss diverging from training — overfitting"
    DIVERGENCE = "loss diverged (NaN/Inf or exceeded threshold)"
    PATIENCE_EXHAUSTED = "patience exhausted — no improvement for N epochs"
    NON_FINITE_METRIC = "NaN/Inf detected in metrics"


@dataclass
class EpochStatus:
    """Status returned by TrainingMonitor.log() after each epoch."""

    should_stop: bool
    reason: StopReason
    epoch: int
    train_loss: float
    val_loss: float
    best_epoch: int
    best_val_loss: float
    diagnosis: str = ""


class TrainingMonitor:
    """Log training metrics, auto-diagnose, and generate reports.

    Usage::

        monitor = TrainingMonitor(patience=15, save_dir="./runs/exp_001")

        for epoch in range(epochs):
            train_loss = model.train_one_epoch(...)
            val_loss = model.validate(...)

            status = monitor.log(epoch, train_loss, val_loss)
            if status.should_stop:
                print(f"Stopping: {status.reason.value}")
                break

        monitor.report()  # Auto-plot + diagnosis
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        divergence_threshold: float = 5.0,
        check_finite: bool = True,
        plateau_threshold: float = 1e-4,
        save_dir: str | None = None,
        model_name: str = "model",
    ) -> None:
        """
        Parameters
        ----------
        patience : epochs without improvement before early stop
        min_delta : minimum change to count as improvement
        divergence_threshold : stop if val_loss > best_val_loss * threshold
        check_finite : stop if NaN/Inf detected
        plateau_threshold : train_loss slope below this = plateau
        save_dir : where to save plots (None = no saving)
        model_name : label for plot titles
        """
        self.patience = patience
        self.min_delta = min_delta
        self.divergence_threshold = divergence_threshold
        self.check_finite = check_finite
        self.plateau_threshold = plateau_threshold
        self.save_dir = save_dir
        self.model_name = model_name

        # History
        self.epochs: list[int] = []
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.grad_norms: list[float] = []

        # State
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0
        self._epochs_without_improvement: int = 0

    def log(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        grad_norm: float | None = None,
    ) -> EpochStatus:
        """Log one epoch's metrics. Returns status with stop decision."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        if grad_norm is not None:
            self.grad_norms.append(grad_norm)

        # Check finite
        if self.check_finite:
            if not np.isfinite(train_loss) or not np.isfinite(val_loss):
                return self._stop(epoch, StopReason.NON_FINITE_METRIC, train_loss, val_loss)

        # Update best
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1

        # Check divergence
        if val_loss > self.best_val_loss * self.divergence_threshold and self.best_val_loss < float("inf"):
            return self._stop(epoch, StopReason.DIVERGENCE, train_loss, val_loss)

        # Check patience
        if self._epochs_without_improvement >= self.patience:
            diagnosis = self._diagnose()
            if "overfitting" in diagnosis.lower():
                return self._stop(epoch, StopReason.OVERFITTING, train_loss, val_loss, diagnosis)
            return self._stop(epoch, StopReason.PATIENCE_EXHAUSTED, train_loss, val_loss, diagnosis)

        # Still training — provide ongoing diagnosis
        diagnosis = self._diagnose()
        return EpochStatus(
            should_stop=False,
            reason=StopReason.NOT_STOPPED,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            best_epoch=self.best_epoch,
            best_val_loss=self.best_val_loss,
            diagnosis=diagnosis,
        )

    def _stop(
        self,
        epoch: int,
        reason: StopReason,
        train_loss: float,
        val_loss: float,
        diagnosis: str = "",
    ) -> EpochStatus:
        return EpochStatus(
            should_stop=True,
            reason=reason,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            best_epoch=self.best_epoch,
            best_val_loss=self.best_val_loss,
            diagnosis=diagnosis or self._diagnose(),
        )

    def _diagnose(self) -> str:
        """Diagnose training pattern from recent history."""
        if len(self.train_losses) < 5:
            return "Not enough data to diagnose yet (need >= 5 epochs)."

        n = len(self.train_losses)
        recent_k = min(10, n)
        recent_train = np.array(self.train_losses[-recent_k:])
        recent_val = np.array(self.val_losses[-recent_k:])

        # Compute slopes
        train_slope = np.polyfit(range(recent_k), recent_train, 1)[0] if recent_k > 1 else 0
        val_slope = np.polyfit(range(recent_k), recent_val, 1)[0] if recent_k > 1 else 0

        # Compute gap
        gap = recent_val[-1] / max(recent_train[-1], 1e-10)

        # Compute variance (instability check)
        train_var = np.std(recent_train) / max(abs(np.mean(recent_train)), 1e-10)
        val_var = np.std(recent_val) / max(abs(np.mean(recent_val)), 1e-10)

        # Determine pattern
        patterns = []

        # 1. Instability
        if train_var > 0.1 or val_var > 0.1:
            patterns.append(
                "Instability: loss is oscillating significantly. "
                "Suggestion: reduce learning rate, add gradient clipping, or use warmup."
            )

        # 2. Overfitting
        if gap > 2.0 and train_slope < 0 and val_slope > 0:
            patterns.append(
                f"Overfitting: train_loss still decreasing but val_loss increasing. "
                f"Gap = {gap:.1f}x. "
                "Suggestion: increase dropout, reduce model capacity, add weight decay."
            )

        # 3. Overfitting (severe)
        elif gap > 3.0:
            patterns.append(
                f"Severe overfitting: val_loss is {gap:.1f}x train_loss. "
                "Suggestion: strong regularization needed, or reduce model size."
            )

        # 4. Slow convergence
        elif abs(train_slope) < self.plateau_threshold and gap < 1.5 and recent_train[-1] > 0.1:
            patterns.append(
                "Slow Convergence: loss barely decreasing. "
                "Suggestion: increase learning rate, check for vanishing gradients."
            )

        # 5. Still improving
        elif train_slope < -self.plateau_threshold and val_slope < -self.plateau_threshold:
            patterns.append(
                "Still Improving: both train and val loss decreasing. "
                "Consider training for more epochs."
            )

        # 6. Healthy convergence
        elif abs(train_slope) < self.plateau_threshold and abs(val_slope) < self.plateau_threshold and gap < 1.5:
            patterns.append(
                "Healthy Convergence: both losses have plateaued at similar values. "
                "Model is well-trained."
            )

        # 7. Early plateau
        elif abs(train_slope) < self.plateau_threshold and recent_train[-1] > 0.3:
            patterns.append(
                "Early Plateau: loss plateaued at a high value. "
                "Model may be underfitting. Suggestion: increase model capacity, "
                "decrease regularization, or check data quality."
            )

        if not patterns:
            patterns.append(
                f"Training in progress. train_slope={train_slope:.6f}, "
                f"val_slope={val_slope:.6f}, gap={gap:.2f}x"
            )

        return "\n      ".join(patterns)

    def report(self) -> None:
        """Print a full training report and save plots."""
        print("=" * 70)
        print(f"  TSPLab Training Report — {self.model_name}")
        print("=" * 70)
        print()

        if not self.train_losses:
            print("  No training data recorded.")
            return

        # Basic stats
        print(f"  Epochs trained: {len(self.epochs)}")
        print(f"  Best epoch: {self.best_epoch}")
        print(f"  Best val_loss: {self.best_val_loss:.6f}")
        print(f"  Final train_loss: {self.train_losses[-1]:.6f}")
        print(f"  Final val_loss: {self.val_losses[-1]:.6f}")
        gap = self.val_losses[-1] / max(self.train_losses[-1], 1e-10)
        print(f"  Final train/val gap: {gap:.2f}x")
        print()

        # Diagnosis
        print("  Diagnosis:")
        diagnosis = self._diagnose()
        print(f"      {diagnosis}")
        print()

        # Gradient health
        if self.grad_norms:
            print(f"  Gradient norm (last): {self.grad_norms[-1]:.4f}")
            print(f"  Gradient norm (mean): {np.mean(self.grad_norms):.4f}")
            if self.grad_norms[-1] < 1e-6:
                print("  [!] Gradient vanishing detected!")
            elif self.grad_norms[-1] > 100:
                print("  [!] Gradient explosion detected!")
            print()

        print("=" * 70)

        # Save plots
        if self.save_dir:
            self._plot()

    def _plot(self) -> None:
        """Generate and save training curve plot."""
        os.makedirs(self.save_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curves
        ax = axes[0]
        ax.plot(self.epochs, self.train_losses, label="Train Loss", color="#2196F3", linewidth=1.5)
        ax.plot(self.epochs, self.val_losses, label="Val Loss", color="#FF5722", linewidth=1.5)
        ax.axvline(x=self.best_epoch, color="#4CAF50", linestyle="--", alpha=0.7, label=f"Best (epoch {self.best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{self.model_name} — Training Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Gradient norms (if available)
        ax = axes[1]
        if self.grad_norms:
            ax.plot(self.epochs[:len(self.grad_norms)], self.grad_norms, color="#9C27B0", linewidth=1.5)
            ax.set_ylabel("Gradient Norm")
            ax.set_title("Gradient Health")
            ax.grid(True, alpha=0.3)
        else:
            # Log scale loss instead
            ax.plot(self.epochs, self.train_losses, label="Train", color="#2196F3")
            ax.plot(self.epochs, self.val_losses, label="Val", color="#FF5722")
            ax.set_yscale("log")
            ax.set_ylabel("Loss (log scale)")
            ax.set_title("Loss Curves (Log Scale)")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.save_dir, f"{self.model_name}_training.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [saved] {path}")
