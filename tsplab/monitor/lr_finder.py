"""
Learning Rate Range Test (LRFinder).

Inspired by Leslie Smith's cyclical learning rate paper.
Runs a short pre-training pass with exponentially increasing LR,
then suggests the optimal learning rate.
"""

from __future__ import annotations

import os
import copy

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class LRFinder:
    """Learning rate range test.

    Usage::

        finder = LRFinder(model, optimizer, criterion)
        finder.range_test(dataloader, start_lr=1e-7, end_lr=1, num_iter=100)
        finder.plot()
        finder.suggest()  # -> "suggested lr = 3.2e-4"
    """

    def __init__(self, model, optimizer, criterion) -> None:
        """
        Parameters
        ----------
        model : PyTorch model (must have .parameters(), .train(), .forward())
        optimizer : PyTorch optimizer
        criterion : loss function
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.lrs: list[float] = []
        self.losses: list[float] = []

        # Save initial state for restoration
        self._initial_state = copy.deepcopy(model.state_dict())
        self._initial_optimizer = copy.deepcopy(optimizer.state_dict())

    def range_test(
        self,
        dataloader,
        start_lr: float = 1e-7,
        end_lr: float = 1.0,
        num_iter: int = 100,
        smooth: float = 0.05,
    ) -> None:
        """Run the LR range test.

        Parameters
        ----------
        dataloader : yields (inputs, targets) batches
        start_lr : starting learning rate
        end_lr : ending learning rate
        num_iter : number of iterations to run
        smooth : smoothing factor for loss (EMA)
        """
        # Set starting LR
        for pg in self.optimizer.param_groups:
            pg["lr"] = start_lr

        # Compute LR multiplier per step
        lr_multiplier = (end_lr / start_lr) ** (1.0 / num_iter)

        self.model.train()
        iterator = iter(dataloader)

        avg_loss = 0.0
        best_loss = float("inf")

        for iteration in range(num_iter):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                inputs, targets = next(iterator)

            # Forward + backward
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            # Record
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.lrs.append(current_lr)

            # Smoothed loss
            current_loss = loss.item()
            if iteration == 0:
                avg_loss = current_loss
            else:
                avg_loss = smooth * current_loss + (1 - smooth) * avg_loss

            # Stop if loss explodes
            if avg_loss > best_loss * 4:
                self.losses.append(avg_loss)
                break

            if avg_loss < best_loss:
                best_loss = avg_loss

            self.losses.append(avg_loss)

            # Update LR
            for pg in self.optimizer.param_groups:
                pg["lr"] *= lr_multiplier

        # Restore initial state
        self.model.load_state_dict(self._initial_state)
        self.optimizer.load_state_dict(self._initial_optimizer)

    def suggest(self) -> float:
        """Suggest the optimal learning rate based on the steepest descent."""
        if len(self.losses) < 5:
            print("Not enough data to suggest LR.")
            return 1e-3

        # Find the point of steepest descent (most negative gradient)
        losses = np.array(self.losses)
        lrs = np.array(self.lrs)

        # Smooth losses
        window = max(3, len(losses) // 10)
        kernel = np.ones(window) / window
        smoothed = np.convolve(losses, kernel, mode="valid")

        # Compute gradients
        log_lrs = np.log10(lrs[:len(smoothed)])
        gradients = np.gradient(smoothed, log_lrs)

        # Find steepest negative gradient
        best_idx = np.argmin(gradients)
        suggested_lr = lrs[best_idx]

        print(f"Suggested LR: {suggested_lr:.2e} "
              f"(steepest descent at log_lr={log_lrs[best_idx]:.2f})")
        return suggested_lr

    def plot(self, save_path: str | None = None) -> None:
        """Plot loss vs learning rate."""
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(self.lrs, self.losses, color="#2196F3", linewidth=1.5)
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate (log scale)")
        ax.set_ylabel("Loss")
        ax.set_title("LR Range Test")
        ax.grid(True, alpha=0.3, which="both")

        # Mark suggested LR
        if len(self.losses) >= 5:
            suggested = self.suggest()
            ax.axvline(x=suggested, color="#FF5722", linestyle="--", alpha=0.7,
                       label=f"Suggested: {suggested:.2e}")
            ax.legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[saved] {save_path}")
        else:
            plt.show()
        plt.close(fig)
