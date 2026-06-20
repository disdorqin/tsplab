"""
tsplab.viz — Visualization tools for training, predictions, and residuals.
"""

from tsplab.viz.training import plot_training_curves
from tsplab.viz.prediction import plot_predictions
from tsplab.viz.residual import plot_residuals

__all__ = ["plot_training_curves", "plot_predictions", "plot_residuals"]
