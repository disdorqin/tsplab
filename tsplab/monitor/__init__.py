"""
tsplab.monitor — Training monitoring and diagnosis.

Features:
  - TrainingMonitor: log loss, auto-diagnose, auto-plot
  - LRFinder: learning rate range test
  - GradientProbe: automatic gradient health check
  - Smart early stopping with stopping_reason
"""

from tsplab.monitor.curve import TrainingMonitor
from tsplab.monitor.lr_finder import LRFinder
from tsplab.monitor.gradient_probe import GradientProbe

__all__ = ["TrainingMonitor", "LRFinder", "GradientProbe"]
