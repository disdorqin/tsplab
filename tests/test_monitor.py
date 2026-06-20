"""Tests for tsplab.monitor module."""

import numpy as np
from tsplab.monitor import TrainingMonitor
from tsplab.monitor.curve import StopReason


def test_monitor_basic():
    """Monitor should track basic metrics."""
    monitor = TrainingMonitor(patience=5)
    for epoch in range(10):
        train_loss = 0.5 * np.exp(-epoch / 5) + 0.05
        val_loss = 0.55 * np.exp(-epoch / 5) + 0.08
        status = monitor.log(epoch, train_loss, val_loss)
        assert not status.should_stop
    assert len(monitor.train_losses) == 10
    assert monitor.best_epoch >= 0


def test_monitor_early_stop():
    """Monitor should stop when patience exhausted."""
    monitor = TrainingMonitor(patience=3, min_delta=0.001)
    for epoch in range(20):
        train_loss = 0.5 * np.exp(-epoch / 3) + 0.05
        val_loss = 0.6 + 0.01 * epoch  # Val loss increasing
        status = monitor.log(epoch, train_loss, val_loss)
        if status.should_stop:
            break
    assert status.should_stop
    assert status.reason in [StopReason.PATIENCE_EXHAUSTED, StopReason.OVERFITTING, StopReason.DIVERGENCE]


def test_monitor_nan_detection():
    """Monitor should detect NaN."""
    monitor = TrainingMonitor(patience=10, check_finite=True)
    monitor.log(0, 0.5, 0.6)
    status = monitor.log(1, float("nan"), 0.6)
    assert status.should_stop
    assert status.reason == StopReason.NON_FINITE_METRIC


def test_monitor_diagnosis():
    """Monitor should provide diagnosis text."""
    monitor = TrainingMonitor(patience=20)
    for epoch in range(10):
        train_loss = 0.5 * np.exp(-epoch / 5) + 0.05
        val_loss = 0.55 * np.exp(-epoch / 5) + 0.08
        status = monitor.log(epoch, train_loss, val_loss)
    assert len(status.diagnosis) > 0


def test_monitor_report():
    """Monitor should generate report without errors."""
    monitor = TrainingMonitor(patience=20, save_dir="./test_runs")
    for epoch in range(10):
        monitor.log(epoch, 0.5 * np.exp(-epoch / 5), 0.55 * np.exp(-epoch / 5))
    monitor.report()  # Should not raise


def test_monitor_overfitting_detection():
    """Monitor should detect overfitting pattern."""
    monitor = TrainingMonitor(patience=20)
    for epoch in range(30):
        train_loss = 0.5 * np.exp(-epoch / 5) + 0.01
        # Val loss: decreases then increases
        if epoch < 10:
            val_loss = 0.55 * np.exp(-epoch / 5) + 0.05
        else:
            val_loss = 0.05 + 0.01 * (epoch - 10)
        status = monitor.log(epoch, train_loss, val_loss)
    # After 30 epochs with increasing val loss, should have stopped
    assert status.should_stop or "Overfitting" in status.diagnosis or "overfitting" in status.diagnosis
