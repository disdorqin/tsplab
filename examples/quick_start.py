"""
TSPLab Quick Start Example

This example demonstrates the core workflow:
  1. Generate synthetic time series data
  2. Run data difficulty assessment
  3. Run all baselines
  4. Audit data for leakage
  5. Use TrainingMonitor (simulated)
"""

import numpy as np
from tsplab import audit, monitor, baselines, eval


def main():
    # --- 1. Generate synthetic data ---
    print("Generating synthetic time series...")
    np.random.seed(42)
    t = np.arange(500)
    trend = 0.02 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 24)
    noise = np.random.normal(0, 1, 500)
    data = trend + seasonal + noise

    # Split
    train, test = data[:400], data[400:]
    print(f"Train: {len(train)} samples, Test: {len(test)} samples\n")

    # --- 2. Data difficulty assessment ---
    print("=" * 60)
    print("STEP 1: Data Difficulty Assessment")
    print("=" * 60)
    difficulty = eval.DataDifficulty(data)
    difficulty.assess()
    print()

    # --- 3. Run all baselines ---
    print("=" * 60)
    print("STEP 2: Run All Baselines")
    print("=" * 60)
    results = baselines.run_all_baselines(
        y_train=train,
        y_test=test,
        horizon=50,
        lookback=24,
        seasonal_length=24,
    )

    # Print baseline results
    for name, result in results.items():
        if "error" in result:
            print(f"  {name}: ERROR - {result['error']}")
        else:
            print(f"  {name}: MAE={result['mae']:.4f}, SMAPE={result['smape']:.2f}%")
    print()

    # --- 4. Simulate a model result and compare ---
    print("=" * 60)
    print("STEP 3: Compare With Baselines")
    print("=" * 60)
    # Simulate: your model gets MAE = 0.85
    model_metrics = {"mae": 0.85, "smape": 15.2, "rmse": 1.1}
    baselines.compare_with_baselines(model_metrics, results, metric="mae")
    print()

    # --- 5. Data audit ---
    print("=" * 60)
    print("STEP 4: Data Audit (Leakage Check)")
    print("=" * 60)
    from tsplab.data import time_series_split, create_windows

    # Correct way: split first, then create windows
    train, test = time_series_split(data, test_ratio=0.2, gap=0)
    X_train, Y_train = create_windows(train, lookback=24, horizon=1)
    X_test, Y_test = create_windows(test, lookback=24, horizon=1)

    report = audit.AuditReport()
    report.check_sequence_generation(X_train, X_test)
    report.check_window_boundaries(
        train_windows=np.arange(0, len(X_train)),
        test_windows=np.arange(len(X_train), len(X_train) + len(X_test)),
        split_index=len(train),
        window_size=24,
    )
    report.summary()
    print()

    # --- 6. Training monitor (simulated) ---
    print("=" * 60)
    print("STEP 5: Training Monitor (Simulated)")
    print("=" * 60)
    mon = monitor.TrainingMonitor(patience=10, save_dir="./runs/example")

    # Simulate training with healthy convergence
    for epoch in range(50):
        train_loss = 0.5 * np.exp(-epoch / 15) + 0.05 + np.random.normal(0, 0.005)
        val_loss = 0.55 * np.exp(-epoch / 15) + 0.08 + np.random.normal(0, 0.005)
        status = mon.log(epoch, train_loss, val_loss)
        if status.should_stop:
            print(f"Early stop at epoch {epoch}: {status.reason.value}")
            break

    mon.report()
    print()
    print("Done! Check ./runs/example/ for saved plots.")


if __name__ == "__main__":
    main()
