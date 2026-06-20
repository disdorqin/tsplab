# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Issue and PR templates
- GitHub Actions CI for Python 3.10, 3.11, 3.12
- Coverage reporting in CI

## [0.1.0] - 2026-06-20

### Added
- **audit** module: 6-check data leakage detection matrix
  - Sequence generation timing check
  - Sliding window boundary check
  - Normalization fitting check
  - CV fold overlap check
  - Covariate availability check
  - RMSE Gain quantification
- **monitor** module: Training curve monitoring and diagnosis
  - `TrainingMonitor`: auto-diagnose 7 training patterns (overfitting, plateau, instability, etc.)
  - `LRFinder`: Learning rate range test
  - `GradientProbe`: automatic gradient health monitoring
- **baselines** module: Prevents DLinear trap
  - Naive, SeasonalNaive, Drift baselines
  - DLinear / NLinear (AAAI 2023)
  - Decision Tree baseline
  - Auto-comparison with your model + warning system
- **tune** module: Time-series-aware hyperparameter tuning
  - Time series CV (expanding/rolling window)
  - Optuna integration with HyperbandPruner
  - Periodic refitting support
  - Hyperparameter importance analysis
- **eval** module: Evaluation metrics and difficulty rating
  - MAE, RMSE, MAPE, SMAPE, MASE, WAPE
  - DataDifficulty: signal-to-noise, autocorrelation, entropy, seasonality
- **experiment** module: Experiment tracking
  - SQLite storage
  - Optional auto-git-commit
  - Experiment comparison utilities
- **viz** module: Visualization
  - Training curves with best-epoch markers
  - Prediction vs ground truth
  - Residual analysis (time series, histogram, Q-Q plot)
- **data** module: Time-series-safe splitting and windowing

[Unreleased]: https://github.com/disdorqin/tsplab/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/disdorqin/tsplab/releases/tag/v0.1.0
