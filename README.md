<div align="center">

<img src="assets/banner.svg" width="100%" alt="tsplab Banner" />

</div>

## Forecasting Experiment Command Center

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org/)
[![License](https://img.shields.io/github/license/disdorqin/tsplab?style=flat-square&color=A3E635&logo=opensourceinitiative)](LICENSE)
[![Stars](https://img.shields.io/github/stars/disdorqin/tsplab?style=flat-square&color=00D4FF&logo=github)](https://github.com/disdorqin/tsplab/stargazers)
[![Status](https://img.shields.io/badge/STATUS-ACTIVE-00D4FF?style=flat-square&logo=circle&logoColor=white)](https://github.com/disdorqin/tsplab)
[![Forecasting](https://img.shields.io/badge/DOMAIN-Time%20Series%20Forecasting-00D4FF?style=flat-square)](https://github.com/disdorqin/tsplab)

---

> TSPLab (Time Series Prediction Lab) is a command center for forecasting experiment management. Audit, monitor, tune, and compare time series forecasting experiments — with built-in support for NSE optimization and power market use cases.

## Why TSPLab Exists

Forecasting experiments are hard to track: which hyperparameters were used? How does this model compare to last week's? Is the experiment reproducible? TSPLab gives you a structured way to audit, monitor, tune, and compare forecasting experiments.

## Features

- **Experiment Audit** — structured logging of all experiment parameters and results  
- **Monitor** — real-time tracking of experiment progress and metric convergence  
- **Tuner** — hyperparameter optimization with experiment comparison  
- **Comparator** — side-by-side model comparison with NSE/MAE/R2 metrics  
- **Reproducibility** — every experiment is fully parameterized and replayable  
- **Power Market Ready** — built-in support for electricity price forecasting benchmarks  

## Architecture

<div align="center">
  <img src="assets/architecture.svg" width="100%" alt="Architecture" />
</div>

## Quick Start

```bash
# Clone and install
git clone https://github.com/disdorqin/tsplab.git
cd tsplab
pip install -e ".[dev]"

# Run an experiment
python -m tsplab experiment run --config examples/lstm_electricity.yaml

# Compare experiments
python -m tsplab compare --exp1 runs/001 --exp2 runs/002

# Launch monitor dashboard
python -m tsplab monitor --port 8080
```

## Example: Experiment Config

```yaml
# examples/lstm_electricity.yaml
model: LSTM
data: electricity_price.csv
features:
  - lag_24
  - lag_168
  - day_of_week
  - temperature
hyperparams:
  hidden_size: 64
  num_layers: 2
  learning_rate: 0.001
  epochs: 100
metrics:
  - NSE
  - MAE
  - R2
output: runs/lstm_v1
```

## Roadmap

- [x] Experiment audit logging
- [x] Basic monitor
- [ ] Hyperparameter tuner
- [ ] Experiment comparator dashboard
- [ ] Integration with DARIS for automated experiment pipelines
- [ ] Cloud experiment tracking

## Tech Stack

Python · pandas · numpy · PyTorch · scikit-learn · YAML

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=disdorqin/tsplab&type=Date)](https://star-history.com/#disdorqin/tsplab&Date)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
