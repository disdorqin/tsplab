# TSPLab — Time Series Prediction Lab

> **不是又一个时序预测框架，而是你实验的"质检站 + 驾驶舱"。**
>
> 你继续用自己的模型代码，TSPLab 负责帮你查数据泄漏、诊断训练、公平对比、自动调参、管理实验。

## 为什么需要 TSPLab？

做时序预测研究时，你是不是也遇到过这些问题：

- 深度学习模型跑不过决策树，但不知道为什么
- 复现别人的模型效果差很多，怀疑是自己代码的问题
- 训练只看最终指标（MAE / SMAPE），从不关注 loss 曲线
- 调参全靠手动试，没有系统化的方法
- 实验记录靠记忆，不知道上次那个结果用的是哪版代码

TSPLab 就是为了解决这些问题而设计的。

## 安装

```bash
# 从 GitHub 安装（开发阶段）
pip install git+https://github.com/tsplab/tsplab.git

# 或从源码安装（可编辑模式）
git clone https://github.com/tsplab/tsplab.git
cd tsplab
pip install -e .

# 可选依赖
pip install -e ".[tune,viz,tree]"  # Optuna + Plotly + XGBoost
```

## 核心功能

### 1. 数据审计 — 六重泄漏检测矩阵

```python
from tsplab.audit import AuditReport

report = AuditReport()
report.check_sequence_generation(X_train, X_test, timestamps)
report.check_window_boundaries(windows, split_time)
report.check_normalization_fitting(scaler, X_train, X_test)
report.check_fold_overlap(cv_splits)
report.check_covariate_availability(covariates, horizon)
report.summary()
```

检测项：
- **序列生成时机** — 先切分还是先生成窗口？
- **窗口边界** — 窗口是否跨越 train/test 边界？
- **归一化泄漏** — scaler 是否偷看了测试集？
- **折叠重叠** — CV 各 fold 是否时间相邻？
- **协变量可得性** — future covariate 部署时真拿得到？
- **RMSE Gain 量化** — 泄漏到底有多大影响？

### 2. 训练监控 — 7 种曲线模式自动诊断

```python
from tsplab.monitor import TrainingMonitor

monitor = TrainingMonitor(patience=15, save_dir="./runs/exp_001")

for epoch in range(epochs):
    train_loss = model.train_one_epoch(...)
    val_loss = model.validate(...)

    status = monitor.log(epoch, train_loss, val_loss)
    if status.should_stop:
        print(f"停止原因: {status.reason}")
        break

monitor.report()  # 自动出图 + 诊断报告
```

自动诊断模式：Healthy Convergence / Overfitting / Instability / Slow Convergence / Still Improving / Early Plateau / High Variance

### 3. 基线对比 — 防 DLinear 陷阱

```python
from tsplab.baselines import run_all_baselines, compare_with_baselines

baselines = run_all_baselines(data, horizon=24)
# 自动跑: Naive / SeasonalNaive / Drift / DLinear / ARIMA / XGBoost

compare_with_baselines(my_model_results, baselines)
# 你的模型打不过 Naive -> 红色警告
# 打不过 DLinear -> 黄色警告
```

### 4. 智能调参 — 时序感知的自动搜索

```python
from tsplab.tune import TimeSeriesTuner

tuner = TimeSeriesTuner(
    model_factory=my_model_fn,
    cv_strategy="expanding",
    n_windows=5,
    gap=1,
    pruner="hyperband",
    timeout=1800,
)

study = tuner.optimize(X, y, timestamps, metric="smape")
tuner.plot_optimization_history()
tuner.plot_param_importances()
```

### 5. 数据集难度评级

```python
from tsplab.eval import DataDifficulty

diff = DataDifficulty(data, timestamps)
diff.assess()
# 信号噪声比 / 自相关 / 季节性强度 / 近似熵 -> 总评级
```

### 6. 实验管理

```python
from tsplab.experiment import ExperimentTracker

tracker = ExperimentTracker(name="informer_etth1")
tracker.log_params({"model": "informer", "lr": 1e-4, "bs": 32})
tracker.log_metrics({"mae": 0.321, "smape": 12.5})
tracker.save()  # 自动 git commit
```

## 设计原则

- **不侵入你的代码** — 不需要改成某个框架的写法，各模块独立 import
- **轻量依赖** — 核心只依赖 numpy/pandas/sklearn，不强制安装 PyTorch
- **时序优先** — 所有交叉验证、切分、调参都遵循时间因果性
- **可诊断** — 不是给你一个数字，而是告诉你为什么

## 项目结构

```
tsplab/
  audit/       # 数据审计（泄漏检测）
  monitor/     # 训练监控（曲线诊断 + 早停 + LR Finder）
  baselines/   # 基线对比（Naive / DLinear / 树模型）
  tune/        # 自动调参（时序CV + Optuna剪枝）
  eval/        # 评估（指标 + 难度评级 + 公平性审计）
  experiment/  # 实验管理（记录 + git hook）
  viz/         # 可视化（训练曲线 + 预测对比 + 残差分析）
```

## License

MIT
