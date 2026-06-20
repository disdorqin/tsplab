"""
tsplab.baselines — Simple baselines and automatic comparison.

Prevents the "DLinear trap": deep learning models that can't beat
trivially simple baselines (Naive, Linear, Decision Tree).
"""

from tsplab.baselines.naive import NaiveBaseline, SeasonalNaive, DriftBaseline
from tsplab.baselines.linear import DLinear, NLinear
from tsplab.baselines.tree import TreeBaseline
from tsplab.baselines.compare import run_all_baselines, compare_with_baselines

__all__ = [
    "NaiveBaseline",
    "SeasonalNaive",
    "DriftBaseline",
    "DLinear",
    "NLinear",
    "TreeBaseline",
    "run_all_baselines",
    "compare_with_baselines",
]
