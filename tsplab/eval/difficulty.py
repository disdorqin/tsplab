"""Data difficulty assessment.

Assesses how predictable a time series is BEFORE training any model.
Helps set realistic expectations: if the data is essentially noise,
no model will work well, and that's not your fault.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


class DataDifficulty:
    """Assess the inherent predictability of a time series.

    Usage::

        diff = DataDifficulty(data, timestamps)
        diff.assess()
        # -> signal-to-noise ratio, autocorrelation, entropy, seasonality...
    """

    def __init__(self, data: np.ndarray, timestamps: np.ndarray | None = None) -> None:
        self.data = np.asarray(data, dtype=float).ravel()
        self.timestamps = timestamps

    def signal_to_noise_ratio(self) -> float:
        """SNR: |mean| / std. Higher = more signal, more predictable."""
        std = np.std(self.data)
        if std == 0:
            return float("inf")
        return abs(np.mean(self.data)) / std

    def autocorrelation(self, lag: int = 1) -> float:
        """Lag-1 autocorrelation. Higher = more predictable."""
        n = len(self.data)
        if n <= lag:
            return 0.0
        return float(np.corrcoef(self.data[:-lag], self.data[lag:])[0, 1])

    def approximate_entropy(self, m: int = 2, r: float | None = None) -> float:
        """Approximate entropy. Higher = more complex/chaotic."""
        if r is None:
            r = 0.2 * np.std(self.data)

        n = len(self.data)
        if n < m + 1:
            return 0.0

        def _phi(mm: int) -> float:
            patterns = np.array([self.data[i:i + mm] for i in range(n - mm + 1)])
            counts = []
            for i in range(len(patterns)):
                dist = np.max(np.abs(patterns - patterns[i]), axis=1)
                counts.append(np.sum(dist <= r))
            counts = np.array(counts, dtype=float)
            counts = counts / (n - mm + 1)
            return np.mean(np.log(counts + 1e-10))

        return _phi(m) - _phi(m + 1)

    def seasonality_strength(self, period: int = 24) -> float:
        """Strength of seasonality (0-1). Based on seasonal decomposition."""
        from scipy.signal import periodogram

        if len(self.data) < 2 * period:
            return 0.0

        # Compute periodogram
        freqs, powers = periodogram(self.data, detrend="linear")

        # Find power at the seasonal frequency
        seasonal_freq = 1.0 / period
        idx = np.argmin(np.abs(freqs - seasonal_freq))
        seasonal_power = powers[idx]

        total_power = np.sum(powers)
        if total_power == 0:
            return 0.0

        return float(seasonal_power / total_power)

    def trend_strength(self) -> float:
        """Strength of linear trend (R^2 of linear fit)."""
        n = len(self.data)
        x = np.arange(n)
        slope, intercept, r_value, _, _ = stats.linregress(x, self.data)
        return float(r_value ** 2)

    def white_noise_ratio(self, period: int = 24) -> float:
        """Fraction of variance attributable to white noise."""
        if len(self.data) < 2 * period:
            return 1.0

        # Autocorrelation at lags 1 to period
        from statsmodels.tsa.stattools import acf

        acf_values = acf(self.data, nlags=period, fft=True)
        # If all ACF values are within +/- 1.96/sqrt(n), it's white noise
        n = len(self.data)
        threshold = 1.96 / np.sqrt(n)
        significant = np.sum(np.abs(acf_values[1:]) > threshold)
        return 1.0 - significant / period

    def assess(self) -> dict[str, any]:
        """Run all assessments and print a summary."""
        snr = self.signal_to_noise_ratio()
        acf1 = self.autocorrelation(1)
        entropy = self.approximate_entropy()
        season = self.seasonality_strength()
        trend = self.trend_strength()
        noise = self.white_noise_ratio()

        # Overall difficulty rating
        score = 0
        if acf1 > 0.7:
            score += 2
        elif acf1 > 0.4:
            score += 1

        if season > 0.1:
            score += 2
        elif season > 0.05:
            score += 1

        if trend > 0.3:
            score += 1

        if entropy < 0.3:
            score += 2
        elif entropy < 0.6:
            score += 1

        if noise < 0.3:
            score += 2
        elif noise < 0.6:
            score += 1

        if score >= 6:
            rating = "[Easy] Highly predictable — simple models should work well"
        elif score >= 4:
            rating = "[Medium] Moderate predictability — DL models may help"
        elif score >= 2:
            rating = "[Hard] Low predictability — DL advantage limited"
        else:
            rating = "[Very Hard] Near-random — consider if this data is truly forecastable"

        # Print report
        print("=" * 60)
        print("  TSPLab Data Difficulty Assessment")
        print("=" * 60)
        print()
        print(f"  Signal-to-Noise Ratio: {snr:.3f}")
        print(f"  Autocorrelation (lag=1): {acf1:.3f}")
        print(f"  Approximate Entropy: {entropy:.3f}")
        print(f"  Seasonality Strength: {season:.3f}")
        print(f"  Trend Strength: {trend:.3f}")
        print(f"  White Noise Ratio: {noise:.1%}")
        print()
        print(f"  Overall Rating: {rating}")
        print()

        if "Very Hard" in rating:
            print("  [!] This data may be close to random walk.")
            print("      Even the best models may not beat Naive baseline.")
            print("      Consider: adding external features, changing target, or")
            print("      using a different forecasting horizon.")
        elif "Hard" in rating:
            print("  [!] Deep learning may not outperform simple models here.")
            print("      Focus on feature engineering rather than model complexity.")
        print("=" * 60)

        return {
            "snr": snr,
            "autocorrelation_lag1": acf1,
            "approximate_entropy": entropy,
            "seasonality_strength": season,
            "trend_strength": trend,
            "white_noise_ratio": noise,
            "score": score,
            "rating": rating,
        }
