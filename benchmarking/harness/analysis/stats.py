"""Statistics used by the summaries.

- Per case: the median of the timed samples and the inter-quartile range
  (25th and 75th percentiles, linear interpolation between order statistics).
- Across cases: the geometric mean of per-case ratios, with a 95 % percentile
  bootstrap interval obtained by resampling cases with replacement.
"""

from __future__ import annotations

import math

import numpy as np

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261006
CONFIDENCE = 0.95


def median(values: list[float]) -> float:
    return float(np.median(values))


def quartiles(values: list[float]) -> tuple[float, float]:
    q1, q3 = np.percentile(values, [25, 75])
    return float(q1), float(q3)


def geometric_mean(values: list[float]) -> float:
    if not values or any(v <= 0 for v in values):
        raise ValueError("geometric mean needs positive values")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def bootstrap_geomean_interval(ratios: list[float], replicates: int = BOOTSTRAP_REPLICATES,
                               seed: int = BOOTSTRAP_SEED,
                               confidence: float = CONFIDENCE) -> tuple[float, float]:
    """Percentile bootstrap interval of the geometric mean of `ratios`."""
    logs = np.log(np.asarray(ratios, dtype=float))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(logs), size=(replicates, len(logs)))
    means = logs[picks].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(math.exp(low)), float(math.exp(high))
