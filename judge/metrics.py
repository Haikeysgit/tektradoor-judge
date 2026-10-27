"""Scoring metrics shared by the verdict and the paper reproduction.

Sharpe = mean / std(ddof=1) * sqrt(365), risk-free 0; std 0 or fewer than
2 points gives 0. Max drawdown is on the compounded daily equity curve.
This module decides reproduction numbers, so the Test-time gate ties it to
the reproduction (judge.REPRO_CODE_PATHS).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365


def sharpe(daily: pd.Series) -> float:
    x = pd.Series(daily, dtype=float).dropna().to_numpy()
    if len(x) < 2:
        return 0.0
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-15:  # float noise on a constant series counts as 0
        return 0.0
    return float(x.mean() / sd * math.sqrt(DAYS_PER_YEAR))


def max_drawdown(daily: pd.Series) -> float:
    """Largest peak-to-trough fall of the compounded equity curve, starting
    from equity 1.0 before the first return. Positive fraction."""
    x = pd.Series(daily, dtype=float).fillna(0.0).to_numpy()
    if len(x) == 0:
        return 0.0
    equity = np.concatenate([[1.0], np.cumprod(1.0 + x)])
    peak = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peak))


def total_return(r: pd.Series) -> float:
    """Compounded total return of daily returns (0 for none)."""
    return float(np.prod(1.0 + r.to_numpy(dtype=float)) - 1.0) if len(r) else 0.0
