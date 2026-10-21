"""Cost model: turns decision-date weights into net daily returns.

Timing: the weight decided at day t's close earns day t+1's return, so
    ret(t+1) = sum_i w_exec_i(t) * r_i(t+1) - cost(t+1) - funding(t+1)
    cost(t+1) = cost_per_side * sum_i |w_exec_i(t) - w_exec_i(t-1)|
    funding(t+1) = sum_i w_exec_i(t) * funding_i(t+1)
The output is indexed by the return day (weights.index[1:]).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _executed_weights(
    weights: pd.DataFrame,
    signal_changed: pd.DataFrame,
    rebalance_threshold: float | None,
) -> pd.DataFrame:
    target = weights.fillna(0.0)
    if rebalance_threshold is None:
        return target
    tgt = target.to_numpy(dtype=float)
    sig = (
        signal_changed.reindex(index=weights.index, columns=weights.columns)
        .fillna(True)
        .to_numpy(dtype=bool)
    )
    out = np.zeros_like(tgt)
    current = np.zeros(tgt.shape[1])
    for t in range(tgt.shape[0]):
        cur_abs = np.abs(current)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(tgt[t] - current) / cur_abs
        execute = sig[t] | (current == 0) | (rel > rebalance_threshold)
        current = np.where(execute, tgt[t], current)
        out[t] = current
    return pd.DataFrame(out, index=weights.index, columns=weights.columns)


def net_returns(
    weights: pd.DataFrame,
    signal_changed: pd.DataFrame,
    close: pd.DataFrame,
    funding: pd.DataFrame | None,
    cost_per_side: float,
    rebalance_threshold: float | None,
) -> pd.Series:
    """Net daily portfolio returns after costs and funding (see module doc).

    A NaN return (coin not listed / delisted) counts as 0, so a position
    closes at the last available price; NaN funding counts as 0.
    """
    w_exec = _executed_weights(weights, signal_changed, rebalance_threshold)
    rets = (
        close.reindex(index=weights.index, columns=weights.columns)
        .pct_change(fill_method=None)
        .fillna(0.0)
    )
    prev_w = w_exec.shift(1).fillna(0.0)
    turnover = (w_exec - prev_w).abs().sum(axis=1)

    gross = (w_exec * rets.shift(-1)).sum(axis=1)
    cost = cost_per_side * turnover
    if funding is None:
        fund = pd.Series(0.0, index=weights.index)
    else:
        f = funding.reindex(index=weights.index, columns=weights.columns).fillna(0.0)
        fund = (w_exec * f.shift(-1)).sum(axis=1)

    net = gross - cost - fund
    net = net.iloc[:-1]
    net.index = weights.index[1:]
    net.name = "net_return"
    return net
