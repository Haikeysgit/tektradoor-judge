"""Scoring: Sharpe, max drawdown, the lazy-investor benchmark, verdicts and
luck odds (how often a no-skill random strategy passes the same check).

Rules (Global Constraints):
  test, final_alone: Sharpe > 1.0 AND max_dd < 0.25 AND Sharpe > bench Sharpe
  final ("holds up"): max_dd < 0.25 AND Sharpe >= bench Sharpe
Sharpe = mean / std(ddof=1) * sqrt(365), risk-free 0; std 0 or fewer than
2 points gives 0.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

from judge.costs import net_returns
from judge.strategies.donchian import _allocation_frame, _apply_allocation, allocate, vol_scale

DAYS_PER_YEAR = 365
SHARPE_MIN = 1.0
MAX_DD_LIMIT = 0.25
KINDS = ("test", "final", "final_alone")

Kind = Literal["test", "final", "final_alone"]


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


def benchmark_returns(
    close: pd.DataFrame,
    funding: pd.DataFrame | None,
    universe_by_month: dict,
    cost_per_side: float,
) -> pd.Series:
    """Equal-weight 1x hold of each month's universe, through costs and funding.

    Target weights are 1/len(universe) per listed member, so trades happen only
    when the universe changes or a coin delists.
    """
    held = pd.DataFrame(1.0, index=close.index, columns=close.columns).where(close.notna(), 0.0)
    weights, changed = allocate(held, close, universe_by_month, gross_cap=1.0)
    return net_returns(weights, changed, close, funding, cost_per_side, rebalance_threshold=None)


def verdict(strategy: pd.Series, bench: pd.Series, kind: Kind) -> dict:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    s, dd, b = sharpe(strategy), max_drawdown(strategy), sharpe(bench)
    if kind == "final":
        rules = {"max_dd_below_25pct": dd < MAX_DD_LIMIT, "sharpe_holds_vs_bench": s >= b}
    else:
        rules = {
            "sharpe_above_1": s > SHARPE_MIN,
            "max_dd_below_25pct": dd < MAX_DD_LIMIT,
            "sharpe_beats_bench": s > b,
        }
    return {"passed": bool(all(rules.values())), "sharpe": s, "max_dd": dd, "bench_sharpe": b, "rules": rules}


def _random_states(rng: np.random.Generator, shape: tuple[int, int], flip_prob: float, states: np.ndarray) -> np.ndarray:
    """State paths: start 0; each day with prob flip_prob draw a new state
    uniformly from ``states`` (it may equal the current one), else keep."""
    t_len, n_coins = shape
    flips = rng.random(shape) < flip_prob
    draws = rng.choice(states, size=shape)
    last = np.where(flips, np.arange(t_len)[:, None], -1)
    last = np.maximum.accumulate(last, axis=0)
    picked = np.take_along_axis(draws, np.clip(last, 0, None), axis=0)
    return np.where(last >= 0, picked, 0).astype(float)


def luck_odds(
    close: pd.DataFrame,
    funding: pd.DataFrame | None,
    universe_by_month: dict,
    long_short: bool,
    flip_prob: float,
    n_sims: int,
    seed: int,
    kind: Kind,
    cost_per_side: float,
    gross_cap: float | None,
    bench: pd.Series | None = None,
) -> float:
    """Share of n_sims random-position strategies that pass ``verdict``.

    Each sim: per coin a random state path (see _random_states), coin weight =
    vol_scale(close) * state, then the same allocation as ``allocate`` (gross
    cap), net_returns without threshold, and verdict against the benchmark.
    """
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")
    rng = np.random.default_rng(seed)
    if bench is None:
        bench = benchmark_returns(close, funding, universe_by_month, cost_per_side)
    states = np.array([-1, 0, 1] if long_short else [0, 1])

    # Everything that does not depend on the random path is computed once;
    # this is allocate() split so the masks are not rebuilt per sim.
    scale = np.column_stack([vol_scale(close[c]).to_numpy(dtype=float) for c in close.columns])
    active, size, alloc_flags = _allocation_frame(close, universe_by_month, None)
    shape = close.shape

    passed = 0
    for _ in range(n_sims):
        state = _random_states(rng, shape, flip_prob, states)
        prev = np.vstack([np.zeros((1, shape[1])), state[:-1]])
        coin_w = pd.DataFrame(scale * state, index=close.index, columns=close.columns)
        weights = _apply_allocation(coin_w, active, size, gross_cap)
        changed = alloc_flags | pd.DataFrame(state != prev, index=close.index, columns=close.columns)
        sim = net_returns(weights, changed, close, funding, cost_per_side, rebalance_threshold=None)
        passed += verdict(sim, bench, kind)["passed"]
    return passed / n_sims
