"""Donchian Combo: Zarattini, Pagani & Barbon (2025), "Catching Crypto Trends",
SSRN 5209907, sections 4.1-4.3 and Appendix (long-short).

All weights are indexed by decision date: the weight on day t is decided at
day t's close and earns day t+1's return (the cost task applies that shift).

Channels are close-based (not high/low):
    Up_n(t)   = max(Close_t .. Close_{t-n+1})
    Down_n(t) = min(Close_t .. Close_{t-n+1})
    Mid_n(t)  = 0.5 * (Up_n(t) + Down_n(t))
A channel needs n closes; before that there is no signal.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

LOOKBACKS = (5, 10, 20, 30, 60, 90, 150, 250, 360)
VOL_TARGET = 0.25
VOL_WINDOW = 90
PER_COIN_CAP = 2.0
DAYS_PER_YEAR = 365


def signal_with_stop(close: pd.Series, n: int, long_short: bool) -> pd.DataFrame:
    """Position state machine for one lookback. Columns: pos in {-1,0,1}, stop.

    ``stop`` on day t is the stop in force for day t (the one the day-t close
    is tested against). On an entry day it is the initial stop, the channel
    mid at entry, which is also the stop in force for the next day. NaN while
    flat.

    Rules, evaluated at each close t (paper 4.1 and Appendix):
      * NaN close or incomplete channel: flat, state reset.
      * Close_t == Up_n(t): long. A fresh entry sets stop = Mid_n(t). A long
        already open stays long and keeps its stop (so the stop never moves
        down). Entry takes precedence over the stop test, as in the paper's
        Pos_t definition. Long wins a tie with a short entry (flat window).
      * long_short and Close_t == Down_n(t): short, mirrored.
      * Otherwise an open long exits when Close_t <= stop_t, an open short
        when Close_t >= stop_t; else the position carries.
      * While a position stays open after day t:
        long stop_{t+1} = max(stop_t, Mid_n(t)); short stop_{t+1} = min(stop_t, Mid_n(t)).
    """
    c = close.to_numpy(dtype=float)
    roll = close.rolling(n, min_periods=n)
    up = roll.max().to_numpy(dtype=float)
    down = roll.min().to_numpy(dtype=float)
    mid = 0.5 * (up + down)

    pos_out = np.zeros(len(c), dtype=np.int64)
    stop_out = np.full(len(c), np.nan)
    pos, stop = 0, math.nan
    for t in range(len(c)):
        ct = c[t]
        if math.isnan(ct) or math.isnan(up[t]):
            pos, stop = 0, math.nan
            continue
        if ct == up[t]:
            if pos != 1:
                pos, stop = 1, mid[t]
        elif long_short and ct == down[t]:
            if pos != -1:
                pos, stop = -1, mid[t]
        elif pos == 1 and ct <= stop:
            pos, stop = 0, math.nan
        elif pos == -1 and ct >= stop:
            pos, stop = 0, math.nan
        pos_out[t] = pos
        stop_out[t] = stop
        # Daily update for tomorrow (on an entry day the stop already equals mid).
        if pos == 1:
            stop = max(stop, mid[t])
        elif pos == -1:
            stop = min(stop, mid[t])
    return pd.DataFrame({"pos": pos_out, "stop": stop_out}, index=close.index)


def signal(close: pd.Series, n: int, long_short: bool) -> pd.Series:
    """Daily position in {-1, 0, 1} for one Donchian lookback n."""
    return signal_with_stop(close, n, long_short)["pos"].rename(close.name)


def vol_scale(
    close: pd.Series,
    vol_target: float = VOL_TARGET,
    vol_window: int = VOL_WINDOW,
    per_coin_cap: float = PER_COIN_CAP,
) -> pd.Series:
    """min(vol_target / sigma_t, per_coin_cap); 0 until sigma has a full window.

    sigma_t = std(last vol_window daily returns, ddof=1) * sqrt(365).
    vol_window returns need vol_window + 1 closes. A sigma of 0 gives the cap.
    """
    rets = close.pct_change(fill_method=None)
    sigma = rets.rolling(vol_window, min_periods=vol_window).std() * math.sqrt(DAYS_PER_YEAR)
    with np.errstate(divide="ignore"):
        raw = vol_target / sigma
    return raw.clip(upper=per_coin_cap).fillna(0.0).rename(close.name)


def _coin_frames(close, long_short, lookbacks, vol_target, vol_window, per_coin_cap):
    """(coin weight, any-lookback-position-changed flags) for one coin."""
    scale = vol_scale(close, vol_target, vol_window, per_coin_cap)
    total = pd.Series(0.0, index=close.index)
    changed = pd.Series(False, index=close.index)
    for n in lookbacks:
        pos = signal(close, n, long_short)
        total = total + pos
        changed |= pos != pos.shift(fill_value=0)
    weight = (scale * total / len(lookbacks)).fillna(0.0)
    return weight.rename(close.name), changed.rename(close.name)


def coin_weight(
    close: pd.Series,
    long_short: bool,
    lookbacks=LOOKBACKS,
    vol_target: float = VOL_TARGET,
    vol_window: int = VOL_WINDOW,
    per_coin_cap: float = PER_COIN_CAP,
) -> pd.Series:
    """Combo weight: mean over lookbacks of vol_scale * signal_n (paper 4.2-4.3)."""
    return _coin_frames(close, long_short, lookbacks, vol_target, vol_window, per_coin_cap)[0]


def _universe_mask(index: pd.DatetimeIndex, columns: list, universe_by_month) -> pd.DataFrame:
    """Membership per day: days in month M use the list keyed by month M-1's last day."""
    keyed = {pd.Timestamp(k).normalize(): set(v) for k, v in universe_by_month.items()}
    prev_end = (index.to_period("M") - 1).to_timestamp(how="end").normalize()
    mask = pd.DataFrame(False, index=index, columns=columns)
    for key in prev_end.unique():
        members = keyed.get(pd.Timestamp(key), set())
        cols = [c for c in columns if c in members]
        if cols:
            mask.loc[prev_end == key, cols] = True
    return mask


def _excluded_mask(index: pd.DatetimeIndex, columns: list, excluded) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=index, columns=columns)
    for day, coins in (excluded or {}).items():
        ts = pd.Timestamp(day).normalize()
        cols = [c for c in columns if c in coins]
        if ts in mask.index and cols:
            mask.loc[ts, cols] = True
    return mask


def _flipped(mask: pd.DataFrame) -> pd.DataFrame:
    return mask != mask.shift(fill_value=False)


def _allocation_frame(close: pd.DataFrame, universe_by_month, excluded):
    """(active mask, universe size per day, allocation signal_changed flags).

    Depends only on close, universe and exclusions, so callers that allocate
    many weight sets over the same coins (luck odds) compute it once.
    """
    idx, cols = close.index, list(close.columns)
    member = _universe_mask(idx, cols, universe_by_month)
    excl = _excluded_mask(idx, cols, excluded)
    delisted = (close.notna().cummax() & close.isna()).cummax()
    size = member.sum(axis=1).replace(0, np.nan)
    active = member & ~excl & ~delisted
    flags = (_flipped(member) | _flipped(excl) | _flipped(delisted)).astype(bool)
    return active, size, flags


def _apply_allocation(coin_weights: pd.DataFrame, active, size, gross_cap: float | None) -> pd.DataFrame:
    weights = coin_weights.where(active, 0.0).div(size, axis=0).fillna(0.0)
    if gross_cap is not None:
        gross = weights.abs().sum(axis=1)
        factor = (gross_cap / gross).where(gross > gross_cap, 1.0)
        weights = weights.mul(factor, axis=0)
    return weights


def allocate(
    coin_weights: pd.DataFrame,
    close: pd.DataFrame,
    universe_by_month: dict[date, list[str]],
    gross_cap: float | None,
    excluded: dict[date, set[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Portfolio allocation of per-coin weights (by decision date).

    coin_weights must share close's index and columns.
    universe_by_month: keyed by the month-end date of selection, i.e. the
        output of top_universe(close, qv, month_end). That list applies to
        every day of the NEXT month. A month whose previous month end has no
        key is flat.
    excluded: day -> coins flagged by exit_flags(close, qv, day). An excluded
        coin gets weight 0 in that day's decision, so it is out of tomorrow's
        return (exit_flags' "drop from tomorrow").

    weight[t, coin] = coin_weights[t] / len(month universe) for members, else 0.
    A coin counts as delisted from its first NaN close after its first valid
    close and gets weight 0 from that day on. If gross_cap is set and
    sum(|w|) on a day exceeds it, that day's weights scale down proportionally.

    signal_changed[t, coin] is True when the coin entered or left the
    universe, its exclusion started or ended, or it delisted that day. Before
    the first row everything counts as outside the universe, not excluded and
    not delisted.
    """
    coin_weights = coin_weights.reindex(index=close.index, columns=close.columns).fillna(0.0)
    active, size, flags = _allocation_frame(close, universe_by_month, excluded)
    return _apply_allocation(coin_weights, active, size, gross_cap), flags


def portfolio_weights(
    close: pd.DataFrame,
    universe_by_month: dict[date, list[str]],
    long_short: bool,
    per_coin_cap: float,
    gross_cap: float | None,
    excluded: dict[date, set[str]] | None = None,
    lookbacks=LOOKBACKS,
    vol_target: float = VOL_TARGET,
    vol_window: int = VOL_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily portfolio weights (by decision date) and signal_changed flags.

    Allocation (universe, exclusions, delisting, gross cap) is ``allocate``.
    signal_changed is allocate's flags OR any lookback's position changing
    at t (before the first row every lookback counts as flat).
    """
    idx, cols = close.index, list(close.columns)
    coin_w = pd.DataFrame(0.0, index=idx, columns=cols)
    pos_chg = pd.DataFrame(False, index=idx, columns=cols)
    for c in cols:
        coin_w[c], pos_chg[c] = _coin_frames(close[c], long_short, lookbacks, vol_target, vol_window, per_coin_cap)

    weights, alloc_chg = allocate(coin_w, close, universe_by_month, gross_cap, excluded)
    signal_changed = pos_chg | alloc_chg
    return weights, signal_changed.astype(bool)
