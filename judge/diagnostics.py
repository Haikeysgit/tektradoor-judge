"""Trade diagnostics for a result file. Report only: nothing here changes
pass or fail. Not part of the reproduction code tie.

Timing follows judge/costs.py: the executed weight decided at day t's close
earns day t+1's return. Inside a window (start, end) the decisions that count
are start - 1 .. end - 1 (their returns are dated start .. end).

Per coin and decision day t, the coin's contribution to portfolio return is
    hold(t) = w(t) * (r(t+1) - funding(t+1))
    cost(t) = cost_per_side * |w(t) - w(t-1)|      (charged on day t+1)
and over all coins hold - cost sums to the day's net portfolio return.

A trade is one position episode of one coin in the executed weights: from
the first non-zero weight until the weight returns to zero (or, long-short,
changes sign: the flip closes one trade and opens the next, and the flip's
turnover is split between them). Its P&L is the sum of its contributions,
entry and exit costs included, in percent of the portfolio. A trade still
open at the window's last decision ends with "window_end" on the window's
end date.

Exit reason, read on the exit decision day:
    left_universe  the coin is out of the month's top-20 (or delisted);
    exit_flag      the daily liquidity/volatility exit check excluded it;
    trailing_stop  otherwise: the coin's combo signal went to zero (or
                   changed sign);
    window_end     still open at the end of the window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from judge.costs import _executed_weights

REASONS = ("trailing_stop", "left_universe", "exit_flag", "window_end")
GROSS_TOLERANCE = 1e-9
_ZERO = 1e-15
UNITS = {
    "pnl": "percent of portfolio: sum of the trade's daily contributions (weight x coin return, "
           "minus that coin's costs and funding); not compounded",
    "holding_days": "return days from the entry decision to the exit decision",
    "exposure": "sum of |executed weights| (gross) and |sum of executed weights| (net), multiples of equity",
    "worst_day": "percent net portfolio return on that return day",
}


class GrossExposureError(AssertionError):
    """A judge bug: executed gross exposure above the gross cap."""


def _iso(ts) -> str:
    return pd.Timestamp(ts).date().isoformat()


def _sign(x: float) -> int:
    return 0 if abs(x) <= _ZERO else (1 if x > 0 else -1)


def worst_day(returns: pd.Series) -> dict | None:
    if returns is None or len(returns) == 0:
        return None
    return {"date": _iso(returns.idxmin()), "return_pct": float(returns.min() * 100)}


def _mask(frame: pd.DataFrame | None, idx, cols, default: bool) -> np.ndarray:
    if frame is None:
        return np.full((len(idx), len(cols)), default)
    return frame.reindex(index=idx, columns=cols).fillna(default).astype(bool).to_numpy()


def strategy_diagnostics(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    funding: pd.DataFrame | None,
    cost_per_side: float,
    window,
    net: pd.Series,
    *,
    gross_cap: float,
    member: pd.DataFrame | None = None,
    excluded: pd.DataFrame | None = None,
    rebalance_threshold: float | None = None,
    changed: pd.DataFrame | None = None,
    bench: pd.Series | None = None,
) -> dict:
    """Diagnostics of one run.

    weights: target weights by decision day, already flat before the window
    (judge.start_flat); with rebalance_threshold they are turned into executed
    weights exactly as costs.net_returns does. net: the window's net daily
    returns (for the worst day and the reconciliation). member / excluded:
    universe membership and exit-check exclusions by decision day.
    Raises GrossExposureError when executed gross exposure exceeds gross_cap.
    """
    idx, cols = weights.index, list(weights.columns)
    if rebalance_threshold is not None:
        w = _executed_weights(weights, changed, rebalance_threshold)
    else:
        w = weights.fillna(0.0)
    rets = close.reindex(index=idx, columns=cols).pct_change(fill_method=None).fillna(0.0)
    fund = (funding.reindex(index=idx, columns=cols).fillna(0.0) if funding is not None
            else pd.DataFrame(0.0, index=idx, columns=cols))
    prev = w.shift(1).fillna(0.0)
    hold = (w * (rets.shift(-1) - fund.shift(-1))).fillna(0.0)
    turn = cost_per_side * (w - prev).abs()

    start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    in_win = (idx >= start - pd.Timedelta(days=1)) & (idx <= end - pd.Timedelta(days=1))
    days = idx[in_win]
    W, P = w.loc[in_win].to_numpy(), prev.loc[in_win].to_numpy()
    H, T = hold.loc[in_win].to_numpy(), turn.loc[in_win].to_numpy()

    gross = np.abs(W).sum(axis=1) if len(days) else np.zeros(1)
    net_exp = np.abs(W.sum(axis=1)) if len(days) else np.zeros(1)
    max_gross, max_net = float(gross.max()), float(net_exp.max())
    if max_gross > gross_cap + GROSS_TOLERANCE:
        at = _iso(days[int(gross.argmax())])
        raise GrossExposureError(
            f"judge bug: executed gross exposure {max_gross:.10g} on {at} exceeds gross_cap {gross_cap}"
        )

    delisted = (close.notna().cummax() & close.isna()).cummax()
    mem = _mask(member, days, cols, True)
    exc = _mask(excluded, days, cols, False)
    dead = _mask(delisted, days, cols, False)

    trades = []

    def finish(tr, exit_ts, reason):
        tr.update(exit=exit_ts, reason=reason)
        trades.append(tr)

    for j, coin in enumerate(cols):
        tr = None
        for k, t in enumerate(days):
            s = _sign(W[k, j])
            if tr is not None and s == tr["sign"]:
                tr["pnl"] += H[k, j] - T[k, j]
            elif tr is not None and s == 0:
                tr["pnl"] -= T[k, j]
                reason = "left_universe" if (not mem[k, j] or dead[k, j]) else (
                    "exit_flag" if exc[k, j] else "trailing_stop")
                finish(tr, t, reason)
                tr = None
            elif tr is not None:  # sign flip: split the turnover between the two trades
                tr["pnl"] -= cost_per_side * abs(P[k, j])
                finish(tr, t, "trailing_stop")
                tr = {"coin": coin, "sign": s, "entry": t, "pnl": H[k, j] - cost_per_side * abs(W[k, j])}
            elif s != 0:
                tr = {"coin": coin, "sign": s, "entry": t, "pnl": H[k, j] - T[k, j]}
        if tr is not None:
            finish(tr, end, "window_end")

    rows = [{
        "coin": tr["coin"],
        "side": "long" if tr["sign"] > 0 else "short",
        "entry": _iso(tr["entry"]),
        "exit": _iso(tr["exit"]),
        "holding_days": int((pd.Timestamp(tr["exit"]) - pd.Timestamp(tr["entry"])).days),
        "pnl_pct": float(tr["pnl"] * 100),
        "reason": tr["reason"],
    } for tr in trades]
    n = len(rows)
    pnls = [r["pnl_pct"] for r in rows]
    wins, losses = [p for p in pnls if p > 0], [p for p in pnls if p < 0]
    held = [r["holding_days"] for r in rows]
    reasons = {k: sum(r["reason"] == k for r in rows) for k in REASONS}
    worst = sorted(rows, key=lambda r: r["pnl_pct"])[:5]
    return {
        "trades": {
            "n_trades": n,
            "win_rate": 100.0 * len(wins) / n if n else None,
            "avg_win": float(np.mean(wins)) if wins else None,
            "avg_loss": float(np.mean(losses)) if losses else None,
            "avg_holding_days": float(np.mean(held)) if held else None,
            "median_holding_days": float(np.median(held)) if held else None,
            "max_holding_days": int(max(held)) if held else None,
        },
        "exits": {
            "reasons": reasons,
            "stop_outs_pct": 100.0 * reasons["trailing_stop"] / n if n else None,
            "stop_out_cost": float(sum(r["pnl_pct"] for r in rows if r["reason"] == "trailing_stop")),
        },
        "worst_trade": worst[0] if worst else None,
        "worst_trades": worst,
        "worst_day": worst_day(net),
        "leverage": {"max_gross_exposure": max_gross, "max_net_exposure": max_net, "gross_cap": gross_cap},
        "reconcile": {"sum_trade_pnl_pct": float(sum(pnls)), "sum_daily_return_pct": float(net.sum() * 100)},
        "units": UNITS,
        "benchmark": {"worst_day": worst_day(bench)} if bench is not None else None,
    }


def summary(diag: dict) -> list[str]:
    """Short human summary for the CLI."""
    t, x, lev = diag["trades"], diag["exits"], diag["leverage"]

    def pct(v, signed=True):
        return "n/a" if v is None else (f"{v:+.2f}%" if signed else f"{v:.1f}%")

    def num(v, fmt="{:.1f}"):
        return "n/a" if v is None else fmt.format(v)

    lines = [
        f"trades {t['n_trades']}: win rate {pct(t['win_rate'], False)}, avg win {pct(t['avg_win'])}, "
        f"avg loss {pct(t['avg_loss'])}; holding avg {num(t['avg_holding_days'])} d, "
        f"median {num(t['median_holding_days'])} d, max {num(t['max_holding_days'], '{}')} d",
        "exits: " + ", ".join(f"{k} {v}" for k, v in x["reasons"].items())
        + f"; stop-outs {pct(x['stop_outs_pct'], False)}, stop-out cost {pct(x['stop_out_cost'])}",
    ]
    wt, wd = diag["worst_trade"], diag["worst_day"]
    worst_trade = (f"{wt['coin']} {wt['side']} {wt['entry']}..{wt['exit']} {pct(wt['pnl_pct'])} ({wt['reason']})"
                   if wt else "none")
    worst_day_s = f"{wd['date']} {pct(wd['return_pct'])}" if wd else "n/a"
    lines.append(f"worst trade: {worst_trade}; worst day {worst_day_s}")
    lines.append(f"leverage: max gross {lev['max_gross_exposure']:.3f}, max net {lev['max_net_exposure']:.3f} "
                 f"(cap {lev['gross_cap']})")
    return lines
