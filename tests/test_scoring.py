import math

import numpy as np
import pandas as pd
import pytest

from judge.judge import benchmark_returns, luck_odds, max_drawdown, sharpe, verdict
from judge.strategies.donchian import allocate, coin_weight, portfolio_weights

C = 0.0011


def _series(values, start="2025-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"), dtype=float)


def _alternating(n, up=0.01, down=-0.005):
    return _series([up if i % 2 == 0 else down for i in range(n)])


# ---------------------------------------------------------------- sharpe / drawdown

def test_sharpe_known_series():
    daily = _alternating(4)
    # mean = 0.0025; deviations are +-0.0075; std(ddof=1) = 0.0075 * sqrt(4/3)
    expected = 0.0025 / (0.0075 * math.sqrt(4 / 3)) * math.sqrt(365)
    assert sharpe(daily) == pytest.approx(expected, abs=1e-9)


def test_sharpe_zero_std_is_zero():
    # Review Focus 4: long-only fully in cash all window.
    flat = _series([0.0] * 30)
    assert sharpe(flat) == 0.0
    assert sharpe(_series([0.003] * 30)) == 0.0
    assert sharpe(_series([0.01])) == 0.0
    assert sharpe(_series([])) == 0.0
    v = verdict(flat, flat, "test")
    assert v["passed"] is False
    assert v["sharpe"] == 0.0 and not math.isnan(v["sharpe"])


def test_max_drawdown_known_path():
    # equity 100 -> 120 -> 90 -> 130
    daily = _series([0.2, -0.25, 130 / 90 - 1])
    assert max_drawdown(daily) == pytest.approx(0.25)


def test_max_drawdown_counts_from_starting_equity():
    assert max_drawdown(_series([-0.1, 0.05])) == pytest.approx(0.1)
    assert max_drawdown(_series([0.01, 0.02])) == 0.0


# ---------------------------------------------------------------- verdict

def _good():
    return _alternating(300)  # Sharpe ~5.5, max dd 0.5%


def _crash():
    s = _alternating(300).copy()
    s.iloc[150] = -0.30
    return s


def test_test_rules_need_all_three():
    zero = _series([0.0] * 300)
    good = _good()
    v = verdict(good, zero, "test")
    assert v["passed"] is True
    assert all(v["rules"].values())

    # Sharpe <= 1.0 alone fails (dd tiny, beats a losing bench).
    weak = _alternating(300, up=0.001, down=-0.00095)
    losing = _series([-0.001 if i % 2 else 0.0005 for i in range(300)])
    v = verdict(weak, losing, "test")
    assert v["sharpe"] <= 1.0 and v["max_dd"] < 0.25 and v["sharpe"] > v["bench_sharpe"]
    assert v["passed"] is False

    # Max dd >= 25% alone fails.
    crash = _crash()
    v = verdict(crash, zero, "test")
    assert v["sharpe"] > 1.0 and v["max_dd"] >= 0.25 and v["sharpe"] > v["bench_sharpe"]
    assert v["passed"] is False

    # Not beating the benchmark alone fails (equal Sharpe is not ">").
    v = verdict(good, good, "test")
    assert v["sharpe"] > 1.0 and v["max_dd"] < 0.25 and v["sharpe"] == v["bench_sharpe"]
    assert v["passed"] is False

    assert set(v) == {"passed", "sharpe", "max_dd", "bench_sharpe", "rules"}


def test_final_holds_up_rule():
    weak = _alternating(300, up=0.001, down=-0.00095)  # Sharpe < 1
    zero = _series([0.0] * 300)
    v = verdict(weak, zero, "final")
    assert v["sharpe"] < 1.0
    assert v["passed"] is True  # Final does not need Sharpe > 1
    # Equal to the benchmark holds up (>=).
    assert verdict(weak, weak, "final")["passed"] is True
    # Below the benchmark fails.
    assert verdict(zero, weak, "final")["passed"] is False
    # Drawdown still applies.
    assert verdict(_crash(), zero, "final")["passed"] is False


def test_final_alone_uses_full_rules():
    weak = _alternating(300, up=0.001, down=-0.00095)
    zero = _series([0.0] * 300)
    assert verdict(weak, zero, "final")["passed"] is True
    assert verdict(weak, zero, "final_alone")["passed"] is False
    good = _good()
    assert verdict(good, good, "final")["passed"] is True
    assert verdict(good, good, "final_alone")["passed"] is False
    assert verdict(good, zero, "final_alone")["passed"] is True
    with pytest.raises(ValueError):
        verdict(good, zero, "nope")


# ---------------------------------------------------------------- allocate / benchmark

DAYS = pd.date_range("2024-01-01", "2024-12-31", freq="D")
ALL = {m: None for m in pd.date_range("2023-12-31", "2024-12-31", freq="ME")}


def _universe(coins, overrides=None):
    uni = {m: list(coins) for m in ALL}
    uni.update(overrides or {})
    return uni


def _close(n_coins, seed=0, days=DAYS, drift=0.002, vol=0.03):
    rng = np.random.default_rng(seed)
    rets = drift + vol * rng.standard_normal((len(days), n_coins))
    cols = [f"C{i:02d}USDT" for i in range(n_coins)]
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=days, columns=cols)


def test_allocate_reproduces_portfolio_weights_allocation():
    close = _close(3, seed=7)
    cols = list(close.columns)
    close.loc["2024-09-10":, cols[2]] = np.nan  # delisting
    uni = _universe(cols, {pd.Timestamp("2024-04-30"): cols[:2]})
    excluded = {pd.Timestamp("2024-07-04"): {cols[0]}}
    w_pf, chg_pf = portfolio_weights(close, uni, long_short=True, per_coin_cap=2.0, gross_cap=1.0, excluded=excluded)
    cw = pd.DataFrame({c: coin_weight(close[c], long_short=True, per_coin_cap=2.0) for c in cols})
    w_al, chg_al = allocate(cw, close, uni, gross_cap=1.0, excluded=excluded)
    pd.testing.assert_frame_equal(w_al, w_pf)
    assert (chg_pf | ~chg_al).all().all()  # every allocate flag is in portfolio's flags
    assert chg_al.loc["2024-05-01", cols[2]]  # universe exit
    assert chg_al.loc["2024-07-04", cols[0]] and chg_al.loc["2024-07-05", cols[0]]  # exclusion start/end
    assert chg_al.loc["2024-09-10", cols[2]]  # delisting
    assert not chg_al.loc["2024-03-15"].any()


def test_benchmark_flat_prices_only_costs_on_rebalance_days():
    close = pd.DataFrame(100.0, index=DAYS, columns=["AUSDT", "BUSDT"])
    uni = _universe(["AUSDT", "BUSDT"], {pd.Timestamp("2024-05-31"): ["AUSDT"]})
    b = benchmark_returns(close, None, uni, C)
    assert b.index.equals(DAYS[1:])
    # Entry on day 0 (0.5 + 0.5 = 1.0 turnover) is charged on day 1.
    assert b.loc["2024-01-02"] == pytest.approx(-C)
    # June: B out, A 0.5 -> 1.0. Turnover 0.5 + 0.5 = 1.0, charged the next day.
    assert b.loc["2024-06-02"] == pytest.approx(-C)
    # July: back to both. Turnover 1.0 again.
    assert b.loc["2024-07-02"] == pytest.approx(-C)
    others = b.drop(pd.to_datetime(["2024-01-02", "2024-06-02", "2024-07-02"]))
    assert (others == 0).all()


def test_benchmark_is_equal_weight_hold_with_funding():
    close = _close(2, seed=3)
    cols = list(close.columns)
    funding = pd.DataFrame(0.0001, index=DAYS, columns=cols)
    b = benchmark_returns(close, funding, _universe(cols), 0.0)
    rets = close.pct_change(fill_method=None)
    expected = 0.5 * rets[cols[0]] + 0.5 * rets[cols[1]] - 0.0001
    pd.testing.assert_series_equal(b, expected.iloc[1:], check_names=False)


# ---------------------------------------------------------------- luck odds

def _luck_inputs():
    close = _close(4, seed=11, drift=0.004, vol=0.02)
    return close, None, _universe(list(close.columns))


def test_luck_odds_deterministic_with_seed():
    close, funding, uni = _luck_inputs()
    kw = dict(long_short=True, flip_prob=0.05, n_sims=30, kind="test", cost_per_side=C, gross_cap=1.0)
    a = luck_odds(close, funding, uni, seed=42, **kw)
    b = luck_odds(close, funding, uni, seed=42, **kw)
    assert isinstance(a, float)
    assert a == b
    bench = benchmark_returns(close, funding, uni, C)
    assert luck_odds(close, funding, uni, seed=42, bench=bench, **kw) == a


def test_luck_odds_in_unit_interval():
    close, funding, uni = _luck_inputs()
    for ls in (False, True):
        for kind in ("test", "final"):
            p = luck_odds(close, funding, uni, long_short=ls, flip_prob=0.1, n_sims=25, seed=1,
                          kind=kind, cost_per_side=C, gross_cap=1.0)
            assert 0.0 <= p <= 1.0
    # flip_prob 0: never leaves state 0, so flat; Sharpe 0 never passes.
    assert luck_odds(close, funding, uni, long_short=False, flip_prob=0.0, n_sims=10, seed=1,
                     kind="test", cost_per_side=C, gross_cap=1.0) == 0.0
