import numpy as np
import pandas as pd
import pytest

from judge.costs import net_returns

C = 0.0011


def _idx(n):
    return pd.date_range("2024-01-01", periods=n, freq="D")


def _frame(vals, n=None, col="A"):
    return pd.DataFrame({col: vals}, index=_idx(len(vals)))


def test_no_trade_no_cost():
    close = _frame([100.0, 100.0, 100.0, 100.0])
    w = _frame([0.0, 0.0, 0.0, 0.0])
    out = net_returns(w, w.astype(bool), close, None, C, None)
    assert (out == 0).all()
    assert list(out.index) == list(close.index[1:])


def test_round_trip_cost_is_two_sides():
    close = _frame([100.0] * 5)
    w = _frame([1.0, 1.0, 0.0, 0.0, 0.0])
    sc = _frame([True, False, True, False, False])
    out = net_returns(w, sc, close, None, C, None)
    assert out.sum() == pytest.approx(-2 * C)
    assert out.iloc[0] == pytest.approx(-C)
    assert out.iloc[2] == pytest.approx(-C)


def test_long_pays_positive_funding():
    close = _frame([100.0] * 3)
    fund = _frame([0.001, 0.001, 0.001])
    w = _frame([1.0, 1.0, 1.0])
    sc = _frame([False] * 3)
    out = net_returns(w, sc, close, fund, 0.0, None)
    assert out.iloc[0] == pytest.approx(-0.001)


def test_short_receives_positive_funding():
    close = _frame([100.0] * 3)
    fund = _frame([0.001, 0.001, 0.001])
    w = _frame([-1.0, -1.0, -1.0])
    sc = _frame([False] * 3)
    out = net_returns(w, sc, close, fund, 0.0, None)
    assert out.iloc[0] == pytest.approx(0.001)


def test_nan_funding_treated_as_zero():
    close = _frame([100.0] * 3)
    fund = _frame([np.nan, np.nan, 0.001])
    w = _frame([1.0, 1.0, 1.0])
    out = net_returns(w, _frame([False] * 3), close, fund, 0.0, None)
    assert out.iloc[0] == 0.0
    assert out.iloc[1] == pytest.approx(-0.001)


def test_threshold_skips_small_vol_rebalance():
    close = _frame([100.0] * 4)
    w = _frame([1.0, 1.05, 1.05, 1.05])
    sc = _frame([True, False, False, False])
    out = net_returns(w, sc, close, None, C, 0.20)
    # only the entry is charged; the 5% vol change is skipped
    assert out.iloc[0] == pytest.approx(-C)
    assert out.iloc[1] == 0.0 and out.iloc[2] == 0.0


def test_vol_change_above_threshold_executes():
    close = _frame([100.0] * 3)
    w = _frame([1.0, 1.5, 1.5])
    sc = _frame([True, False, False])
    out = net_returns(w, sc, close, None, C, 0.20)
    assert out.iloc[1] == pytest.approx(-C * 0.5)


def test_signal_change_always_executes_despite_threshold():
    close = _frame([100.0] * 3)
    w = _frame([1.0, 1.05, 1.05])
    sc = _frame([True, True, False])
    out = net_returns(w, sc, close, None, C, 0.20)
    assert out.iloc[1] == pytest.approx(-C * 0.05)


def test_returns_earn_next_day_with_executed_weight():
    close = _frame([100.0, 110.0, 99.0])
    w = _frame([1.0, 1.0, 1.0])
    out = net_returns(w, _frame([False] * 3), close, None, 0.0, None)
    assert out.iloc[0] == pytest.approx(0.10)
    assert out.iloc[1] == pytest.approx(-0.10)


def test_delisted_coin_closes_at_last_price_and_charges_exit():
    # close goes NaN on day 3 (delisted); strategy zeroes weight on that day
    close = _frame([100.0, 110.0, 121.0, np.nan, np.nan, np.nan])
    w = _frame([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    sc = _frame([True, False, False, True, False, False])
    fund = _frame([0.0, 0.0, 0.0, np.nan, np.nan, np.nan])
    out = net_returns(w, sc, close, fund, C, None)
    assert out.notna().all()
    assert out.iloc[0] == pytest.approx(0.10 - C)
    assert out.iloc[1] == pytest.approx(0.10)
    # day 3: return NaN -> 0, exit charged on the day it would start earning
    assert out.iloc[2] == pytest.approx(0.0)
    assert out.iloc[3] == pytest.approx(-C)
    assert out.iloc[4] == 0.0
