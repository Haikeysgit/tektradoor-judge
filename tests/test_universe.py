from datetime import date

import numpy as np
import pandas as pd

from judge.data import exit_flags, top_universe

DAYS = pd.date_range("2024-01-01", "2025-12-31", freq="D", name="date")
END = date(2025, 12, 31)


def frames(vols, first_day=None, prices=None):
    """vols: symbol -> constant daily quote volume. first_day: symbol -> first listed day."""
    first_day = first_day or {}
    prices = prices or {}
    close = pd.DataFrame(index=DAYS, columns=list(vols), dtype=float)
    qv = close.copy()
    for i, (s, v) in enumerate(vols.items()):
        close[s] = prices.get(s, 100.0 * (1 + 0.02 * (-1) ** np.arange(len(DAYS))))
        qv[s] = v
        if s in first_day:
            close.loc[: pd.Timestamp(first_day[s]) - pd.Timedelta(days=1), s] = np.nan
            qv.loc[: pd.Timestamp(first_day[s]) - pd.Timedelta(days=1), s] = np.nan
    return close, qv


def test_excludes_coin_younger_than_365_days():
    close, qv = frames({"AAAUSDT": 9e6, "NEWUSDT": 50e6}, first_day={"NEWUSDT": "2025-06-01"})
    assert top_universe(close, qv, END) == ["AAAUSDT"]


def test_excludes_stablecoin_even_if_top_volume():
    vols = {"USDCUSDT": 900e6, "WBTCUSDT": 800e6, "FDUSDUSDT": 700e6, "AAAUSDT": 9e6, "BBBUSDT": 8e6}
    close, qv = frames(vols)
    assert top_universe(close, qv, END) == ["AAAUSDT", "BBBUSDT"]


def test_multiplier_prefix_is_not_excluded():
    close, qv = frames({"1000PEPEUSDT": 9e6, "USDEUSDT": 99e6})
    assert top_universe(close, qv, END) == ["1000PEPEUSDT"]


def test_ranks_by_30d_median_volume_returns_top_n():
    close, qv = frames({"AUSDT": 3e6, "BUSDT": 9e6, "CUSDT": 5e6, "DUSDT": 7e6, "EUSDT": 4e6})
    assert top_universe(close, qv, END, size=3) == ["BUSDT", "DUSDT", "CUSDT"]


def test_min_median_volume_and_delisted_excluded():
    close, qv = frames({"LOWUSDT": 1e6, "OKUSDT": 3e6, "GONEUSDT": 90e6})
    close.loc["2025-12-20":, "GONEUSDT"] = np.nan
    qv.loc["2025-12-20":, "GONEUSDT"] = np.nan
    assert top_universe(close, qv, END) == ["OKUSDT"]


def test_exit_flag_low_volume():
    close, qv = frames({"LOWUSDT": 0.5e6, "OKUSDT": 3e6})
    assert exit_flags(close, qv, END) == ["LOWUSDT"]


def test_exit_flag_flat_price():
    flat = pd.Series(100.0, index=DAYS)
    close, qv = frames({"FLATUSDT": 5e6, "OKUSDT": 5e6}, prices={"FLATUSDT": flat})
    assert exit_flags(close, qv, END) == ["FLATUSDT"]
