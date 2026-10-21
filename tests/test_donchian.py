import numpy as np
import pandas as pd
import pytest

from judge.strategies.donchian import (
    coin_weight,
    portfolio_weights,
    signal,
    signal_with_stop,
    vol_scale,
)

D12 = pd.date_range("2024-01-01", periods=12, freq="D", name="date")


def s(values, start="2024-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D", name="date"), dtype=float)


def noisy_trend(n_days, drift, seed, start="2023-01-01", vol=0.03):
    rng = np.random.default_rng(seed)
    rets = drift + vol * rng.standard_normal(n_days)
    return s(100 * np.cumprod(1 + rets), start=start)


# ---------------------------------------------------------------- signal

def test_long_entry_on_n_day_high():
    # n=5: first full window at index 4. Index 4 (10) is not the 5-day high (12 at index 0).
    # Index 5 (10.5) is below 11. Index 6 closes at 13, above every close in indices 2..6 -> entry on index 6.
    close = pd.Series([12, 11, 10, 9, 10, 10.5, 13, 12.5, 12.8, 12.6, 12.7, 12.9], index=D12, dtype=float)
    pos = signal(close, 5, long_short=False)
    assert pos.iloc[:6].tolist() == [0] * 6
    assert pos.iloc[6] == 1
    assert (pos.iloc[:4] == 0).all()  # no signal before n closes exist


def test_no_signal_before_full_window():
    # Rising from day 0: every close is the running max, but no window exists before index n-1.
    close = s(np.arange(1, 13))
    pos = signal(close, 5, long_short=False)
    assert pos.iloc[:4].tolist() == [0, 0, 0, 0]
    assert pos.iloc[4] == 1


def test_trailing_stop_never_moves_down():
    close = noisy_trend(400, 0.002, seed=1)
    df = signal_with_stop(close, 20, long_short=False)
    held = df["pos"] == 1
    # Within each continuous long, the stop is non-decreasing.
    run_id = (held != held.shift()).cumsum()
    for _, run in df[held].groupby(run_id[held]):
        assert run["stop"].is_monotonic_increasing
    assert held.sum() > 20


def test_initial_stop_is_mid_at_entry():
    close = pd.Series([12, 11, 10, 9, 10, 10.5, 13, 12.5, 12.8, 12.6, 12.7, 12.9], index=D12, dtype=float)
    df = signal_with_stop(close, 5, long_short=False)
    # Entry index 6: window closes 10, 9, 10, 11, 13 -> mid = (13 + 9) / 2 = 11.
    assert df["stop"].iloc[6] == pytest.approx(11.0)


def test_exit_when_close_below_stop():
    # Entry at index 6 with stop 11; then a drop below the stop must be flat on that exact day.
    close = pd.Series([12, 11, 10, 9, 10, 10.5, 13, 12.5, 12.8, 10.9, 12.7, 12.9], index=D12, dtype=float)
    df = signal_with_stop(close, 5, long_short=False)
    assert df["pos"].iloc[6] == 1
    # Stop in force for day 9 = max(11, mid_7, mid_8). mid_7 = (13+9)/2 = 11; mid_8 = (13+10)/2 = 11.5.
    assert df["pos"].iloc[8] == 1
    assert df["pos"].iloc[9] == 0  # 10.9 <= 11.5
    # Exactly at the stop also exits.
    close2 = close.copy()
    close2.iloc[9] = 11.5
    assert signal(close2, 5, long_short=False).iloc[9] == 0
    close3 = close.copy()
    close3.iloc[9] = 11.6
    assert signal(close3, 5, long_short=False).iloc[9] == 1


def test_short_entry_only_when_long_short_true():
    # Index 6 is a new 5-day low.
    close = pd.Series([8, 9, 10, 11, 10, 9.5, 7, 7.5, 7.2, 7.4, 7.3, 7.1], index=D12, dtype=float)
    assert (signal(close, 5, long_short=False) == 0).all()
    pos = signal(close, 5, long_short=True)
    assert pos.iloc[:6].tolist() == [0] * 6
    assert pos.iloc[6] == -1
    # Short stop: entry mid = (11 + 7) / 2 = 9; exits when close >= stop.
    up = close.copy()
    up.iloc[8] = 9.0
    assert signal(up, 5, long_short=True).iloc[8] == 0


def test_short_stop_never_moves_up():
    close = noisy_trend(400, -0.002, seed=2)
    df = signal_with_stop(close, 20, long_short=True)
    short = df["pos"] == -1
    run_id = (short != short.shift()).cumsum()
    for _, run in df[short].groupby(run_id[short]):
        assert run["stop"].is_monotonic_decreasing
    assert short.sum() > 20


def test_nan_close_closes_position():
    close = s(np.arange(1, 21, dtype=float))
    close.iloc[15:] = np.nan
    pos = signal(close, 5, long_short=False)
    assert pos.iloc[14] == 1
    assert (pos.iloc[15:] == 0).all()


# ---------------------------------------------------------------- sizing

def test_vol_sizing_capped_at_2x():
    # Very calm coin: tiny sigma -> 0.25 / sigma is huge -> capped at 2.0.
    calm = s(100 * np.cumprod(1 + 0.0001 * (-1) ** np.arange(200)))
    vs = vol_scale(calm)
    assert vs.iloc[:90].eq(0).all()  # 90 returns need 91 closes
    assert vs.iloc[90:].eq(2.0).all()
    # Wild coin: sigma well above 25% -> uncapped value 0.25 / sigma.
    wild = noisy_trend(200, 0.0, seed=3, vol=0.05)
    vw = vol_scale(wild)
    sigma = wild.pct_change(fill_method=None).rolling(90).std() * np.sqrt(365)
    assert vw.iloc[150] == pytest.approx(0.25 / sigma.iloc[150])
    assert vw.iloc[150] < 2.0
    assert vol_scale(calm, per_coin_cap=1.0).iloc[150] == 1.0


def test_combo_is_mean_of_lookbacks():
    close = noisy_trend(300, 0.003, seed=4)
    w = coin_weight(close, long_short=False, lookbacks=(5, 20))
    vs = vol_scale(close)
    expected = vs * (signal(close, 5, False) + signal(close, 20, False)) / 2
    pd.testing.assert_series_equal(w, expected, check_names=False)
    assert (w.iloc[150:] != 0).any()


def test_weight_zero_when_sigma_window_incomplete():
    close = s(np.arange(1, 60, dtype=float))  # always long, but < 91 closes
    assert (coin_weight(close, long_short=False, lookbacks=(5,)) == 0).all()


# ---------------------------------------------------------------- portfolio

DAYS = pd.date_range("2023-01-01", "2024-03-31", freq="D", name="date")


def rising(seed, drift=0.004):
    return noisy_trend(len(DAYS), drift, seed=seed, start="2023-01-01", vol=0.01).values


def two_coin_close():
    return pd.DataFrame({"AUSDT": rising(10), "BUSDT": rising(11)}, index=DAYS)


ALL_MONTHS = {pd.Timestamp(m): ["AUSDT", "BUSDT"] for m in pd.date_range("2022-12-31", "2024-02-29", freq="ME")}


def test_portfolio_is_equal_split_of_coin_weights():
    close = two_coin_close()
    w, _ = portfolio_weights(close, ALL_MONTHS, long_short=False, per_coin_cap=2.0, gross_cap=None)
    for c in close:
        cw = coin_weight(close[c], long_short=False, per_coin_cap=2.0)
        pd.testing.assert_series_equal(w[c], cw / 2, check_names=False)


def test_universe_uses_previous_month_end_key():
    close = two_coin_close()
    uni = dict(ALL_MONTHS)
    uni[pd.Timestamp("2023-12-31")] = ["AUSDT"]  # selection at Dec end applies to January
    w, chg = portfolio_weights(close, uni, long_short=False, per_coin_cap=2.0, gross_cap=None)
    jan = (w.index >= "2024-01-01") & (w.index <= "2024-01-31")
    assert (w.loc[jan, "BUSDT"] == 0).all()
    assert (w.loc["2023-12", "BUSDT"] != 0).any()
    assert (w.loc["2024-02", "BUSDT"] != 0).any()
    # A's share doubles in January (universe of 1).
    cw = coin_weight(close["AUSDT"], long_short=False)
    pd.testing.assert_series_equal(w.loc[jan, "AUSDT"], cw[jan], check_names=False)
    assert chg.loc["2024-01-01", "BUSDT"]
    assert chg.loc["2024-02-01", "BUSDT"]


def test_missing_universe_key_means_flat():
    close = two_coin_close()
    uni = {k: v for k, v in ALL_MONTHS.items() if k != pd.Timestamp("2023-10-31")}
    w, _ = portfolio_weights(close, uni, long_short=False, per_coin_cap=2.0, gross_cap=None)
    assert (w.loc["2023-11"] == 0).all().all()


def test_gross_cap_scales_to_1():
    days = DAYS
    close = pd.DataFrame({"AUSDT": rising(10), "BUSDT": rising(11)}, index=days)
    w_raw, _ = portfolio_weights(close, ALL_MONTHS, long_short=False, per_coin_cap=2.0, gross_cap=None)
    gross = w_raw.abs().sum(axis=1)
    day = gross[gross > 1.0].index[-1]
    w_cap, _ = portfolio_weights(close, ALL_MONTHS, long_short=False, per_coin_cap=2.0, gross_cap=1.0)
    assert w_cap.loc[day].abs().sum() == pytest.approx(1.0)
    ratio = w_raw.loc[day] / w_cap.loc[day]
    assert ratio.iloc[0] == pytest.approx(ratio.iloc[1])
    below = gross <= 1.0
    pd.testing.assert_frame_equal(w_cap[below], w_raw[below])


def test_gross_cap_exact_1_6_example(monkeypatch):
    import judge.strategies.donchian as d

    idx = pd.date_range("2024-01-01", periods=3, freq="D", name="date")
    close = pd.DataFrame({"AUSDT": [1.0, 1.0, 1.0], "BUSDT": [1.0, 1.0, 1.0]}, index=idx)
    fake = {"AUSDT": pd.Series([2.0, 2.0, 2.0], index=idx), "BUSDT": pd.Series([1.2, 1.2, 1.2], index=idx)}
    monkeypatch.setattr(d, "_coin_frames", lambda c, ls, lb, vt, vw, cap: (fake[c.name], pd.Series(False, index=idx)))
    uni = {pd.Timestamp("2023-12-31"): ["AUSDT", "BUSDT"]}
    w, _ = d.portfolio_weights(close, uni, long_short=False, per_coin_cap=2.0, gross_cap=1.0)
    # raw: 1.0 and 0.6 (gross 1.6) -> 0.625 and 0.375
    assert w["AUSDT"].tolist() == pytest.approx([0.625] * 3)
    assert w["BUSDT"].tolist() == pytest.approx([0.375] * 3)


def test_delisted_coin_weight_zero_after_last_close():
    close = two_coin_close()
    last = pd.Timestamp("2023-11-15")
    close.loc[last + pd.Timedelta(days=1):, "BUSDT"] = np.nan
    w, chg = portfolio_weights(close, ALL_MONTHS, long_short=False, per_coin_cap=2.0, gross_cap=None)
    assert w.loc[last, "BUSDT"] != 0  # still held at its last close
    assert (w.loc[last + pd.Timedelta(days=1):, "BUSDT"] == 0).all()
    assert chg.loc[last + pd.Timedelta(days=1), "BUSDT"]
    assert not w.isna().any().any()


def test_excluded_coin_weight_zero_that_day():
    close = two_coin_close()
    ex_days = pd.date_range("2023-10-10", "2023-10-12", freq="D")
    excluded = {d.date(): {"AUSDT"} for d in ex_days}
    base, _ = portfolio_weights(close, ALL_MONTHS, long_short=False, per_coin_cap=2.0, gross_cap=None)
    w, chg = portfolio_weights(close, ALL_MONTHS, long_short=False, per_coin_cap=2.0, gross_cap=None, excluded=excluded)
    assert (base.loc[ex_days, "AUSDT"] != 0).all()
    assert (w.loc[ex_days, "AUSDT"] == 0).all()
    pd.testing.assert_series_equal(w["BUSDT"], base["BUSDT"])
    other = ~w.index.isin(ex_days)
    pd.testing.assert_series_equal(w.loc[other, "AUSDT"], base.loc[other, "AUSDT"])
    assert chg.loc["2023-10-10", "AUSDT"]  # exclusion starts
    assert not chg.loc["2023-10-11", "AUSDT"]
    assert chg.loc["2023-10-13", "AUSDT"]  # exclusion ends


def test_signal_changed_marks_lookback_changes():
    close = two_coin_close()
    close["AUSDT"] = noisy_trend(len(DAYS), 0.0, seed=12, start="2023-01-01", vol=0.03).values
    _, chg = portfolio_weights(close, ALL_MONTHS, long_short=True, per_coin_cap=2.0, gross_cap=None,
                               lookbacks=(5, 20))
    p5 = signal(close["AUSDT"], 5, True)
    p20 = signal(close["AUSDT"], 20, True)
    flips = (p5 != p5.shift(fill_value=0)) | (p20 != p20.shift(fill_value=0))
    # No universe/exclusion/delisting events for AUSDT except entering on day 0.
    expected = flips.copy()
    expected.iloc[0] = True
    pd.testing.assert_series_equal(chg["AUSDT"], expected, check_names=False)
    assert chg["AUSDT"].sum() > 10


def test_no_lookahead():
    close = two_coin_close()
    close["AUSDT"] = noisy_trend(len(DAYS), 0.001, seed=13, start="2023-01-01", vol=0.04).values
    for long_short in (False, True):
        w0, c0 = portfolio_weights(close, ALL_MONTHS, long_short=long_short, per_coin_cap=2.0, gross_cap=1.0)
        for t in (150, 200, 300, 400):
            bumped = close.copy()
            bumped.iloc[t + 1:] *= np.linspace(0.5, 2.0, len(bumped) - t - 1)[:, None]
            w1, c1 = portfolio_weights(bumped, ALL_MONTHS, long_short=long_short, per_coin_cap=2.0, gross_cap=1.0)
            pd.testing.assert_frame_equal(w0.iloc[: t + 1], w1.iloc[: t + 1])
            pd.testing.assert_frame_equal(c0.iloc[: t + 1], c1.iloc[: t + 1])
