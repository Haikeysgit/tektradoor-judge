"""Trade diagnostics (report only): trades, exit reasons, worst trade and day, leverage."""
import json
from datetime import date

import pandas as pd
import pytest

from judge import diagnostics as D
from judge import judge as J
from judge.__main__ import main as cli_main
from judge.costs import net_returns
from test_prereg import NAME, _commit_repro_pass, _fake_loader, _register, repo  # noqa: F401

DAYS = pd.date_range("2025-01-01", "2025-01-12", freq="D", name="date")
WIN = (date(2025, 1, 2), date(2025, 1, 12))  # decisions 01-01..01-11 earn returns 01-02..01-12
CPS = 0.0011
FUND = 0.0001


def _frame(values: dict, default=0.0) -> pd.DataFrame:
    """{coin: {day-of-month: value}} -> frame over DAYS (other days: default)."""
    out = pd.DataFrame(default, index=DAYS, columns=list(values), dtype=object if isinstance(default, bool) else float)
    for coin, by_day in values.items():
        for d, v in by_day.items():
            out.loc[pd.Timestamp(2025, 1, d), coin] = v
    return out.astype(bool) if isinstance(default, bool) else out


def _closes(path: dict) -> pd.DataFrame:
    """{coin: [12 closes]}"""
    return pd.DataFrame(path, index=DAYS, dtype=float)


def _diag(weights, close, member=None, excluded=None, gross_cap=1.0, funding=FUND, threshold=None, changed=None):
    coins = list(weights.columns)
    changed = changed if changed is not None else pd.DataFrame(True, index=DAYS, columns=coins)
    fund = pd.DataFrame(funding, index=DAYS, columns=coins)
    member = member if member is not None else pd.DataFrame(True, index=DAYS, columns=coins)
    excluded = excluded if excluded is not None else pd.DataFrame(False, index=DAYS, columns=coins)
    net = net_returns(weights, changed, close, fund, CPS, threshold)
    net = J.window_slice(net, WIN)
    diag = D.strategy_diagnostics(
        weights, close, fund, CPS, WIN, net, gross_cap=gross_cap, member=member, excluded=excluded,
        rebalance_threshold=threshold, changed=changed,
    )
    return diag, net


ONE = [100, 100, 100, 110, 121, 108.9, 108.9, 108.9, 108.9, 108.9, 108.9, 108.9]


def test_one_coin_entry_stop_out_and_known_pnl():
    # Long 0.5 decided at the 01-03 close, held for the returns of 01-04
    # (+10%), 01-05 (+10%) and 01-06 (-10%); the signal goes to zero at the
    # 01-06 close (still in the universe, not excluded): a trailing stop.
    w = _frame({"A": {3: 0.5, 4: 0.5, 5: 0.5}})
    diag, net = _diag(w, _closes({"A": ONE}))
    expected = 0.5 * 0.10 + 0.5 * 0.10 + 0.5 * -0.10 - CPS * 0.5 * 2 - FUND * 0.5 * 3
    t = diag["trades"]
    assert t["n_trades"] == 1 and t["win_rate"] == 100.0
    (worst,) = diag["worst_trades"]
    assert worst == {
        "coin": "A", "side": "long", "entry": "2025-01-03", "exit": "2025-01-06", "holding_days": 3,
        "pnl_pct": pytest.approx(expected * 100), "reason": "trailing_stop",
    }
    assert t["avg_win"] == pytest.approx(expected * 100) and t["avg_loss"] is None
    assert t["avg_holding_days"] == 3 and t["median_holding_days"] == 3 and t["max_holding_days"] == 3
    assert diag["exits"]["reasons"] == {"trailing_stop": 1, "left_universe": 0, "exit_flag": 0, "window_end": 0}
    assert diag["exits"]["stop_outs_pct"] == 100.0
    assert diag["exits"]["stop_out_cost"] == pytest.approx(expected * 100)
    # Every per-coin contribution belongs to a trade: trades sum to the window's daily returns.
    assert diag["reconcile"]["sum_trade_pnl_pct"] == pytest.approx(net.sum() * 100, abs=1e-12)
    assert diag["worst_day"] == {"date": "2025-01-06", "return_pct": pytest.approx(net.min() * 100)}
    assert diag["leverage"] == {
        "max_target_gross": 0.5, "max_executed_gross": 0.5, "days_executed_above_cap": 0,
        "max_net_exposure": 0.5, "gross_cap": 1.0, "rebalance_threshold": None,
    }


def test_left_universe_and_exit_flag_classified():
    w = _frame({"A": {3: 0.5, 4: 0.5}, "B": {3: 0.3, 4: 0.3}, "C": {3: 0.2}})
    member = _frame({"A": {5: False}, "B": {}, "C": {}}, default=True)  # A dropped from the universe on 01-05
    excluded = _frame({"A": {}, "B": {5: True}, "C": {}}, default=False)  # B flagged by the exit check on 01-05
    flat = [100.0] * 12
    diag, _ = _diag(w, _closes({"A": flat, "B": flat, "C": flat}), member=member, excluded=excluded)
    by_coin = {x["coin"]: x for x in diag["worst_trades"]}
    assert by_coin["A"]["reason"] == "left_universe" and by_coin["A"]["exit"] == "2025-01-05"
    assert by_coin["B"]["reason"] == "exit_flag"
    assert by_coin["C"]["reason"] == "trailing_stop" and by_coin["C"]["exit"] == "2025-01-04"
    assert diag["exits"]["reasons"] == {"trailing_stop": 1, "left_universe": 1, "exit_flag": 1, "window_end": 0}
    assert diag["exits"]["stop_outs_pct"] == pytest.approx(100 / 3)


def test_long_short_sign_flip_splits_into_two_trades():
    # Long 0.5 (decided 01-03, 01-04), then short 0.5 (01-05, 01-06), flat 01-07.
    w = _frame({"A": {3: 0.5, 4: 0.5, 5: -0.5, 6: -0.5}})
    close = _closes({"A": [100, 100, 100, 110, 99, 89.1, 89.1, 89.1, 89.1, 89.1, 89.1, 89.1]})
    diag, net = _diag(w, close, funding=0.0)
    trades = sorted(diag["worst_trades"], key=lambda x: x["entry"])
    assert [(x["side"], x["entry"], x["exit"], x["reason"]) for x in trades] == [
        ("long", "2025-01-03", "2025-01-05", "trailing_stop"),
        ("short", "2025-01-05", "2025-01-07", "trailing_stop"),
    ]
    # The flip's turnover (|-0.5 - 0.5| = 1.0) is split: 0.5 closes the long, 0.5 opens the short.
    long_pnl = 0.5 * 0.10 + 0.5 * -0.10 - CPS * 0.5 * 2
    short_pnl = -0.5 * -0.10 + -0.5 * 0.0 - CPS * 0.5 * 2  # earns 01-06 (-10%) and 01-07 (flat)
    assert trades[0]["pnl_pct"] == pytest.approx(long_pnl * 100)
    assert trades[1]["pnl_pct"] == pytest.approx(short_pnl * 100)
    assert diag["trades"]["n_trades"] == 2
    assert diag["reconcile"]["sum_trade_pnl_pct"] == pytest.approx(net.sum() * 100, abs=1e-12)
    assert diag["leverage"]["max_net_exposure"] == 0.5  # largest |sum of weights|


def test_open_trade_ends_at_window_end():
    w = _frame({"A": {9: 0.4, 10: 0.4, 11: 0.4, 12: 0.4}})
    diag, _ = _diag(w, _closes({"A": ONE}))
    (t,) = diag["worst_trades"]
    assert t["reason"] == "window_end" and t["exit"] == "2025-01-12" and t["holding_days"] == 3


def test_worst_five_trades_sorted():
    coins = {c: {3: 0.1} for c in "ABCDEFG"}
    paths = {c: [100, 100, 100, 100 - i, 100 - i] + [100 - i] * 7 for i, c in enumerate("ABCDEFG")}
    diag, _ = _diag(_frame(coins), _closes(paths), funding=0.0)
    assert [t["coin"] for t in diag["worst_trades"]] == ["G", "F", "E", "D", "C"]
    assert diag["worst_trade"] == diag["worst_trades"][0]


def test_target_gross_above_cap_is_a_judge_bug():
    flat = [100.0] * 12
    w = _frame({"A": {3: 0.6}, "B": {3: 0.6}})
    with pytest.raises(D.GrossExposureError, match="target gross exposure 1.2"):
        _diag(w, _closes({"A": flat, "B": flat}))
    ok = _frame({"A": {3: 0.5}, "B": {3: 0.5 + 1e-12}})
    assert _diag(ok, _closes({"A": flat, "B": flat}))[0]["leverage"]["max_target_gross"] == pytest.approx(1.0)


# Targets always sum to the cap (1.0), but the rebalance threshold (0.25 here,
# exact in binary) leaves A and then B unrebalanced while new coins execute:
#   decision 01-03: A 1.0                     -> executed A 1.0           (1.0)
#   decision 01-04: A 0.75, B 0.25            -> A stays 1.0 (rel 0.25)   (1.25)
#   decision 01-05: A 0.75, B 0.1875, C 0.0625 -> B stays 0.25 (rel 0.25) (1.3125)
DRIFT = {"A": {3: 1.0, 4: 0.75, 5: 0.75}, "B": {4: 0.25, 5: 0.1875}, "C": {5: 0.0625}}


def test_executed_drift_from_the_rebalance_rule_is_reported_not_refused():
    flat = [100.0] * 12
    w = _frame(DRIFT)
    changed = pd.DataFrame(False, index=DAYS, columns=list(w.columns))
    diag, _ = _diag(w, _closes({c: flat for c in "ABC"}), threshold=0.25, changed=changed)
    lev = diag["leverage"]
    assert lev["max_target_gross"] == 1.0
    assert lev["max_executed_gross"] == 1.3125  # above cap x (1 + threshold) = 1.25: legitimate drift
    assert lev["days_executed_above_cap"] == 2
    assert lev["max_executed_gross"] <= D.executed_gross_bound(1.0, 0.25)  # cap / (1 - threshold)
    assert "1.31x actual (drift from the 25% rebalance rule) on 2 days" in "\n".join(D.summary(diag))


def test_executed_gross_above_drift_bound_is_a_judge_bug(monkeypatch):
    flat = [100.0] * 12
    w = _frame({"A": {3: 0.5}, "B": {3: 0.5}})
    real = D._executed_weights
    monkeypatch.setattr(D, "_executed_weights", lambda *a: real(*a) * 1.4)  # an executor bug
    with pytest.raises(D.GrossExposureError, match="executed gross exposure 1.4"):
        _diag(w, _closes({"A": flat, "B": flat}), threshold=0.2,
              changed=pd.DataFrame(True, index=DAYS, columns=["A", "B"]))


# ---------------------------------------------------------------- run() and CLI

KEYS = {"trades", "exits", "worst_trade", "worst_trades", "worst_day", "leverage", "reconcile", "units", "benchmark"}


@pytest.mark.parametrize("force", [True, False])
def test_run_writes_diagnostics_on_pass_and_fail(repo, monkeypatch, force):
    p = _register(repo)
    _commit_repro_pass(repo, p)
    monkeypatch.setattr(J, "_load_data", _fake_loader(drift=0.003, calls=[]))
    real = J.verdict

    def forced(strategy, bench, kind):
        v = real(strategy, bench, kind)
        return {**v, "passed": force}

    monkeypatch.setattr(J, "verdict", forced)
    res = json.loads(J.run(p, "test").read_text())
    assert res["passed"] is force
    diag = res["diagnostics"]
    assert set(diag) == KEYS
    assert diag["leverage"]["max_target_gross"] <= 1.0 + 1e-9
    assert diag["leverage"]["max_executed_gross"] <= 1.25 + 1e-9 and diag["leverage"]["rebalance_threshold"] == 0.2
    assert diag["trades"]["n_trades"] > 0
    assert sum(diag["exits"]["reasons"].values()) == diag["trades"]["n_trades"]
    assert diag["benchmark"]["worst_day"]["date"] >= "2025-03-20"


def test_cli_prints_diagnostics_summary(repo, monkeypatch, capsys):
    p = _register(repo)
    _commit_repro_pass(repo, p)
    monkeypatch.setattr(J, "_load_data", _fake_loader(drift=-0.004, calls=[]))
    assert cli_main(["run", str(p), "--kind", "test"]) == 1
    out = capsys.readouterr().out
    assert "trades" in out and "win rate" in out and "stop-outs" in out
    assert "worst trade" in out and "worst day" in out and "highest exposure" in out and "x target" in out


def test_cli_gross_exposure_bug_is_loud_and_writes_nothing(repo, monkeypatch, capsys):
    p = _register(repo)
    _commit_repro_pass(repo, p)
    monkeypatch.setattr(J, "_load_data", _fake_loader(drift=0.003, calls=[]))

    def bug(*a, **k):
        raise D.GrossExposureError("judge bug: executed gross exposure 1.2 on 2025-06-01 exceeds gross_cap 1.0")

    monkeypatch.setattr(D, "strategy_diagnostics", bug)
    assert cli_main(["run", str(p), "--kind", "test"]) == 4
    assert "JUDGE BUG" in capsys.readouterr().err
    assert not (repo / "results" / f"{NAME}-test.json").exists()
