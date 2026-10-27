"""Task 8: reproduction gate (exchange price loaders, reproduce, Test-time gate)."""
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from judge import data
from judge import judge as J
from judge.__main__ import main as cli_main
from test_prereg import NAME, _commit, _fake_loader, _git, _register, _write_bytes, repo  # noqa: F401

DRAFTS = Path(__file__).resolve().parent.parent / "prereg" / "drafts"


# ---------------------------------------------------------------- HTTP fakes

class _Resp:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload
        self.text = json.dumps(payload)
        self.content = self.text.encode()

    def json(self):
        return self._payload


def _day_s(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


def _bitstamp_rows(days, closes):
    return [{"timestamp": str(_day_s(d)), "open": "1", "high": "1", "low": "1", "close": f"{c}",
             "volume": "1"} for d, c in zip(days, closes)]


# A live Poloniex ETH_BTC DAY_1 row (2026-10-27 probe): the first day of the
# market. Fields: low, high, open, close, amount, quantity, buyTakerAmount,
# buyTakerQuantity, tradeCount, ts, weightedAverage, interval, startTime, closeTime.
POLONIEX_LIVE_ROW = ["0.00262", "50", "50", "0.00312", "5.46178", "1205.803", "0", "0", 0, 1438992000000,
                     "0.00452", "DAY_1", 1438992000000, 1439078399999]


def _poloniex_row(d: date, close: float):
    ms = _day_s(d) * 1000
    return ["0.1", "0.3", "0.2", f"{close}", "1", "1", "0", "0", 1, ms, "0.2", "DAY_1", ms, ms + 86_399_999]


# ---------------------------------------------------------------- rel_error

def test_rel_error_formula():
    c = J.repro_check(1.56, 1.30)
    assert c["target"] == 1.56 and c["ours"] == 1.30
    assert c["rel_error"] == pytest.approx(0.1667, abs=1e-4)
    assert c["passed"] is True


def test_rel_error_fail():
    c = J.repro_check(1.56, 1.20)
    assert c["rel_error"] == pytest.approx(0.2308, abs=1e-4)
    assert c["passed"] is False


def test_rel_error_boundary_and_sign():
    assert J.repro_check(1.0, 1.2)["passed"] is True  # exactly 20%
    assert J.repro_check(1.56, 1.88)["passed"] is False  # too high fails too
    assert J.repro_check(1.31, -1.31)["rel_error"] == pytest.approx(2.0)


# ---------------------------------------------------------------- Bitstamp

def test_bitstamp_paginates_and_parses_close(tmp_path):
    days = pd.date_range("2015-01-01", "2015-01-08", freq="D").date
    closes = [100.0 + i for i in range(len(days))]
    rows = _bitstamp_rows(days, closes)
    calls = []

    def fake_get(url, params=None, timeout=None, **kw):
        assert url == data.BITSTAMP_OHLC
        calls.append(dict(params))
        start = int(params["start"])
        assert params["step"] == 86400
        page = [r for r in rows if int(r["timestamp"]) >= start][: params["limit"]]
        return _Resp({"data": {"pair": "BTC/USD", "ohlc": page}})

    with mock.patch.object(data, "BITSTAMP_LIMIT", 3), mock.patch("judge.data.requests.get", fake_get):
        s = data.bitstamp_btcusd(date(2015, 1, 1), date(2015, 1, 7), tmp_path)
    # Pages of 3 from start; the last page (2 rows, short) ends the loop.
    assert [c["start"] for c in calls] == [_day_s(days[0]), _day_s(days[3]), _day_s(days[6])]
    assert list(s.index.date) == list(days[:7])  # cut to end
    assert list(s) == closes[:7]
    assert s.dtype == float
    # Raw responses are cached and reused.
    assert list((tmp_path / "bitstamp").glob("*.json"))
    with mock.patch.object(data, "BITSTAMP_LIMIT", 3), \
            mock.patch("judge.data.requests.get", lambda *a, **k: pytest.fail("cache must be used")):
        s2 = data.bitstamp_btcusd(date(2015, 1, 1), date(2015, 1, 7), tmp_path)
    pd.testing.assert_series_equal(s, s2)


def test_bitstamp_rejects_non_midnight_candle(tmp_path):
    bad = [{"timestamp": str(_day_s(date(2015, 1, 1)) + 3600), "close": "1"}]
    with mock.patch("judge.data.requests.get", lambda *a, **k: _Resp({"data": {"ohlc": bad}})):
        with pytest.raises(data.DataIntegrityError, match="00:00"):
            data.bitstamp_btcusd(date(2015, 1, 1), date(2015, 1, 1), tmp_path)


# ---------------------------------------------------------------- Poloniex

def test_poloniex_field_order_pinned_on_live_row():
    s = data._parse_poloniex([POLONIEX_LIVE_ROW])
    assert list(s.index) == [pd.Timestamp("2015-08-08")]
    assert s.iloc[0] == 0.00312  # close, index 3 (not low 0.00262, not open/high 50)
    assert data.POLONIEX_CLOSE == 3 and data.POLONIEX_START_MS == 12


def test_poloniex_paginates_in_limit_sized_windows(tmp_path):
    days = list(pd.date_range("2015-08-08", "2015-08-14", freq="D").date)
    rows = [_poloniex_row(d, 0.001 * (i + 1)) for i, d in enumerate(days)]
    calls = []

    def fake_get(url, params=None, timeout=None, **kw):
        assert url == data.POLONIEX_CANDLES
        assert params["interval"] == "DAY_1"
        calls.append((params["startTime"], params["endTime"]))
        return _Resp([r for r in rows if params["startTime"] <= r[12] <= params["endTime"]])

    with mock.patch.object(data, "POLONIEX_LIMIT", 3), mock.patch("judge.data.requests.get", fake_get):
        s = data.poloniex_ethbtc(date(2015, 8, 8), date(2015, 8, 14), tmp_path)
    assert len(calls) == 3
    for lo, hi in calls:
        assert hi - lo + 1 == 3 * 86_400_000
    assert list(s.index.date) == days
    assert s.iloc[-1] == pytest.approx(0.007)


def test_eth_is_ethbtc_times_btcusd_same_day(tmp_path):
    days = pd.date_range("2015-08-08", "2015-08-12", freq="D")
    btc = pd.Series([200.0, 210.0, 220.0, 230.0, 240.0], index=days)
    ethbtc = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=days)
    with mock.patch.object(data, "bitstamp_btcusd", lambda s, e, c: btc), \
            mock.patch.object(data, "poloniex_ethbtc", lambda s, e, c: ethbtc):
        with data.record_api_prices() as rec:
            eth = data.exchange_daily("ETH", date(2015, 8, 8), date(2015, 8, 12), tmp_path)
            btc_usd = data.exchange_daily("BTC", date(2015, 8, 8), date(2015, 8, 12), tmp_path)
    assert list(eth) == pytest.approx([2.0, 4.2, 6.6, 9.2, 12.0])
    assert eth.index.equals(days)
    assert list(btc_usd) == list(btc)
    assert set(rec) == {"bitstamp:btcusd", "poloniex:ETH_BTC", "usd:ETH", "usd:BTC"}
    assert rec["usd:ETH"] == data.price_series_sha256(eth)


def test_exchange_daily_refuses_short_coverage(tmp_path):
    days = pd.date_range("2015-08-10", "2015-08-12", freq="D")
    s = pd.Series([1.0, 2.0, 3.0], index=days)
    with mock.patch.object(data, "bitstamp_btcusd", lambda a, b, c: s):
        with pytest.raises(data.DataIntegrityError, match="2015-08-10"):
            data.exchange_daily("BTC", date(2015, 8, 8), date(2015, 8, 12), tmp_path)
        with pytest.raises(data.DataIntegrityError, match="2015-08-12"):
            data.exchange_daily("BTC", date(2015, 8, 10), date(2015, 8, 14), tmp_path)


def test_price_series_sha256_canonical():
    s = pd.Series([1.5, 2.0], index=pd.to_datetime(["2015-01-02", "2015-01-01"]))
    expected = hashlib.sha256(b"date,close\n2015-01-01,2.0\n2015-01-02,1.5\n").hexdigest()
    assert data.price_series_sha256(s) == expected


# ---------------------------------------------------------------- gaps

def test_gap_forward_fill_limit():
    days = pd.to_datetime(["2015-01-01", "2015-01-05", "2015-01-06"])  # 3 days missing: filled
    s = data.fill_gaps(pd.Series([1.0, 2.0, 3.0], index=days), "x")
    assert list(s.index) == list(pd.date_range("2015-01-01", "2015-01-06", freq="D"))
    assert list(s) == [1.0, 1.0, 1.0, 1.0, 2.0, 3.0]
    days = pd.to_datetime(["2015-01-01", "2015-01-06"])  # 4 days missing: refused, named
    with pytest.raises(data.DataIntegrityError, match="2015-01-02.*2015-01-05"):
        data.fill_gaps(pd.Series([1.0, 2.0], index=days), "x")


def test_exchange_daily_applies_gap_limit(tmp_path):
    days = pd.to_datetime(["2015-01-01", "2015-01-02", "2015-01-07"])
    s = pd.Series([1.0, 2.0, 3.0], index=days)
    with mock.patch.object(data, "bitstamp_btcusd", lambda a, b, c: s):
        with pytest.raises(data.DataIntegrityError, match="bitstamp"):
            data.exchange_daily("BTC", date(2015, 1, 1), date(2015, 1, 7), tmp_path)


# ---------------------------------------------------------------- drafts and params

def test_drafts_hold_the_brief_values():
    for name, ls, seed in (("01-zarattini-long", False, 20251020), ("02-zarattini-long-short", True, 20251021)):
        p = DRAFTS / f"{name}.json"
        params = J.load_prereg_params(p)
        assert params["candidate_id"] == name  # equals the file stem, as a registered prereg must
        assert params["long_short"] is ls and params["seed"] == seed
        assert params["lookbacks"] == [5, 10, 20, 30, 60, 90, 150, 250, 360]
        assert params["vol_target"] == 0.25 and params["vol_window"] == 90
        assert params["per_coin_cap_repro"] == 2.0
        assert params["paper_sample_end"] == "2025-03-19"
        assert params["judge_tree_hash"] == "0" * 40
        assert "5209907" in params["paper"]
        for k, v in J.JUDGING.items():
            assert params[k] == v


def test_load_prereg_params_validates_without_integrity(tmp_path):
    d = json.loads((DRAFTS / "01-zarattini-long.json").read_text())
    p = tmp_path / "01-zarattini-long.json"
    p.write_text(json.dumps(d))
    assert J.load_prereg_params(p)["candidate_id"] == "01-zarattini-long"  # not in a git repo: fine
    d["vol_window"] = 90.5
    p.write_text(json.dumps(d))
    with pytest.raises(J.PreregError, match="vol_window"):
        J.load_prereg_params(p)


def test_strategy_sha256_canonical_and_selective():
    base = json.loads((DRAFTS / "01-zarattini-long.json").read_text())
    sha = J.strategy_sha256(base)
    canon = json.dumps({"long_short": False, "lookbacks": [5, 10, 20, 30, 60, 90, 150, 250, 360],
                        "per_coin_cap_repro": 2.0, "vol_target": 0.25, "vol_window": 90},
                       sort_keys=True, separators=(",", ":"))
    assert sha == hashlib.sha256(canon.encode()).hexdigest()
    assert J.strategy_sha256({**base, "seed": 1, "candidate_id": "x"}) == sha  # not strategy fields
    assert J.strategy_sha256({**base, "per_coin_cap_repro": 2}) == sha  # 2 and 2.0 are one value
    for key, val in (("long_short", True), ("lookbacks", [5, 10]), ("vol_target", 0.3), ("vol_window", 60),
                     ("per_coin_cap_repro", 1.5)):
        assert J.strategy_sha256({**base, key: val}) != sha


# ---------------------------------------------------------------- reproduce

def _price_loader(calls, drift=0.0005, seed=1):
    def loader(asset, start, end, cache_dir):
        calls.append((asset, start, end))
        days = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
        rng = np.random.default_rng(seed + (asset == "ETH"))
        s = pd.Series(100 * np.cumprod(1 + drift + 0.03 * rng.standard_normal(len(days))), index=days)
        data.note_api_price(f"usd:{asset}", s)
        return s
    return loader


def _draft(repo, name="01-demo-long", **over) -> Path:
    d = json.loads((DRAFTS / "01-zarattini-long.json").read_text())
    d.update({"candidate_id": name, "n_sims": J.JUDGING["n_sims"]}, **over)  # the fixture lowers n_sims
    p = repo / "prereg" / "drafts" / f"{name}.json"
    _write_bytes(p, json.dumps(d, indent=2) + "\n")
    _commit(repo, "draft")
    return p


def test_repro_uses_paper_costs_not_judging_costs(repo, monkeypatch):
    seen = []
    real = J.net_returns

    def spy(weights, changed, close, funding, cost_per_side, threshold):
        seen.append((cost_per_side, threshold, funding))
        return real(weights, changed, close, funding, cost_per_side, threshold)

    monkeypatch.setattr(J, "net_returns", spy)
    p = _draft(repo, "02-demo-ls", long_short=True)
    res = J.reproduce(p, loader=_price_loader([]))
    checks = res["checks"]
    assert set(checks) == {"btc_long_only_net", "eth_long_only_net", "btc_long_short_gross"}
    assert checks["btc_long_only_net"]["cost_per_side"] == 0.0010 == J.REPRO["cost_per_side"]
    assert checks["eth_long_only_net"]["cost_per_side"] == 0.0010
    assert checks["btc_long_short_gross"]["cost_per_side"] == 0.0
    assert sorted(c for c, _, _ in seen) == [0.0, 0.0010, 0.0010]
    assert all(t == 0.2 and f is None for _, t, f in seen)  # paper threshold, spot: no funding
    assert res["settings"]["cost_per_side"] == 0.0010 and res["settings"]["gross_cap"] == 2.0


def test_repro_checks_periods_and_targets(repo):
    calls = []
    p = _draft(repo)
    res = J.reproduce(p, loader=_price_loader(calls))
    assert calls == [("BTC", date(2015, 1, 1), date(2025, 3, 19)), ("ETH", date(2015, 8, 8), date(2025, 3, 19))]
    checks = res["checks"]
    assert set(checks) == {"btc_long_only_net", "eth_long_only_net"}  # long-only candidate
    assert checks["btc_long_only_net"]["target"] == 1.56 and checks["eth_long_only_net"]["target"] == 1.51
    assert checks["btc_long_only_net"]["period"] == {"start": "2015-01-01", "end": "2025-03-19"}
    assert checks["eth_long_only_net"]["period"] == {"start": "2015-08-08", "end": "2025-03-19"}
    # Returns from the day after the first close to the end, inclusive.
    assert checks["btc_long_only_net"]["scored_days"] == (date(2025, 3, 19) - date(2015, 1, 1)).days
    assert "2015-08-08" in res["period_notes"]["ETH"] and "2015-08-01" in res["period_notes"]["ETH"]
    for c in checks.values():
        assert c["rel_error"] == pytest.approx(abs(c["ours"] - c["target"]) / abs(c["target"]))
    assert res["passed"] == all(c["passed"] for c in checks.values())


def test_repro_result_written_on_fail_and_never_overwritten(repo):
    p = _draft(repo)
    res = J.reproduce(p, loader=_price_loader([], drift=-0.002))
    out = repo / "results" / "01-demo-long-repro.json"
    assert out.exists()
    saved = json.loads(out.read_text())
    assert saved == res and saved["passed"] is False and saved["kind"] == "repro"
    assert saved["candidate_id"] == "01-demo-long"
    assert saved["strategy_sha256"] == J.strategy_sha256(J.load_prereg_params(p))
    m = saved["manifest"]
    assert m["api_price_sha256"]["usd:BTC"] and m["api_price_sha256"]["usd:ETH"]
    assert m["draft_file"] == "prereg/drafts/01-demo-long.json"
    assert m["draft_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert m["judge_tree_hash"] == _git(repo, "rev-parse", "HEAD:judge")
    assert m["pip_freeze"] == ["numpy==0.0-test"] and m["run_utc"].endswith("Z")
    with pytest.raises(J.ResultExists):
        J.reproduce(p, loader=_price_loader([]))


def test_repro_overall_pass_needs_every_check(repo, monkeypatch):
    p = _draft(repo, "02-demo-ls", long_short=True)
    monkeypatch.setattr(J, "sharpe", lambda r: 1.5)  # within 20% of 1.56, 1.51 and 1.31
    assert J.reproduce(p, loader=_price_loader([]))["passed"] is True
    p2 = _draft(repo, "03-demo-ls", long_short=True)
    values = iter([1.5, 1.5, 1.6])  # check order: btc long-only, eth long-only, btc long-short gross
    monkeypatch.setattr(J, "sharpe", lambda r: next(values))
    res = J.reproduce(p2, loader=_price_loader([]))
    assert res["checks"]["btc_long_short_gross"]["passed"] is False and res["passed"] is False


def test_repro_refuses_uncommitted_draft_or_judge(repo):
    p = _draft(repo)
    _write_bytes(p, p.read_text().replace('"seed": 20251020', '"seed": 1'))
    with pytest.raises(J.PreregMismatch, match="commit"):
        J.reproduce(p, loader=_price_loader([]))
    _commit(repo, "fix draft")
    _write_bytes(repo / "judge" / "__init__.py", "# dirty\n")
    with pytest.raises(J.PreregMismatch, match="judge"):
        J.reproduce(p, loader=_price_loader([]))
    assert not (repo / "results" / "01-demo-long-repro.json").exists()


def test_repro_refuses_candidate_id_not_file_stem(repo):
    p = _draft(repo, "01-demo-long")
    _write_bytes(p, p.read_text().replace('"01-demo-long"', '"zarattini-long"'))
    _commit(repo, "rename id")
    with pytest.raises(J.PreregError, match="file name"):
        J.reproduce(p, loader=_price_loader([]))


def test_repro_refuses_other_paper_sample_end(repo):
    p = _draft(repo, paper_sample_end="2024-12-31")
    with pytest.raises(J.PreregError, match="paper_sample_end"):
        J.reproduce(p, loader=_price_loader([]))


def test_cli_reproduce(repo, monkeypatch, capsys):
    p = _draft(repo)
    monkeypatch.setattr(J, "exchange_daily", _price_loader([], drift=-0.002))
    assert cli_main(["reproduce", str(p)]) == 1  # completed, did not pass
    out = capsys.readouterr().out
    assert "FAILED" in out and "btc_long_only_net" in out and "01-demo-long-repro.json" in out
    assert cli_main(["reproduce", str(p)]) == 2  # result exists: refused
    assert "already exists" in capsys.readouterr().err


# ---------------------------------------------------------------- Test-time gate

def _commit_repro(repo: Path, p: Path, passed=True, strategy=None, kind="repro", commit=True) -> Path:
    params = J.load_prereg(p)
    rec = {"candidate_id": params["candidate_id"], "kind": kind, "passed": passed,
           "strategy_sha256": strategy or J.strategy_sha256(params)}
    out = repo / "results" / f"{params['candidate_id']}-repro.json"
    _write_bytes(out, json.dumps(rec) + "\n")
    if commit:
        _commit(repo, "repro result")
    return out


def test_repro_gate_refuses_test_without_repro_result(repo, monkeypatch):
    p = _register(repo)
    monkeypatch.setattr(J, "_load_data", lambda *a: pytest.fail("data must not load"))
    with pytest.raises(J.ReproGateError, match="reproduce"):
        J.run(p, "test")
    assert not (repo / "results" / f"{NAME}-test.json").exists()


def test_repro_gate_refuses_failed_repro(repo, monkeypatch):
    p = _register(repo)
    _commit_repro(repo, p, passed=False)
    monkeypatch.setattr(J, "_load_data", lambda *a: pytest.fail("data must not load"))
    with pytest.raises(J.ReproGateError, match="passed: false"):
        J.run(p, "test")


def test_repro_gate_refuses_mismatched_strategy_sha256(repo, monkeypatch):
    p = _register(repo)
    _commit_repro(repo, p, strategy="f" * 64)
    monkeypatch.setattr(J, "_load_data", lambda *a: pytest.fail("data must not load"))
    with pytest.raises(J.ReproGateError, match="strategy_sha256"):
        J.run(p, "test")


def test_repro_gate_refuses_uncommitted_or_wrong_kind(repo, monkeypatch):
    monkeypatch.setattr(J, "_load_data", lambda *a: pytest.fail("data must not load"))
    p = _register(repo)
    _commit_repro(repo, p, commit=False)
    with pytest.raises(J.ReproGateError, match="commit"):
        J.run(p, "test")
    _commit(repo, "repro")
    _commit_repro(repo, p, kind="test")
    with pytest.raises(J.ReproGateError, match="repro result"):
        J.run(p, "test")


def test_repro_gate_passes_matching_repro_and_records_it(repo, monkeypatch):
    p = _register(repo)
    _commit_repro(repo, p)
    monkeypatch.setattr(J, "_load_data", _fake_loader(drift=0.003, calls=[]))
    out = J.run(p, "test")
    m = json.loads(out.read_text())["manifest"]
    assert m["repro"]["path"] == f"results/{NAME}-repro.json"
    assert m["repro"]["strategy_sha256"] == J.strategy_sha256(J.load_prereg(p))


def test_repro_gate_applies_to_final_alone(repo, monkeypatch):
    # A Final-alone candidate: Test window under 12 months.
    p = _register(repo, paper_sample_end="2026-01-31")
    monkeypatch.setattr(J, "_load_data", lambda *a: pytest.fail("data must not load"))
    _write_bytes(p.with_suffix(".stamp_confirmed"), "2026-10-25\n")
    _commit(repo, "stamp")
    with pytest.raises(J.ReproGateError):
        J.prepare_final(p)
    with pytest.raises(J.ReproGateError):
        J.run(p, "final_alone")
    assert not (repo / "results" / f"{NAME}.final_opened").exists()
