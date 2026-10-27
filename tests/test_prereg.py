"""Task 7: pre-registration guard, windows, Final gate, result file, CLI."""
import hashlib
import json
import subprocess
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from judge import data
from judge import judge as J
from judge.__main__ import main as cli_main
from judge.judge import (
    FinalGateError,
    FinalNotReady,
    PreregError,
    PreregMismatch,
    check_integrity,
    load_prereg,
    open_final,
    run,
    window,
)

NAME = "01-demo-long"


def _git(repo: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", *args],
        cwd=repo, capture_output=True, text=True, check=True,
    )
    return res.stdout.strip()


def _prereg_dict(tree_hash: str, **over) -> dict:
    d = {
        "candidate_id": NAME,
        "paper": "SSRN 5209907",
        "long_short": False,
        "lookbacks": [5, 10, 20],
        "vol_target": 0.25,
        "vol_window": 30,
        "per_coin_cap_repro": 2.0,
        "gross_cap": 1.0,
        "cost_per_side": 0.0011,
        "rebalance_threshold": 0.2,
        "universe_size": 20,
        "paper_sample_end": "2025-03-19",
        "seed": 7,
        "n_sims": 8,
        "judge_tree_hash": tree_hash,
    }
    d.update(over)
    return d


def _write_bytes(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode())


def _register(repo: Path, **over) -> Path:
    """Write prereg/NN-name.json (+ .sha256) for the committed judge tree; commit."""
    tree = _git(repo, "rev-parse", "HEAD:judge")
    p = repo / "prereg" / f"{NAME}.json"
    _write_bytes(p, json.dumps(_prereg_dict(tree, **over), indent=2) + "\n")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    _write_bytes(p.with_suffix(".sha256"), f"{sha}  {p.name}\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "register")
    return p


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _write_bytes(root / ".gitattributes", "* text=auto eol=lf\n")
    _write_bytes(root / ".gitignore", "results/\ndata_cache/\n")
    _write_bytes(root / "judge" / "__init__.py", "# judge\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "judge")
    monkeypatch.setattr(J, "CODE_ROOT", root)
    monkeypatch.setattr(J, "_pip_freeze", lambda: ["numpy==0.0-test"])
    return root


# ---------------------------------------------------------------- load / integrity

def test_missing_key_raises_prereg_error(tmp_path):
    d = _prereg_dict("abc")
    del d["seed"]
    p = tmp_path / "01-x.json"
    p.write_text(json.dumps(d))
    with pytest.raises(PreregError, match="seed"):
        load_prereg(p)
    d = _prereg_dict("abc")
    p.write_text(json.dumps(d))
    assert load_prereg(p)["candidate_id"] == NAME


def test_bad_candidate_id_refused(tmp_path):
    p = tmp_path / "01-x.json"
    p.write_text(json.dumps(_prereg_dict("abc", candidate_id="../evil")))
    with pytest.raises(PreregError, match="candidate_id"):
        load_prereg(p)


def test_edited_prereg_refused(repo):
    # Review Focus 5: an edit after stamping, even one that is committed.
    p = _register(repo)
    check_integrity(p)  # as registered: fine
    registered = p.with_suffix(".sha256").read_text().split()[0]
    p.write_bytes(p.read_bytes().replace(b'"seed": 7', b'"seed": 8'))
    _git(repo, "commit", "-q", "-am", "tweak seed")
    edited = hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(PreregMismatch) as ei:
        check_integrity(p)
    msg = str(ei.value)
    assert p.name in msg and registered in msg and edited in msg


def test_uncommitted_edit_under_prereg_or_judge_refused(repo):
    p = _register(repo)
    _write_bytes(repo / "prereg" / "notes.txt", "untracked\n")
    with pytest.raises(PreregMismatch, match="prereg"):
        check_integrity(p)
    (repo / "prereg" / "notes.txt").unlink()
    _write_bytes(repo / "judge" / "__init__.py", "# changed, not committed\n")
    with pytest.raises(PreregMismatch, match="judge"):
        check_integrity(p)


def test_judge_tree_hash_mismatch_refused(repo):
    p = _register(repo)
    recorded = load_prereg(p)["judge_tree_hash"]
    _write_bytes(repo / "judge" / "__init__.py", "# judge v2\n")
    _git(repo, "commit", "-q", "-am", "change judge")
    head = _git(repo, "rev-parse", "HEAD:judge")
    with pytest.raises(PreregMismatch) as ei:
        check_integrity(p)
    msg = str(ei.value)
    assert "judge" in msg and recorded in msg and head in msg


def test_missing_sha256_companion_refused(repo):
    p = _register(repo)
    p.with_suffix(".sha256").unlink()
    _git(repo, "commit", "-q", "-am", "drop sha")
    with pytest.raises(PreregMismatch, match="sha256"):
        check_integrity(p)


# ---------------------------------------------------------------- windows

def test_test_window_starts_after_paper_sample():
    pre = _prereg_dict("x", paper_sample_end="2025-03-19")
    assert window("test", pre, None) == (date(2025, 3, 20), date(2026, 10, 19))
    early = _prereg_dict("x", paper_sample_end="2021-06-30")
    assert window("test", early, None) == (date(2023, 1, 1), date(2026, 10, 19))


def test_final_needs_stamp_and_full_data():
    pre = _prereg_dict("x")
    with pytest.raises(FinalNotReady, match="stamp_confirmed"):
        window("final", pre, None, latest_data_date=date(2026, 10, 20))
    # 90 days starting the day after the stamp date.
    assert window("final", pre, date(2026, 6, 1), latest_data_date=date(2026, 10, 20)) == (
        date(2026, 6, 2), date(2026, 8, 30))
    assert window("final_alone", pre, date(2026, 6, 1), latest_data_date=date(2026, 8, 30))[1] == date(2026, 8, 30)
    with pytest.raises(FinalNotReady, match="2026-08-29"):
        window("final", pre, date(2026, 6, 1), latest_data_date=date(2026, 8, 29))


# ---------------------------------------------------------------- final gate

def _write_test_result(results: Path, passed: bool) -> None:
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{NAME}-test.json").write_text(json.dumps({"passed": passed}))


def test_second_final_open_refused(tmp_path):
    res = tmp_path / "results"
    _write_test_result(res, True)
    open_final(NAME, results_dir=res)
    assert (res / f"{NAME}.final_opened").exists()
    with pytest.raises(FinalGateError, match="already"):
        open_final(NAME, results_dir=res)


def test_final_gate_needs_test_pass_unless_final_alone(tmp_path):
    res = tmp_path / "results"
    with pytest.raises(FinalGateError, match="Test result"):
        open_final(NAME, results_dir=res)  # no Test result at all
    _write_test_result(res, False)
    with pytest.raises(FinalGateError, match="passed: false"):
        open_final(NAME, results_dir=res)
    assert not (res / f"{NAME}.final_opened").exists()
    open_final(NAME, final_alone=True, results_dir=res)
    assert (res / f"{NAME}.final_opened").exists()


def test_final_opened_written_before_data_load(repo, monkeypatch):
    p = _register(repo)
    _write_test_result(repo / "results", True)
    seen = {}

    class Stop(Exception):
        pass

    def loader(market, start, end, cache_dir):
        seen["opened"] = (repo / "results" / f"{NAME}.final_opened").exists()
        seen["args"] = (market, start, end)
        raise Stop

    monkeypatch.setattr(J, "_load_data", loader)
    with pytest.raises(Stop):
        run(p, "final", stamp_confirmed=date(2026, 6, 1))
    assert seen["opened"] is True
    assert seen["args"] == ("um", date(2026, 6, 2) - timedelta(days=450), date(2026, 8, 30))


def test_final_kind_must_match_candidate_type(repo, monkeypatch):
    monkeypatch.setattr(J, "_load_data", lambda *a: pytest.fail("data must not load"))
    p = _register(repo)  # Test window 2025-03-20..2026-10-19: not Final-alone
    with pytest.raises(FinalGateError, match="final_alone"):
        run(p, "final_alone", stamp_confirmed=date(2026, 6, 1))
    assert not (repo / "results" / f"{NAME}.final_opened").exists()


# ---------------------------------------------------------------- run end to end

def _synthetic(start: date, end: date, n_coins=5, drift=0.0, seed=0):
    days = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D", name="date")
    rng = np.random.default_rng(seed)
    rets = drift + 0.03 * rng.standard_normal((len(days), n_coins))
    cols = [f"C{i}USDT" for i in range(n_coins)]
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=days, columns=cols)
    qv = pd.DataFrame(5e7, index=days, columns=cols)
    funding = pd.DataFrame(0.0001, index=days, columns=cols)
    close.attrs["last_date"] = {c: days[-1] for c in cols}
    close.attrs["skipped"] = ["DEADUSDT"]
    return close, qv, funding


def _fake_loader(drift, calls, seed=0):
    def loader(market, start, end, cache_dir):
        calls.append((market, start, end))
        # Pretend one archive file was fetched and verified.
        z = Path(cache_dir) / "data.binance.vision" / "data" / "futures" / "um" / "x.zip"
        z.parent.mkdir(parents=True, exist_ok=True)
        z.write_bytes(b"zip")
        data.checksum_path(z).write_text(f"{'ab' * 32}  x.zip\n")
        data.note_fetch(z)
        data.note_api_funding("C1USDT", [(1_700_000_000_000, 0.0001)])
        return _synthetic(start, end, drift=drift, seed=seed)
    return loader


def test_failed_run_still_writes_result_file(repo, monkeypatch):
    p = _register(repo)
    calls = []
    monkeypatch.setattr(J, "_load_data", _fake_loader(drift=-0.004, calls=calls))
    out = run(p, "test")
    assert out == repo / "results" / f"{NAME}-test.json"
    res = json.loads(out.read_text())
    assert res["passed"] is False
    assert res["verdict"]["passed"] is False
    assert calls == [("um", date(2025, 3, 20) - timedelta(days=450), date(2026, 10, 19))]


def test_run_writes_result_with_manifest(repo, monkeypatch):
    p = _register(repo)
    monkeypatch.setattr(J, "_load_data", _fake_loader(drift=0.003, calls=[]))
    out = run(p, "test")
    res = json.loads(out.read_text())
    assert res["candidate_id"] == NAME and res["kind"] == "test"
    assert res["window"] == {"start": "2025-03-20", "end": "2026-10-19"}
    # Returns scored from the window's first day to its last, inclusive.
    assert res["scored_days"] == (date(2026, 10, 19) - date(2025, 3, 20)).days + 1
    assert 0.0 <= res["luck_odds"] <= 1.0
    assert 0.0 < res["luck_odds_flip_prob"] < 1.0
    assert set(res["verdict"]) == {"passed", "sharpe", "max_dd", "bench_sharpe", "rules"}
    assert res["passed"] == res["verdict"]["passed"]
    m = res["manifest"]
    assert m["prereg_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert m["judge_tree_hash"] == _git(repo, "rev-parse", "HEAD:judge")
    assert m["seed"] == 7 and m["n_sims"] == 8
    assert m["pip_freeze"] == ["numpy==0.0-test"]
    assert m["archive_checksums"] == {"data.binance.vision/data/futures/um/x.zip": "ab" * 32}
    assert m["api_funding_sha256"] == {"C1USDT": data.funding_rows_sha256("C1USDT", [(1_700_000_000_000, 0.0001)])}
    assert m["data_last_date"] == "2026-10-19"
    assert m["skipped_symbols"] == ["DEADUSDT"]
    assert m["market"] == "um"
    assert m["window"] == res["window"]
    assert "lookback" in m["luck_odds_method"]
    assert m["run_utc"].endswith("Z")
    # A result is never overwritten.
    with pytest.raises(J.ResultExists):
        run(p, "test")


def test_run_refuses_when_data_short_of_window(repo, monkeypatch):
    p = _register(repo)

    def short(market, start, end, cache_dir):
        return _synthetic(start, end - timedelta(days=5))

    monkeypatch.setattr(J, "_load_data", short)
    with pytest.raises(J.DataNotReady, match="2026-10-14"):
        run(p, "test")
    assert not (repo / "results" / f"{NAME}-test.json").exists()


def test_run_refuses_code_outside_prereg_repo(repo, monkeypatch, tmp_path):
    p = _register(repo)
    monkeypatch.setattr(J, "CODE_ROOT", tmp_path / "elsewhere")
    with pytest.raises(PreregMismatch, match="elsewhere"):
        run(p, "test")


# ---------------------------------------------------------------- CLI

def test_cli_final_without_stamp_is_clear_error(repo, capsys):
    p = _register(repo)
    _write_test_result(repo / "results", True)
    assert cli_main(["run", str(p), "--kind", "final"]) == 2
    err = capsys.readouterr().err
    assert "stamp_confirmed" in err and ".stamp_confirmed" in err and "--stamp-confirmed" in err
    assert not (repo / "results" / f"{NAME}.final_opened").exists()


def test_stamp_file_wins_and_conflicting_flag_refused(repo):
    p = _register(repo)
    stamp = p.with_suffix(".stamp_confirmed")
    _write_bytes(stamp, "2026-06-01\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "stamp confirmed")
    assert J.resolve_stamp_confirmed(p, None) == date(2026, 6, 1)
    assert J.resolve_stamp_confirmed(p, date(2026, 6, 1)) == date(2026, 6, 1)
    with pytest.raises(PreregError, match="conflict"):
        J.resolve_stamp_confirmed(p, date(2026, 6, 2))
    stamp.unlink()
    assert J.resolve_stamp_confirmed(p, date(2026, 6, 2)) == date(2026, 6, 2)
    assert J.resolve_stamp_confirmed(p, None) is None
    _write_bytes(stamp, "2026-06-01\n2026-06-02\n")
    with pytest.raises(PreregError, match="single"):
        J.resolve_stamp_confirmed(p, None)


def test_cli_test_run_reports_and_exit_code(repo, monkeypatch, capsys):
    p = _register(repo)
    monkeypatch.setattr(J, "_load_data", _fake_loader(drift=-0.004, calls=[]))
    assert cli_main(["run", str(p), "--kind", "test"]) == 1  # completed, did not pass
    out = capsys.readouterr().out
    assert "FAILED" in out and f"{NAME}-test.json" in out


# ---------------------------------------------------------------- data: checksum sidecar

def test_fetch_file_keeps_checksum_sidecar_and_records(tmp_path):
    import io
    import zipfile
    from unittest import mock

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("f.csv", "a,b\n")
    blob = buf.getvalue()
    url = "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2026-01.zip"
    body = f"{hashlib.sha256(blob).hexdigest()}  BTCUSDT-1d-2026-01.zip\n"

    class R:
        def __init__(self, content):
            self.status_code, self.content, self.text = 200, content, content.decode("utf-8", "ignore")

    def fake_get(u, params=None, timeout=None, **kw):
        return R(body.encode()) if u.endswith(".CHECKSUM") else R(blob)

    with mock.patch("judge.data.requests.get", fake_get):
        with data.record_fetches() as rec:
            path = data.fetch_file(url, tmp_path)
            data.fetch_file(url, tmp_path)  # cache hit is recorded too
    assert rec == [path, path]
    assert data.checksum_path(path).read_text() == body
    sums = J.archive_checksums(rec, tmp_path)
    assert sums == {"data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2026-01.zip":
                    hashlib.sha256(blob).hexdigest()}
    # Outside the block nothing is recorded.
    with mock.patch("judge.data.requests.get", fake_get):
        data.fetch_file(url, tmp_path)
    assert rec == [path, path]
