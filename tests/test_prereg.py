"""Task 7: pre-registration guard, windows, Final gate, result file, CLI."""
import hashlib
import json
import os
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
COMMIT_DATE = "2026-10-20T12:00:00+00:00"  # every test commit, so first-commit dates are fixed
STAMP = date(2026, 10, 25)
FINAL_WIN = (date(2026, 10, 26), date(2027, 1, 23))
FINAL_WIN_JSON = {"start": "2026-10-26", "end": "2027-01-23"}
TODAY = date(2027, 3, 1)  # Final data for FINAL_WIN exists
TEST_RUN_UTC = "2026-10-27T10:00:00Z"  # after the stamp: allowed


def _git(repo: Path, *args: str, when: str = COMMIT_DATE) -> str:
    res = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", *args],
        cwd=repo, capture_output=True, text=True, check=True,
        env={**os.environ, "GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when},
    )
    return res.stdout.strip()


def _prereg_dict(tree_hash: str = "0" * 40, **over) -> dict:
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
        "n_sims": J.JUDGING["n_sims"],
        "judge_tree_hash": tree_hash,
    }
    d.update(over)
    return d


def _write_bytes(path: Path, text) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text if isinstance(text, bytes) else text.encode())


def _commit(repo: Path, msg: str, when: str = COMMIT_DATE) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg, when=when)


def _register(repo: Path, name: str = NAME, **over) -> Path:
    """Write prereg/NN-name.json (+ .sha256) for the committed judge tree; commit."""
    tree = _git(repo, "rev-parse", "HEAD:judge")
    p = repo / "prereg" / f"{name}.json"
    _write_bytes(p, json.dumps(_prereg_dict(tree, **over), indent=2) + "\n")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    _write_bytes(p.with_suffix(".sha256"), f"{sha}  {p.name}\n")
    _commit(repo, "register")
    return p


def _stamp(repo: Path, p: Path, day: date = STAMP) -> None:
    _write_bytes(p.with_suffix(".stamp_confirmed"), f"{day.isoformat()}\n")
    _commit(repo, "stamp confirmed")


def _commit_test_result(repo: Path, p: Path, passed=True, **manifest_over) -> Path:
    m = {
        "prereg_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "judge_tree_hash": _git(repo, "rev-parse", "HEAD:judge"),
        "run_utc": TEST_RUN_UTC,
    }
    m.update(manifest_over)
    out = repo / "results" / f"{NAME}-test.json"
    _write_bytes(out, json.dumps({"candidate_id": NAME, "kind": "test", "passed": passed, "manifest": m}) + "\n")
    _commit(repo, "test result")
    return out


def _repro_record(repo: Path, params: dict, tree=None, ours=None) -> dict:
    """A repro result in reproduce()'s format that recomputes as a pass (each
    ``ours`` defaults to its target), with its price CSVs written under
    results/repro-data/ (not committed)."""
    checks = {}
    for name, asset, ls, gross, target, table in J.repro_plan(params["long_short"]):
        o = (ours or {}).get(name, target)
        rel = abs(o - target) / abs(target)
        checks[name] = {
            "target": target, "ours": o, "rel_error": rel, "passed": rel <= J.REPRO_REL_ERROR_LIMIT,
            "asset": asset, "long_short": ls, "gross_of_costs": gross,
            "cost_per_side": 0.0 if gross else J.REPRO["cost_per_side"],
            "rebalance_threshold": J.REPRO["rebalance_threshold"], "paper_table": table,
            "period": {"start": J.REPRO_START[asset].isoformat(), "end": J.REPRO_SAMPLE_END.isoformat()},
        }
    hashes = {}
    for asset in sorted({c["asset"] for c in checks.values()}):
        key = f"usd:{asset}:{J.REPRO_START[asset].isoformat()}..{J.REPRO_SAMPLE_END.isoformat()}"
        csv = f"date,close\n{J.REPRO_START[asset].isoformat()},100.0\n"
        _write_bytes(repo / "results" / J.PRICE_DATA_DIR / f"{J.price_data_name(key)}.csv", csv)
        hashes[key] = hashlib.sha256(csv.encode()).hexdigest()
    return {
        "candidate_id": params["candidate_id"], "kind": "repro",
        "passed": all(c["passed"] for c in checks.values()), "checks": checks,
        "rel_error_limit": J.REPRO_REL_ERROR_LIMIT, "worst_rel_error": max(c["rel_error"] for c in checks.values()),
        "strategy_sha256": J.strategy_sha256(params),
        "manifest": {"judge_tree_hash": tree or _git(repo, "rev-parse", "HEAD:judge"), "api_price_sha256": hashes},
    }


def _commit_repro_pass(repo: Path, p: Path, tree=None, ours=None, mutate=None, commit=True) -> Path:
    """Write (and commit) results/{id}-repro.json for p's rules: a pass unless
    ``ours`` or ``mutate(record)`` says otherwise (Task 8 gate for Test runs)."""
    params = load_prereg(p)
    rec = _repro_record(repo, params, tree=tree, ours=ours)
    if mutate is not None:
        mutate(rec)
    out = repo / "results" / f"{params['candidate_id']}-repro.json"
    _write_bytes(out, json.dumps(rec) + "\n")
    if commit:
        _commit(repo, "repro result")
    return out


def _gate_kw(repo: Path, p: Path, **over) -> dict:
    kw = dict(
        prereg_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        judge_tree_hash=_git(repo, "rev-parse", "HEAD:judge"),
        window=FINAL_WIN,
        results_dir=repo / "results",
    )
    kw.update(over)
    return kw


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _write_bytes(root / ".gitattributes", "* text=auto eol=lf\n")
    _write_bytes(root / ".gitignore", "data_cache/\n")  # results/ is committed, as in the real repo
    _write_bytes(root / "judge" / "__init__.py", "# judge\n")
    _commit(root, "judge")
    monkeypatch.setattr(J, "CODE_ROOT", root)
    monkeypatch.setattr(J, "_pip_freeze", lambda: ["numpy==0.0-test"])
    monkeypatch.setitem(J.JUDGING, "n_sims", 8)  # keep run() tests fast
    monkeypatch.setattr(J, "_today_utc", lambda: TODAY)
    return root


# ---------------------------------------------------------------- load / validation

def test_missing_key_raises_prereg_error(tmp_path):
    d = _prereg_dict()
    del d["seed"]
    p = tmp_path / "01-x.json"
    p.write_text(json.dumps(d))
    with pytest.raises(PreregError, match="seed"):
        load_prereg(p)
    p.write_text(json.dumps(_prereg_dict()))
    assert load_prereg(p)["candidate_id"] == NAME


def test_bad_candidate_id_refused(tmp_path):
    p = tmp_path / "01-x.json"
    p.write_text(json.dumps(_prereg_dict(candidate_id="../evil")))
    with pytest.raises(PreregError, match="candidate_id"):
        load_prereg(p)


@pytest.mark.parametrize("key,value", [
    ("cost_per_side", 0.0),
    ("cost_per_side", 0.001),  # the paper's 10 bps is repro only
    ("rebalance_threshold", None),
    ("rebalance_threshold", 0.1),
    ("gross_cap", None),
    ("gross_cap", 3.0),
    ("n_sims", 8),
    ("universe_size", 10),
    ("universe_size", 20.0),  # strict: must be a JSON integer
    ("long_short", "false"),  # strict: bool("false") is True
    ("long_short", 0),
    ("lookbacks", []),
    ("lookbacks", [5, 0]),
    ("lookbacks", [5, True]),
    ("vol_window", 30.5),
    ("vol_target", "0.25"),
    ("seed", True),
    ("judge_tree_hash", 123),
])
def test_judging_constants_and_types_enforced(tmp_path, key, value):
    p = tmp_path / "01-x.json"
    p.write_text(json.dumps(_prereg_dict(**{key: value})))
    with pytest.raises(PreregError, match=key):
        load_prereg(p)


def test_repro_cost_is_separate_from_judging():
    assert J.JUDGING["cost_per_side"] == 0.0011
    assert J.REPRO["cost_per_side"] == 0.0010
    assert "repro" not in J.KINDS  # run() can only judge with JUDGING


# ---------------------------------------------------------------- integrity

def test_edited_prereg_refused(repo):
    # Review Focus 5: an edit after stamping, even one that is committed.
    p = _register(repo)
    check_integrity(p)  # as registered: fine
    registered = p.with_suffix(".sha256").read_text().split()[0]
    p.write_bytes(p.read_bytes().replace(b'"seed": 7', b'"seed": 8'))
    _commit(repo, "tweak seed")
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
    _commit(repo, "change judge")
    head = _git(repo, "rev-parse", "HEAD:judge")
    with pytest.raises(PreregMismatch) as ei:
        check_integrity(p)
    msg = str(ei.value)
    assert "judge" in msg and recorded in msg and head in msg


def test_missing_sha256_companion_refused(repo):
    p = _register(repo)
    p.with_suffix(".sha256").unlink()
    _commit(repo, "drop sha")
    with pytest.raises(PreregMismatch, match="sha256"):
        check_integrity(p)


def test_prereg_outside_prereg_dir_refused(repo):
    # Review C1: an untracked copy with tweaked rules and a self-made .sha256.
    p = _register(repo)
    fake = repo / "scratch" / p.name
    _write_bytes(fake, p.read_bytes())
    _write_bytes(fake.with_suffix(".sha256"), p.with_suffix(".sha256").read_bytes())
    with pytest.raises(PreregMismatch, match="prereg/NN-name.json"):
        check_integrity(fake)


def test_prereg_must_be_nn_name_and_match_candidate_id(repo):
    p = _register(repo)
    with pytest.raises(PreregMismatch, match="candidate_id"):
        check_integrity(_register(repo, name="02-other"))  # candidate_id is still NAME
    with pytest.raises(PreregMismatch, match="NN-name"):
        check_integrity(_register(repo, name="demo", candidate_id="demo"))
    check_integrity(p)


def test_untracked_prereg_refused_even_if_tree_clean_elsewhere(repo):
    # A prereg/ file that was never committed (even with prereg/ otherwise clean).
    p = _register(repo)
    _git(repo, "rm", "-q", "--cached", str(p.relative_to(repo)))
    _git(repo, "commit", "-q", "-m", "untrack")
    with pytest.raises(PreregMismatch, match="not committed"):
        check_integrity(p)


def test_prereg_rewritten_with_new_sha_in_later_commit_refused(repo):
    # Review R2: edit the rules, regenerate .sha256, commit. Hashes agree, but
    # a registered prereg is written exactly once.
    p = _register(repo)
    p.write_bytes(p.read_bytes().replace(b'"seed": 7', b'"seed": 99'))
    _write_bytes(p.with_suffix(".sha256"), f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n")
    _commit(repo, "rewrite rules")
    with pytest.raises(PreregMismatch, match="exactly one commit"):
        check_integrity(p)


def test_sha256_companion_rewritten_alone_refused(repo):
    p = _register(repo)
    sha = p.with_suffix(".sha256")
    _write_bytes(sha, sha.read_text().replace("  ", " *"))  # same hash, new commit
    _commit(repo, "touch sha")
    with pytest.raises(PreregMismatch, match="exactly one commit"):
        check_integrity(p)


def test_crlf_working_copy_differs_from_committed_blob_refused(repo):
    # Review I7: with eol=lf, a CRLF working file commits as LF and git status
    # stays clean, but the bytes hashed and the bytes in git differ.
    tree = _git(repo, "rev-parse", "HEAD:judge")
    p = repo / "prereg" / f"{NAME}.json"
    _write_bytes(p, (json.dumps(_prereg_dict(tree), indent=2) + "\n").replace("\n", "\r\n"))
    _write_bytes(p.with_suffix(".sha256"), f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n")
    _commit(repo, "register crlf")
    assert _git(repo, "status", "--porcelain") == ""
    with pytest.raises(PreregMismatch, match="committed blob"):
        check_integrity(p)


# ---------------------------------------------------------------- windows

def test_test_window_starts_after_paper_sample():
    pre = _prereg_dict(paper_sample_end="2025-03-19")
    assert window("test", pre, None) == (date(2025, 3, 20), date(2026, 10, 19))
    early = _prereg_dict(paper_sample_end="2021-06-30")
    assert window("test", early, None) == (date(2023, 1, 1), date(2026, 10, 19))


def test_final_needs_stamp_and_full_data():
    pre = _prereg_dict()
    with pytest.raises(FinalNotReady, match="stamp_confirmed"):
        window("final", pre, None, latest_data_date=TODAY)
    # 90 days starting the day after the stamp date.
    assert window("final", pre, STAMP, latest_data_date=TODAY) == FINAL_WIN
    assert window("final_alone", pre, STAMP, latest_data_date=FINAL_WIN[1])[1] == FINAL_WIN[1]
    with pytest.raises(FinalNotReady, match="2027-01-22"):
        window("final", pre, STAMP, latest_data_date=date(2027, 1, 22))


# ---------------------------------------------------------------- stamp_confirmed

def test_final_needs_committed_stamp_file_flag_only_confirms(repo):
    p = _register(repo)
    with pytest.raises(FinalNotReady, match="only confirm"):
        J.resolve_stamp_confirmed(p, STAMP)  # flag without the file
    assert J.resolve_stamp_confirmed(p, None) == (None, None)
    _stamp(repo, p)
    assert J.resolve_stamp_confirmed(p, None) == (STAMP, "file")
    assert J.resolve_stamp_confirmed(p, STAMP) == (STAMP, "file+flag")
    with pytest.raises(PreregError, match="conflict"):
        J.resolve_stamp_confirmed(p, STAMP + timedelta(days=1))
    _write_bytes(p.with_suffix(".stamp_confirmed"), "2026-10-25\n2026-10-26\n")
    with pytest.raises(PreregError, match="single"):
        J.resolve_stamp_confirmed(p, None)


def test_uncommitted_stamp_file_refused(repo):
    p = _register(repo)
    _write_bytes(p.with_suffix(".stamp_confirmed"), "2026-10-25\n")
    with pytest.raises(PreregMismatch):
        check_integrity(p)


# ---------------------------------------------------------------- final gate, step 1: open-final

def _open(repo: Path, p: Path, commit: bool = True) -> Path:
    marker = J.prepare_final(p)
    if commit:
        _commit(repo, "open final")
    return marker


def test_second_final_open_refused(repo):
    p = _register(repo)
    _stamp(repo, p)
    _commit_test_result(repo, p)
    marker = open_final(NAME, **_gate_kw(repo, p))
    assert marker == repo / "results" / f"{NAME}.final_opened"
    rec = json.loads(marker.read_text())
    assert rec["window"] == FINAL_WIN_JSON
    assert rec["prereg_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert rec["judge_tree_hash"] == _git(repo, "rev-parse", "HEAD:judge")
    # Uncommitted and matching: idempotent. A different window: refused.
    assert open_final(NAME, **_gate_kw(repo, p)) == marker
    with pytest.raises(FinalGateError, match="does not match"):
        open_final(NAME, **_gate_kw(repo, p, window=(date(2026, 10, 27), date(2027, 1, 24))))
    # Once committed, Final is open: opening again is refused.
    _commit(repo, "open final")
    with pytest.raises(FinalGateError, match="already"):
        open_final(NAME, **_gate_kw(repo, p))


def test_final_marker_or_result_ever_committed_refused(repo):
    # Review I3: deleting a committed marker/result does not reopen Final.
    p = _register(repo)
    _stamp(repo, p)
    _commit_test_result(repo, p)
    marker = _open(repo, p)
    marker.unlink()
    _commit(repo, "delete marker")
    with pytest.raises(FinalGateError, match="history"):
        open_final(NAME, **_gate_kw(repo, p))
    with pytest.raises(FinalGateError):
        run(p, "final")


def test_final_gate_refuses_uncommitted_results(repo):
    p = _register(repo)
    _stamp(repo, p)
    _commit_test_result(repo, p)
    _write_bytes(repo / "results" / "other-test.json", "{}\n")
    with pytest.raises(FinalGateError, match="uncommitted"):
        open_final(NAME, **_gate_kw(repo, p))


def test_final_gate_needs_matching_committed_test_pass(repo):
    p = _register(repo)
    _stamp(repo, p)
    with pytest.raises(FinalGateError, match="Test result"):
        open_final(NAME, **_gate_kw(repo, p))  # no Test result at all
    _commit_test_result(repo, p, passed=False)
    with pytest.raises(FinalGateError, match="passed: false"):
        open_final(NAME, **_gate_kw(repo, p))
    _commit_test_result(repo, p, prereg_sha256="f" * 64)
    with pytest.raises(FinalGateError, match="prereg_sha256"):
        open_final(NAME, **_gate_kw(repo, p))
    _commit_test_result(repo, p, judge_tree_hash="e" * 40)
    with pytest.raises(FinalGateError, match="judge_tree_hash"):
        open_final(NAME, **_gate_kw(repo, p))
    assert not (repo / "results" / f"{NAME}.final_opened").exists()
    _commit_test_result(repo, p)
    assert open_final(NAME, **_gate_kw(repo, p)).exists()


def test_final_gate_refuses_tracked_but_edited_test_result(repo):
    p = _register(repo)
    _stamp(repo, p)
    out = _commit_test_result(repo, p, passed=False)
    out.write_text(out.read_text().replace('"passed": false', '"passed": true'))
    with pytest.raises(FinalGateError, match="uncommitted|committed"):
        open_final(NAME, **_gate_kw(repo, p))


def test_final_alone_needs_no_test_result(repo):
    p = _register(repo)
    assert open_final(NAME, final_alone=True, **_gate_kw(repo, p)).exists()


def test_prepare_final_loads_no_data(repo, monkeypatch):
    monkeypatch.setattr(J, "_load_data", lambda *a: pytest.fail("open-final must not load data"))
    p = _register(repo)
    _stamp(repo, p)
    _commit_test_result(repo, p)
    marker = J.prepare_final(p)
    assert json.loads(marker.read_text())["window"] == FINAL_WIN_JSON
    assert not (repo / "results" / f"{NAME}-final.json").exists()


# ---------------------------------------------------------------- final gate, step 2: run needs a committed marker

def test_run_final_needs_committed_single_commit_marker(repo, monkeypatch):
    p = _register(repo)
    _stamp(repo, p)
    _commit_test_result(repo, p)
    monkeypatch.setattr(J, "_load_data", lambda *a: pytest.fail("data must not load"))
    with pytest.raises(FinalGateError, match="open-final"):
        run(p, "final")  # no marker
    marker = _open(repo, p, commit=False)
    with pytest.raises(FinalGateError, match="not committed"):
        run(p, "final")  # marker not committed
    _commit(repo, "open final")
    # A marker touched by a second commit is refused, even if it still matches.
    rec = json.loads(marker.read_text())
    rec["note"] = "edited"
    _write_bytes(marker, json.dumps(rec) + "\n")
    _commit(repo, "edit marker")
    with pytest.raises(FinalGateError, match="exactly one commit"):
        run(p, "final")


def test_run_final_refuses_committed_marker_that_does_not_match(repo, monkeypatch):
    p = _register(repo)
    _stamp(repo, p)
    _commit_test_result(repo, p)
    marker = repo / "results" / f"{NAME}.final_opened"
    _write_bytes(marker, json.dumps({
        "prereg_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "judge_tree_hash": _git(repo, "rev-parse", "HEAD:judge"),
        "window": {"start": "2026-11-01", "end": "2027-01-29"},
    }) + "\n")
    _commit(repo, "hand-made marker")
    monkeypatch.setattr(J, "_load_data", lambda *a: pytest.fail("data must not load"))
    with pytest.raises(FinalGateError, match="does not match"):
        run(p, "final")


def test_final_loads_data_only_after_marker_committed(repo, monkeypatch):
    p = _register(repo)
    _stamp(repo, p)
    _commit_test_result(repo, p)
    _open(repo, p)
    seen = {}

    class Stop(Exception):
        pass

    def loader(market, start, end, cache_dir):
        seen["args"] = (market, start, end)
        raise Stop

    monkeypatch.setattr(J, "_load_data", loader)
    with pytest.raises(Stop):
        run(p, "final")
    assert seen["args"] == ("um", FINAL_WIN[0] - timedelta(days=450), FINAL_WIN[1])


def test_final_resumes_after_crash_with_committed_marker(repo, monkeypatch):
    # Review R3: committing the marker (as the README says) never burns Final.
    p = _register(repo)
    _stamp(repo, p)
    _commit_test_result(repo, p)
    _open(repo, p)

    def crash(*a):
        raise data.ArchiveFetchError("https://x", 429)

    monkeypatch.setattr(J, "_load_data", crash)
    with pytest.raises(data.ArchiveFetchError):
        run(p, "final")
    monkeypatch.setattr(J, "_load_data", _fake_loader(drift=0.003, calls=[]))
    out = run(p, "final", stamp_confirmed=STAMP)
    res = json.loads(out.read_text())
    assert res["kind"] == "final" and res["window"] == FINAL_WIN_JSON
    assert res["scored_days"] == 90
    m = res["manifest"]
    assert m["stamp_confirmed"] == "2026-10-25" and m["stamp_source"] == "file+flag"
    assert m["final_marker"]["path"] == f"results/{NAME}.final_opened"
    assert len(m["final_marker"]["commit"]) == 40
    # Final result exists: no second run; once committed, never again.
    with pytest.raises(J.ResultExists):
        run(p, "final")
    _commit(repo, "final result")
    out.unlink()
    _commit(repo, "delete final result")
    with pytest.raises(FinalGateError, match="history"):
        run(p, "final")


# ---------------------------------------------------------------- stamp rules

def test_final_window_must_start_after_test_window(repo, monkeypatch):
    # Amended I2 (a): stamp + 1 day must be after 2026-10-19.
    monkeypatch.setattr(J, "_load_data", lambda *a: pytest.fail("data must not load"))
    p = _register(repo)
    _commit_test_result(repo, p)
    J.check_stamp(p, date(2026, 10, 20))
    with pytest.raises(FinalGateError, match="after the Test window"):
        J.check_stamp(p, date(2026, 10, 18))  # Final would start 2026-10-19
    _stamp(repo, p, date(2026, 10, 1))
    with pytest.raises(FinalGateError, match="after the Test window"):
        J.prepare_final(p)
    assert not (repo / "results" / f"{NAME}.final_opened").exists()


def test_stamp_not_before_prereg_first_commit(repo):
    # Amended I2 (b): the stamp cannot predate the prereg's first commit (2026-10-20 here).
    p = _register(repo)
    J.check_stamp(p, date(2026, 10, 20))  # same day: allowed
    with pytest.raises(FinalGateError, match="first commit"):
        J.check_stamp(p, date(2026, 10, 19))  # passes (a), fails (b)


def test_stamp_file_single_commit_declared_before_final_start(repo):
    # Review R1: the stamp file is written once, before the Final window starts.
    p = _register(repo)
    _stamp(repo, p)
    J.check_stamp_file(p, FINAL_WIN[0])
    _write_bytes(p.with_suffix(".stamp_confirmed"), "2026-11-20\n")
    _commit(repo, "move stamp")
    with pytest.raises(FinalGateError, match="exactly one commit"):
        J.check_stamp_file(p, date(2026, 11, 21))

    p2 = _register(repo, name="02-late", candidate_id="02-late")
    _write_bytes(p2.with_suffix(".stamp_confirmed"), "2026-11-15\n")
    _commit(repo, "late stamp", when="2027-03-01T12:00:00+00:00")  # written after the window's data existed
    with pytest.raises(FinalGateError, match="on or before"):
        J.check_stamp_file(p2, date(2026, 11, 16))


def test_final_kind_must_match_candidate_type(repo, monkeypatch):
    monkeypatch.setattr(J, "_load_data", lambda *a: pytest.fail("data must not load"))
    p = _register(repo)  # Test window 2025-03-20..2026-10-19: not Final-alone
    _stamp(repo, p)
    with pytest.raises(FinalGateError, match="final_alone"):
        run(p, "final_alone")
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
    close.attrs["listed_symbols"] = cols + ["DEADUSDT"]
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
    _commit_repro_pass(repo, p)
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
    _commit_repro_pass(repo, p)
    monkeypatch.setattr(J, "_load_data", _fake_loader(drift=0.003, calls=[]))
    out = run(p, "test")
    res = json.loads(out.read_text())
    assert res["candidate_id"] == NAME and res["kind"] == "test"
    assert res["window"] == {"start": "2025-03-20", "end": "2026-10-19"}
    # Returns scored from the window's first day to its last, inclusive.
    assert res["scored_days"] == (date(2026, 10, 19) - date(2025, 3, 20)).days + 1
    assert 0.0 <= res["luck_odds"] <= 1.0
    assert set(res["verdict"]) == {"passed", "sharpe", "max_dd", "bench_sharpe", "rules"}
    assert res["passed"] == res["verdict"]["passed"]
    m = res["manifest"]
    blob = subprocess.run(["git", "show", f"HEAD:prereg/{NAME}.json"], cwd=repo, capture_output=True).stdout
    assert m["prereg_sha256"] == hashlib.sha256(blob).hexdigest()
    assert m["judge_tree_hash"] == _git(repo, "rev-parse", "HEAD:judge")
    assert m["seed"] == 7 and m["n_sims"] == 8
    assert m["pip_freeze"] == ["numpy==0.0-test"]
    assert m["archive_checksums"] == {"data.binance.vision/data/futures/um/x.zip": "ab" * 32}
    assert m["api_funding_sha256"] == {"C1USDT": data.funding_rows_sha256("C1USDT", [(1_700_000_000_000, 0.0001)])}
    assert m["data_last_date"] == "2026-10-19"
    assert m["skipped_symbols"] == ["DEADUSDT"]
    assert m["listed_symbols"] == [f"C{i}USDT" for i in range(5)] + ["DEADUSDT"]
    assert m["market"] == "um"
    assert m["window"] == res["window"]
    assert m["stamp_source"] is None and m["final_marker"] is None
    assert m["settings"]["cost_per_side"] == 0.0011 and m["settings"]["gross_cap"] == 1.0
    assert "2025-03-31" in m["universe_by_month"] and isinstance(m["excluded"], dict)
    assert m["results_git_status"] == []
    assert "lookback" in m["luck_odds_method"]
    paths = m["luck_odds_paths"]
    assert [x["lookback"] for x in paths] == [5, 10, 20]
    for x in paths:
        assert x["p_short"] == 0.0  # long-only
        assert x["p_long"] + x["p_flat"] == pytest.approx(1.0)
        assert 0.0 < x["change_rate"] <= x["flip_prob"] <= 1.0
    assert m["run_utc"].endswith("Z")
    # A result is never overwritten.
    with pytest.raises(J.ResultExists):
        run(p, "test")


def test_run_refuses_when_data_short_of_window(repo, monkeypatch):
    p = _register(repo)
    _commit_repro_pass(repo, p)

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
    _commit_test_result(repo, p)
    assert cli_main(["run", str(p), "--kind", "final"]) == 2
    err = capsys.readouterr().err
    assert "stamp_confirmed" in err and ".stamp_confirmed" in err
    assert cli_main(["run", str(p), "--kind", "final", "--stamp-confirmed", "2026-10-25"]) == 2
    assert "only confirm" in capsys.readouterr().err
    assert not (repo / "results" / f"{NAME}.final_opened").exists()


def test_cli_open_final_writes_marker_and_stops(repo, monkeypatch, capsys):
    monkeypatch.setattr(J, "_load_data", lambda *a: pytest.fail("open-final must not load data"))
    p = _register(repo)
    _commit_test_result(repo, p)
    assert cli_main(["open-final", str(p)]) == 2  # no committed stamp file
    assert "stamp_confirmed" in capsys.readouterr().err
    _stamp(repo, p)
    assert cli_main(["open-final", str(p)]) == 0
    out = capsys.readouterr().out
    assert f"{NAME}.final_opened" in out and "commit" in out
    assert (repo / "results" / f"{NAME}.final_opened").exists()


def test_cli_test_run_reports_and_exit_code(repo, monkeypatch, capsys):
    p = _register(repo)
    _commit_repro_pass(repo, p)
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
