"""Task 9a: OpenTimestamps proof check, announcement gate, ots-status."""
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from opentimestamps.core.notary import BitcoinBlockHeaderAttestation
from opentimestamps.core.op import OpPrepend, OpSHA256
from opentimestamps.core.serialize import BytesSerializationContext
from opentimestamps.core.timestamp import DetachedTimestampFile, Timestamp

from judge import judge as J
from judge import stamp as S
from judge.__main__ import main as cli_main
from test_prereg import (  # noqa: F401
    CALENDAR, FINAL_WIN, NAME, STAMP, FakeExplorer, _announce, _block_height, _commit, _commit_repro_pass,
    _commit_test_result, _fake_loader, _git, _noon, _ots_path, _register, _stamp, _upgrade_ots, _write_bytes,
    ots_bytes, repo,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ots"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _no_data(monkeypatch):
    monkeypatch.setattr(J, "_load_data", lambda *a: pytest.fail("data must not load"))


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- real block fixture (byte order pin)

class _Resp:
    def __init__(self, payload):
        self.status_code = 200
        self.text = payload if isinstance(payload, str) else json.dumps(payload)
        self._payload = payload

    def json(self):
        return self._payload


def _recorded_get():
    rec = json.loads((FIXTURES / "block170-explorer.json").read_text())
    calls = []

    def get(url, timeout=None):
        calls.append(url)
        path = url[len(rec["base"]):]
        if path not in rec["responses"]:
            raise AssertionError(f"unrecorded explorer call {url}")
        return _Resp(rec["responses"][path])

    return get, calls


def test_real_block_170_proof_verifies_through_blockstream_explorer():
    # The proof stamps the raw bytes of tx f4184fc5...9e16 (block 170) and
    # commits to block 170's real merkle root. Blockstream serves merkle_root
    # in display (reversed) byte order; the proof's message is internal order.
    tx = (FIXTURES / "block170-tx.bin").read_bytes()
    blob = (FIXTURES / "block170-tx.bin.ots").read_bytes()
    get, calls = _recorded_get()
    explorer = S.BlockstreamExplorer(get=get)
    info = S.verify_proof(blob, hashlib.sha256(tx).hexdigest(), require_bitcoin=True, explorer=explorer, name="block170")
    assert info["block_height"] == 170
    assert info["block_hash"] == "00000000d1145790a8694403d4063f323d499e655c83426834d4ce2f8dd4a2ee"
    assert info["block_time_utc"] == "2009-01-12T03:30:25Z"
    assert info["verified"] is True and info["bitcoin_heights"] == [170] and info["pending_calendars"] == []
    assert calls == [
        "https://blockstream.info/api/block-height/170",
        "https://blockstream.info/api/block/00000000d1145790a8694403d4063f323d499e655c83426834d4ce2f8dd4a2ee/header",
    ]


def test_block_170_header_fields_match_the_explorer_json():
    # The merkle root and time come from the 80 header bytes; the explorer's
    # own JSON (recorded for reference only) agrees with them.
    get, _ = _recorded_get()
    blk = S.BlockstreamExplorer(get=get).block_at(170)
    ref = json.loads((FIXTURES / "block170-explorer.json").read_text())["reference_block_json"]
    assert blk["hash"] == ref["id"]
    assert blk["merkle_root"] == ref["merkle_root"]  # display order, reversed from the header bytes
    assert blk["timestamp"] == ref["timestamp"] == 1231731025
    assert blk["height"] == 170


def test_real_block_170_fixture_is_the_documented_merkle_path():
    # Rebuild the committed proof from the tx bytes and block 170's coinbase txid.
    tx = (FIXTURES / "block170-tx.bin").read_bytes()
    ts = Timestamp(hashlib.sha256(tx).digest())
    t = ts.ops.add(OpSHA256())
    assert t.msg[::-1].hex() == "f4184fc596403b9d638783cf57adfe4c75c605f6356fbc91338530e9831e9e16"  # the txid
    coinbase = bytes.fromhex("b1fea52486ce0c62bb442b530a3f0132b826c74e473d1f2c220bfa78111c5082")[::-1]
    t = t.ops.add(OpPrepend(coinbase)).ops.add(OpSHA256()).ops.add(OpSHA256())
    t.attestations.add(BitcoinBlockHeaderAttestation(170))
    ctx = BytesSerializationContext()
    DetachedTimestampFile(OpSHA256(), ts).serialize(ctx)
    assert ctx.getbytes() == (FIXTURES / "block170-tx.bin.ots").read_bytes()


def test_unreversed_merkle_root_is_refused():
    # An explorer (or a bug) comparing internal byte order to display order fails.
    tx = (FIXTURES / "block170-tx.bin").read_bytes()
    blob = (FIXTURES / "block170-tx.bin.ots").read_bytes()
    root = "7dac2c5666815c17a3b36427de37bb9d2e2c5ccec3f8633eb91a4205cb4c10ff"
    ex = FakeExplorer()
    ex.blocks[170] = {"height": 170, "hash": "00" * 32, "merkle_root": bytes.fromhex(root)[::-1].hex(), "timestamp": 1231731025}
    with pytest.raises(J.StampError, match="merkle"):
        S.verify_proof(blob, hashlib.sha256(tx).hexdigest(), require_bitcoin=True, explorer=ex, name="block170")


def _tampered(edit):
    """The recorded explorer, with the header response passed through edit(hex)."""
    get, _ = _recorded_get()

    def tampered(url, timeout=None):
        r = get(url, timeout)
        return _Resp(edit(r.text)) if url.endswith("/header") else r

    return tampered


def test_blockstream_explorer_refuses_header_not_hashing_to_block_hash():
    # A header whose merkle root was swapped no longer hashes to the block hash.
    other_root = hashlib.sha256(b"other").digest().hex()
    with pytest.raises(J.StampError, match="does not hash to"):
        S.BlockstreamExplorer(get=_tampered(lambda h: h[:72] + other_root + h[136:])).block_at(170)
    # So does a header with its time moved.
    with pytest.raises(J.StampError, match="does not hash to"):
        S.BlockstreamExplorer(get=_tampered(lambda h: h[:136] + "00000000" + h[144:])).block_at(170)


@pytest.mark.parametrize("edit", [lambda h: h[:-2], lambda h: h + "00", lambda h: "zz" + h[2:], lambda h: ""])
def test_blockstream_explorer_refuses_malformed_header(edit):
    with pytest.raises(J.StampError, match="80-byte header"):
        S.BlockstreamExplorer(get=_tampered(edit)).block_at(170)

    def http_error(url, timeout=None):
        r = _Resp("not found")
        r.status_code = 404
        return r

    with pytest.raises(J.StampError, match="404"):
        S.BlockstreamExplorer(get=http_error).block_at(170)


# ---------------------------------------------------------------- check_ots: digest, pending, bitcoin

def test_ots_digest_mismatch_refused(repo):
    p = _register(repo)
    _write_bytes(_ots_path(p), ots_bytes(hashlib.sha256(b"some other file").digest()))
    _commit(repo, "wrong proof")
    with pytest.raises(J.StampError) as ei:
        S.check_ots(p, require_bitcoin=False)
    msg = str(ei.value)
    assert _sha(p) in msg and hashlib.sha256(b"some other file").hexdigest() in msg


def test_ots_missing_uncommitted_or_garbage_refused(repo):
    p = _register(repo, stamp_ots=False)
    with pytest.raises(J.StampError, match="ots stamp"):
        S.check_ots(p, require_bitcoin=False)
    _write_bytes(_ots_path(p), ots_bytes(bytes.fromhex(_sha(p))))
    with pytest.raises(J.StampError, match="not committed"):
        S.check_ots(p, require_bitcoin=False)
    _write_bytes(_ots_path(p), b"not a proof")
    _commit(repo, "garbage proof")
    with pytest.raises(J.StampError, match="not a valid OpenTimestamps proof"):
        S.check_ots(p, require_bitcoin=False)


def test_pending_ok_for_test_refused_for_final(repo):
    p = _register(repo)
    info = S.check_ots(p, require_bitcoin=False)
    assert info["pending_calendars"] == [CALENDAR] and info["bitcoin_heights"] == []
    assert info["block_height"] is None and info["block_time_utc"] is None and info["verified"] is False
    assert info["file"] == f"prereg/{NAME}.json.ots" and info["digest_sha256"] == _sha(p)
    assert info["ots_sha256"] == _sha(_ots_path(p))
    with pytest.raises(J.StampError, match="no Bitcoin attestation"):
        S.check_ots(p, require_bitcoin=True)


def test_bitcoin_attestation_verified_against_mocked_explorer(repo):
    p = _register(repo)
    _upgrade_ots(p, STAMP)
    _commit(repo, "upgrade proof")
    ex = S.default_explorer()
    info = S.check_ots(p, require_bitcoin=True)
    assert info["block_height"] == _block_height(STAMP)
    assert info["block_time_utc"] == f"{STAMP.isoformat()}T12:00:00Z"
    assert info["verified"] is True and ex.calls == [_block_height(STAMP)]
    # The block at that height has another merkle root: refused.
    ex.blocks[_block_height(STAMP)]["merkle_root"] = hashlib.sha256(b"other").hexdigest()
    with pytest.raises(J.StampError, match="merkle"):
        S.check_ots(p, require_bitcoin=True)


def test_ots_may_be_upgraded_in_later_commits(repo):
    # Unlike the .json and .sha256, the .ots is exempt from the one-commit rule.
    p = _register(repo)
    _upgrade_ots(p, STAMP)
    _commit(repo, "chore: upgrade OpenTimestamps proofs")
    J.check_integrity(p)
    assert S.check_ots(p, require_bitcoin=False)["bitcoin_heights"] == [_block_height(STAMP)]


def test_earliest_of_several_attestations_is_used(repo):
    p = _register(repo)
    later = date(2026, 10, 27)
    _upgrade_ots(p, later, STAMP)
    _commit(repo, "two attestations")
    info = S.check_ots(p, require_bitcoin=True)
    assert info["block_height"] == _block_height(STAMP)
    assert info["bitcoin_heights"] == [_block_height(STAMP), _block_height(later)]


def test_any_false_bitcoin_attestation_refused(repo):
    p = _register(repo)
    digest = bytes.fromhex(_sha(p))
    ex = S.default_explorer()
    # Register a true block for STAMP, and a block for an earlier attestation
    # whose merkle root does not match it.
    early = date(2026, 10, 21)
    ots_bytes(digest, bitcoin=[(_block_height(STAMP), _noon(STAMP))], explorer=ex)
    ots_bytes(digest, bitcoin=[(_block_height(early), _noon(early))], explorer=ex, wrong_root=True)
    _write_bytes(_ots_path(p), ots_bytes(digest, bitcoin=[(_block_height(STAMP), _noon(STAMP)),
                                                          (_block_height(early), _noon(early))]))
    _commit(repo, "one false attestation")
    with pytest.raises(J.StampError, match=str(_block_height(early))):
        S.check_ots(p, require_bitcoin=True)


# ---------------------------------------------------------------- Final: stamp_confirmed vs block date

def test_stamp_confirmed_must_equal_block_date(repo, monkeypatch):
    _no_data(monkeypatch)
    p = _register(repo)
    _commit_test_result(repo, p)
    _stamp(repo, p, day=STAMP, block_day=date(2026, 10, 24))
    with pytest.raises(J.StampError) as ei:
        J.prepare_final(p)
    assert "2026-10-24" in str(ei.value) and "2026-10-25" in str(ei.value)
    with pytest.raises(J.StampError):
        J.run(p, "final")
    assert not (repo / "results" / f"{NAME}.final_opened").exists()


def test_final_refuses_pending_only_proof(repo, monkeypatch):
    _no_data(monkeypatch)
    p = _register(repo)
    _commit_test_result(repo, p)
    _write_bytes(p.with_suffix(".stamp_confirmed"), f"{STAMP.isoformat()}\n")
    _commit(repo, "stamp file, proof still pending")
    with pytest.raises(J.StampError, match="no Bitcoin attestation"):
        J.prepare_final(p)


def test_final_records_verified_ots_in_manifest(repo, monkeypatch):
    p = _register(repo)
    _stamp(repo, p)
    _commit_test_result(repo, p)
    J.prepare_final(p)
    _commit(repo, "open final")
    monkeypatch.setattr(J, "_load_data", _fake_loader(drift=0.003, calls=[]))
    m = json.loads(J.run(p, "final").read_text())["manifest"]
    assert m["ots"]["block_height"] == _block_height(STAMP)
    assert m["ots"]["block_time_utc"] == "2026-10-25T12:00:00Z"
    assert m["ots"]["verified"] is True and m["ots"]["explorer"] == "fake://explorer"
    assert m["stamp_confirmed"] == "2026-10-25"


# ---------------------------------------------------------------- announcement gate

def test_test_run_records_ots_and_announcement(repo, monkeypatch):
    p = _register(repo)
    _commit_repro_pass(repo, p)
    monkeypatch.setattr(J, "_load_data", _fake_loader(drift=0.003, calls=[]))
    m = json.loads(J.run(p, "test").read_text())["manifest"]
    ann = json.loads((repo / "prereg" / "announced.json").read_text())[NAME]
    assert m["announcement"]["url"] == ann["url"] and m["announcement"]["sha256"] == _sha(p)
    assert m["announcement"]["file"] == "prereg/announced.json" and len(m["announcement"]["commit"]) == 40
    assert m["ots"]["pending_calendars"] == [CALENDAR] and m["ots"]["verified"] is False


def test_announcement_missing_refused(repo, monkeypatch):
    _no_data(monkeypatch)
    p = _register(repo, announce=False)
    _commit_repro_pass(repo, p)
    with pytest.raises(J.AnnouncementError, match="announced.json"):
        J.run(p, "test")
    _announce(repo, "02-someone-else", "a" * 64)
    with pytest.raises(J.AnnouncementError, match=NAME):
        J.run(p, "test")


def test_announcement_wrong_sha_refused(repo, monkeypatch):
    _no_data(monkeypatch)
    p = _register(repo, announce=False)
    _commit_repro_pass(repo, p)
    _announce(repo, NAME, "b" * 64)
    with pytest.raises(J.AnnouncementError) as ei:
        J.run(p, "test")
    assert "b" * 64 in str(ei.value) and _sha(p) in str(ei.value)


@pytest.mark.parametrize("url", [
    "http://x.com/tektradoor/status/123",
    "https://x.com.evil.example/tektradoor/status/123",
    "https://x.com/tektradoor/status/123/photo/1",
    "https://x.com/tektradoor/status/abc",
    "https://x.com/tekt-radoor/status/123",
    "https://mobile.twitter.com/tektradoor/status/123",
    "https://x.com/tektradoor/status/123\n",
    "https://x.com/tektradoor/status/١٢٣",  # non-ASCII digits
    "",
])
def test_announcement_bad_url_refused(repo, monkeypatch, url):
    _no_data(monkeypatch)
    p = _register(repo, announce=False)
    _commit_repro_pass(repo, p)
    _announce(repo, NAME, _sha(p), url=url)
    with pytest.raises(J.AnnouncementError, match="url"):
        J.run(p, "test")


def test_announcement_good_urls_accepted(repo):
    p = _register(repo, announce=False)
    for i, url in enumerate(["https://x.com/tektradoor/status/1849", "https://twitter.com/Tek_1/status/1"]):
        _announce(repo, NAME if i == 0 else "02-other", _sha(p), url=url)
    assert S.check_announcement(p, _sha(p))["url"] == "https://x.com/tektradoor/status/1849"


def test_announcement_entry_edited_in_later_commit_refused(repo, monkeypatch):
    _no_data(monkeypatch)
    p = _register(repo)
    _commit_repro_pass(repo, p)
    ann = repo / "prereg" / "announced.json"
    original = json.loads(ann.read_text())[NAME]
    _announce(repo, NAME, _sha(p), url="https://x.com/tektradoor/status/999")
    with pytest.raises(J.AnnouncementError, match="changed"):
        J.run(p, "test")
    # Changing it back does not help: history still holds the other value.
    _announce(repo, NAME, original["sha256"], url=original["url"])
    with pytest.raises(J.AnnouncementError, match="changed"):
        J.run(p, "test")


def test_announcement_entries_may_be_added_over_time(repo):
    p = _register(repo)
    _announce(repo, "02-next", "c" * 64)
    _announce(repo, "03-later", "d" * 64)
    assert S.check_announcement(p, _sha(p))["sha256"] == _sha(p)


def test_announcement_uncommitted_or_duplicate_key_refused(repo):
    p = _register(repo)
    ann = repo / "prereg" / "announced.json"
    _announce(repo, "02-next", "c" * 64, commit=False)
    with pytest.raises(J.AnnouncementError, match="commit"):
        S.check_announcement(p, _sha(p))
    entry = json.dumps({"sha256": _sha(p), "url": "https://x.com/a/status/1"})
    _write_bytes(ann, f'{{"{NAME}": {entry}, "{NAME}": {entry}}}\n')
    _commit(repo, "duplicate key")
    with pytest.raises(J.AnnouncementError, match="duplicate"):
        S.check_announcement(p, _sha(p))


def test_final_alone_needs_announcement(repo, monkeypatch):
    _no_data(monkeypatch)
    p = _register(repo, announce=False, paper_sample_end="2026-01-31")  # Test window under 12 months
    _commit_repro_pass(repo, p)
    _stamp(repo, p)
    with pytest.raises(J.AnnouncementError):
        J.prepare_final(p)
    with pytest.raises(J.AnnouncementError):
        J.run(p, "final_alone")
    _announce(repo, NAME, _sha(p))
    assert J.prepare_final(p).exists()


# ---------------------------------------------------------------- ots-status

def test_ots_status_reports_each_prereg(repo, monkeypatch, capsys):
    monkeypatch.setattr(S, "_now_utc", lambda: datetime(2026, 10, 22, 0, 0, tzinfo=timezone.utc))
    p1 = _register(repo)  # committed 2026-10-20T12:00Z: 36 h old, pending
    p2 = _register(repo, name="02-done", candidate_id="02-done")
    _upgrade_ots(p2, STAMP)
    _commit(repo, "upgrade 02")
    _register(repo, name="03-unstamped", candidate_id="03-unstamped", stamp_ots=False)
    assert cli_main(["ots-status"]) == 1  # a prereg without a proof is always a failure
    out = capsys.readouterr().out
    lines = {ln.split()[0]: ln for ln in out.splitlines() if ln.strip() and ln.split()[0][:2].isdigit()}
    assert "pending" in lines[p1.stem] and CALENDAR in lines[p1.stem] and "36.0h" in lines[p1.stem]
    assert "bitcoin" in lines["02-done"] and str(_block_height(STAMP)) in lines["02-done"]
    assert "missing" in lines["03-unstamped"]


def test_ots_status_fails_on_stale_pending(repo, monkeypatch, capsys):
    p = _register(repo)
    monkeypatch.setattr(S, "_now_utc", lambda: datetime(2026, 10, 22, 11, 0, tzinfo=timezone.utc))  # 47 h
    assert cli_main(["ots-status", "--fail-pending-hours", "48"]) == 0
    monkeypatch.setattr(S, "_now_utc", lambda: datetime(2026, 10, 22, 13, 0, tzinfo=timezone.utc))  # 49 h
    assert cli_main(["ots-status"]) == 0  # no threshold given: report only
    assert cli_main(["ots-status", "--fail-pending-hours", "48"]) == 1
    assert "STALE" in capsys.readouterr().out
    _upgrade_ots(p, STAMP)
    _commit(repo, "upgrade")
    assert cli_main(["ots-status", "--fail-pending-hours", "48"]) == 0


def test_ots_status_flags_digest_mismatch(repo, capsys):
    p = _register(repo)
    _write_bytes(_ots_path(p), ots_bytes(hashlib.sha256(b"x").digest()))
    _commit(repo, "wrong proof")
    assert cli_main(["ots-status"]) == 1
    assert "digest-mismatch" in capsys.readouterr().out


def test_ots_status_with_no_preregs(repo, capsys):
    assert cli_main(["ots-status", "--fail-pending-hours", "48"]) == 0
    assert "no registered prereg" in capsys.readouterr().out


# ---------------------------------------------------------------- workflow

def test_ots_workflow_shape():
    # Status only: the founder is the only author, so the workflow never
    # commits or pushes. Upgrades are tried on a scratch copy and reported.
    wf = (REPO_ROOT / ".github" / "workflows" / "ots.yml").read_text()
    for needle in ("schedule:", "cron:", "workflow_dispatch:", "contents: read", "fetch-depth: 0",
                   'python-version: "3.11"', "pip install --require-hashes -r requirements.txt",
                   "ots upgrade", "RUNNER_TEMP", "::notice", "python -m judge ots-status --fail-pending-hours 48"):
        assert needle in wf, needle
    for banned in ("contents: write", "git commit", "git push", "git add", "github-actions[bot]"):
        assert banned not in wf, banned
