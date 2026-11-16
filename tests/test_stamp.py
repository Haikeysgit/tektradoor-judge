"""Task 9a: OpenTimestamps proof check, announcement gate, ots-status."""
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
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
    CALENDAR, FINAL_WIN, NAME, NOW_UTC, POST_TIME, STAMP, FakeExplorer, status_id, status_url, _announce, _block_height, _commit, _commit_repro_pass,
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


def test_later_false_attestation_refused_even_when_earliest_is_real(repo):
    # Review M5: verifying only the earliest attestation would pass this proof.
    p = _register(repo)
    digest = bytes.fromhex(_sha(p))
    ex = S.default_explorer()
    late = date(2026, 10, 28)
    ots_bytes(digest, bitcoin=[(_block_height(STAMP), _noon(STAMP))], explorer=ex)  # real, earliest
    ots_bytes(digest, bitcoin=[(_block_height(late), _noon(late))], explorer=ex, wrong_root=True)  # false, later
    _write_bytes(_ots_path(p), ots_bytes(digest, bitcoin=[(_block_height(STAMP), _noon(STAMP)),
                                                          (_block_height(late), _noon(late))]))
    _commit(repo, "later false attestation")
    with pytest.raises(J.StampError, match=str(_block_height(late))):
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
    assert m["announcement"]["post_time_utc"] == "2026-11-01T01:00:00.000Z"  # review I1 / M1


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
    x_url = status_url(POST_TIME)
    tw_url = f"https://twitter.com/Tek_1/status/{status_id(POST_TIME, seq=5)}"
    for i, url in enumerate([x_url, tw_url]):
        _announce(repo, NAME if i == 0 else "02-other", _sha(p), url=url)
    ann = S.check_announcement(p, _sha(p))
    assert ann["url"] == x_url and ann["post_time_utc"] == "2026-11-01T01:00:00.000Z"


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
    assert "bitcoin (unverified, run open-final to verify)" in lines["02-done"]
    assert str(_block_height(STAMP)) in lines["02-done"]
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


# ---------------------------------------------------------------- review I1: the post time (snowflake)

def test_snowflake_post_time_decoded_offline():
    assert S.snowflake_time(0) == datetime(2010, 11, 4, 1, 42, 54, 657000, tzinfo=timezone.utc)
    sid = ((1_700_000_000_000 - 1288834974657) << 22) + 12345  # low 22 bits: worker and sequence
    assert S.snowflake_time(sid) == datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)
    assert S.snowflake_time(status_id(POST_TIME, seq=7)) == POST_TIME


@pytest.mark.parametrize("post,match", [
    (datetime(2026, 10, 31, 23, 59, tzinfo=timezone.utc), "2026-11-01"),  # (c) before the floor
    (NOW_UTC + timedelta(minutes=1), "after the run"),  # (b) not before the run's start
])
def test_post_time_floor_and_run_start(repo, monkeypatch, post, match):
    _no_data(monkeypatch)
    p = _register(repo, announce=False)
    _commit_repro_pass(repo, p)
    _announce(repo, NAME, _sha(p), post=post)
    with pytest.raises(J.AnnouncementError, match=match):
        J.run(p, "test")


def test_ancient_post_refused(repo, monkeypatch):
    # Review probe P2: a 2006 post id is well formed but not an announcement.
    _no_data(monkeypatch)
    p = _register(repo, announce=False)
    _commit_repro_pass(repo, p)
    _announce(repo, NAME, _sha(p), url="https://x.com/jack/status/20")
    with pytest.raises(J.AnnouncementError, match="2026-11-01"):
        J.run(p, "test")


def test_post_not_before_prereg_first_commit_minus_one_day(repo):
    # (a) Registered 2026-11-05: a post on 11-04 is allowed (one day of slack
    # for local git dates), one on 11-03 is refused.
    when = "2026-11-05T12:00:00+00:00"
    p = _register(repo, announce=False, when=when)
    _announce(repo, NAME, _sha(p), post=datetime(2026, 11, 3, 23, 0, tzinfo=timezone.utc), when=when)
    with pytest.raises(J.AnnouncementError, match="first commit"):
        S.check_announcement(p, _sha(p))
    p2 = _register(repo, name="02-ok", candidate_id="02-ok", announce=False, when=when)
    _announce(repo, "02-ok", _sha(p2), post=datetime(2026, 11, 4, 0, 0, tzinfo=timezone.utc), when=when)
    assert S.check_announcement(p2, _sha(p2))["post_time_utc"] == "2026-11-04T00:00:00.000Z"


def test_ots_must_exist_when_first_announced(repo):
    # C1: some version of the .ots is committed no later than the commit that
    # first added the announcement entry.
    p = _register(repo, stamp_ots=False)
    _write_bytes(_ots_path(p), ots_bytes(bytes.fromhex(_sha(p))))
    _commit(repo, "proof added after the announcement")
    with pytest.raises(J.AnnouncementError, match="ots"):
        S.check_announcement(p, _sha(p))


def test_manifest_records_post_time_for_test(repo, monkeypatch):
    p = _register(repo)
    _commit_repro_pass(repo, p)
    monkeypatch.setattr(J, "_load_data", _fake_loader(drift=0.003, calls=[]))
    m = json.loads(J.run(p, "test").read_text())["manifest"]
    assert m["announcement"]["post_time_utc"] == "2026-11-01T01:00:00.000Z"


# ---------------------------------------------------------------- review C1: the stamp is bound to the post

def test_restamped_proof_after_announcement_refused(repo, monkeypatch):
    # Review probe P1: registered with a pending proof, announced, Test passed;
    # a month later the unchanged prereg is stamped again and committed with a
    # matching stamp_confirmed, to move the Final window. Its block is far
    # outside [post - 7 days, post + 48 h], so open-final refuses.
    _no_data(monkeypatch)
    p = _register(repo)
    _commit_test_result(repo, p)
    restamp = date(2026, 11, 20)
    digest = bytes.fromhex(_sha(p))
    _write_bytes(_ots_path(p), ots_bytes(digest, pending=(), bitcoin=[(_block_height(restamp), _noon(restamp))],
                                         explorer=S.default_explorer()))
    _write_bytes(p.with_suffix(".stamp_confirmed"), f"{restamp.isoformat()}\n")
    _commit(repo, "fresh stamp")
    with pytest.raises(J.StampError, match="48 h"):
        J.prepare_final(p)
    assert not (repo / "results" / f"{NAME}.final_opened").exists()


def test_block_too_long_before_post_refused(repo, monkeypatch):
    # The earliest block must be at most 7 days before the post.
    _no_data(monkeypatch)
    p = _register(repo, announce=False)
    _commit_test_result(repo, p)
    _announce(repo, NAME, _sha(p), post=datetime(2026, 11, 2, 13, 0, tzinfo=timezone.utc))  # block + 8 days
    _stamp(repo, p)
    with pytest.raises(J.StampError, match="7 days"):
        J.prepare_final(p)


@pytest.mark.parametrize("block_day", [
    STAMP,  # block 2026-10-25 12:00 UTC: 6.5 days before the post
    date(2026, 11, 2),  # block 2026-11-02 12:00 UTC: 36 h after the post
])
def test_block_within_post_window_accepted(repo, monkeypatch, block_day):
    _no_data(monkeypatch)
    p = _register(repo, announce=False)
    _commit_test_result(repo, p)
    _announce(repo, NAME, _sha(p), post=datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc))
    _stamp(repo, p, day=block_day)  # stamp_confirmed is the block's date
    assert J.prepare_final(p).exists()


def test_bitcoin_block_after_post_plus_48h_refused(repo, monkeypatch):
    _no_data(monkeypatch)
    p = _register(repo)  # post 2026-11-01 01:00 UTC
    _commit_test_result(repo, p)
    late = date(2026, 11, 3)  # block 2026-11-03 12:00 UTC: post + 59 h
    _stamp(repo, p, day=late)
    with pytest.raises(J.StampError, match="48 h"):
        J.prepare_final(p)


# ---------------------------------------------------------------- review M1: Final re-reads the announcement

def test_final_rechecks_announcement(repo, monkeypatch):
    # Review probe P3: the entry is changed after Test; open-final and run(final) refuse.
    _no_data(monkeypatch)
    p = _register(repo)
    _commit_test_result(repo, p)
    _stamp(repo, p)
    _announce(repo, NAME, "e" * 64, url=status_url(POST_TIME, seq=9))
    with pytest.raises(J.AnnouncementError):
        J.prepare_final(p)
    with pytest.raises(J.AnnouncementError):
        J.run(p, "final")


# ---------------------------------------------------------------- review M2: proof of work

def test_bits_target():
    assert S.bits_target(0x1D00FFFF) == 0xFFFF << 208
    assert S.bits_target(0x1B0404CB) == 0x0404CB << (8 * (0x1B - 3))
    with pytest.raises(ValueError):
        S.bits_target(0x1D80FFFF)  # sign bit set


def test_block_170_header_meets_its_target():
    get, _ = _recorded_get()
    rec = json.loads((FIXTURES / "block170-explorer.json").read_text())
    header = bytes.fromhex(next(v for k, v in rec["responses"].items() if k.endswith("/header")))
    assert S.header_pow_ok(header)
    assert S.BlockstreamExplorer(get=get).block_at(170)["height"] == 170


def _self_consistent_tamper(edit_header):
    """The recorded explorer, serving an edited header together with the
    edited header's own hash, so the hash check passes and only proof of work
    can refuse it."""
    rec = json.loads((FIXTURES / "block170-explorer.json").read_text())
    real_hash = rec["responses"]["/block-height/170"]
    header = bytes.fromhex(rec["responses"][f"/block/{real_hash}/header"])
    forged = edit_header(header)
    forged_hash = S._sha256d(forged)[::-1].hex()
    responses = {"/block-height/170": forged_hash, f"/block/{forged_hash}/header": forged.hex()}

    def get(url, timeout=None):
        return _Resp(responses[url[len(rec["base"]):]])

    return get


@pytest.mark.parametrize("edit", [
    # another merkle root: the hash no longer meets the target
    lambda h: h[:36] + hashlib.sha256(b"forged").digest() + h[68:],
    # nBits lowered to a harder target than the header's work
    lambda h: h[:72] + (0x1C00FFFF).to_bytes(4, "little") + h[76:],
    # nBits raised above the proof-of-work limit
    lambda h: h[:72] + (0x2100FFFF).to_bytes(4, "little") + h[76:],
])
def test_header_without_proof_of_work_refused(edit):
    with pytest.raises(J.StampError, match="proof of work"):
        S.BlockstreamExplorer(get=_self_consistent_tamper(edit)).block_at(170)
