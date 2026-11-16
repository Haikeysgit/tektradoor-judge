"""The prereg's OpenTimestamps proof and its public announcement.

check_ots: prereg/NN-name.json.ots must be committed (it may be recommitted:
``ots upgrade`` rewrites it), stamp the committed prereg's sha256, and carry
an attestation. Test accepts a pending calendar attestation. Final needs a
Bitcoin block header attestation, checked against a public explorer: the
merkle root the proof commits to at that height must equal the block's.

check_announcement: prereg/announced.json maps candidate_id to the sha256
and the public post (x.com or twitter.com status URL) that announced it. An
entry, once committed, never changes: every version of the file in git
history must hold the same value for it. The post's time is decoded offline
from the status id (a Twitter snowflake) and must lie between the prereg's
registration and the run. Final binds the Bitcoin stamp to that post time.

ots_status: each prereg's proof state, for ``python -m judge ots-status``.

No local Bitcoin node is used, and ``ots verify`` is never called. This
module imports judge.judge only inside its functions (judge.judge imports it).
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from opentimestamps.core.notary import BitcoinBlockHeaderAttestation, PendingAttestation
from opentimestamps.core.op import OpSHA256
from opentimestamps.core.serialize import BytesDeserializationContext, DeserializationError
from opentimestamps.core.timestamp import DetachedTimestampFile

BLOCKSTREAM_API = "https://blockstream.info/api"
ANNOUNCED_FILE = "announced.json"
# ^https://(x|twitter)\.com/[A-Za-z0-9_]+/status/\d+$ with ASCII digits only,
# applied with fullmatch (so no trailing newline slips past "$").
ANNOUNCE_URL_RE = re.compile(r"https://(x|twitter)\.com/[A-Za-z0-9_]+/status/[0-9]+")
SHA256_RE = re.compile(r"[0-9a-f]{64}")

# X/Twitter status ids are snowflakes: ms since this epoch, shifted left 22 bits.
SNOWFLAKE_EPOCH_MS = 1288834974657
# A post announcing a prereg cannot be older than this (Task 9a ruling I1).
ANNOUNCE_FLOOR = datetime(2026, 11, 1, tzinfo=timezone.utc)
# Slack for the prereg's first commit date (a weak, local git claim).
POST_COMMIT_SLACK = timedelta(days=1)
# Final: the earliest Bitcoin block of the stamp must lie in
# [post_time - STAMP_BEFORE_POST, post_time + STAMP_AFTER_POST].
STAMP_BEFORE_POST = timedelta(days=7)
STAMP_AFTER_POST = timedelta(hours=48)
# Bitcoin's proof-of-work limit (the target of nBits 0x1d00ffff).
POW_LIMIT = 0xFFFF << 208
# Final attestations must be in blocks at or above this height, and their
# headers must carry a target at most FLOOR_TARGET (difficulty >= 1e13), so a
# forged header needs real mining work, not a laptop.
MIN_ATTESTATION_HEIGHT = 840_000
FLOOR_DIFFICULTY = 10**13
FLOOR_TARGET = POW_LIMIT // FLOOR_DIFFICULTY


def _J():
    from judge import judge as J
    return J


def _utc_iso(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_utc() -> datetime:
    """Now (UTC): the run's start time for the announcement check, and ots-status's clock. Tests replace this."""
    return datetime.now(timezone.utc)


def snowflake_time(status_id: int) -> datetime:
    """Creation time (UTC) of an X/Twitter status id: (id >> 22) + 1288834974657 ms."""
    ms = (int(status_id) >> 22) + SNOWFLAKE_EPOCH_MS
    return datetime.fromtimestamp(ms // 1000, tz=timezone.utc) + timedelta(milliseconds=ms % 1000)


def utc_ms_iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def parse_utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def bits_target(bits: int) -> int:
    """The proof-of-work target encoded in a header's nBits (compact form)."""
    exponent, mantissa = bits >> 24, bits & 0x007FFFFF
    if bits & 0x00800000:
        raise ValueError(f"nBits {bits:#010x} has the sign bit set")
    return mantissa >> (8 * (3 - exponent)) if exponent <= 3 else mantissa << (8 * (exponent - 3))


def header_pow_ok(header: bytes) -> bool:
    """The 80-byte header's double sha256 (as a little-endian number) is at
    most its nBits target, and that target is at most Bitcoin's limit."""
    try:
        target = bits_target(int.from_bytes(header[72:76], "little"))
    except ValueError:
        return False
    return 0 < target <= POW_LIMIT and int.from_bytes(_sha256d(header), "little") <= target


def ots_path(prereg_path: Path) -> Path:
    """prereg/NN-name.json -> prereg/NN-name.json.ots (what ``ots stamp`` writes)."""
    p = Path(prereg_path)
    return p.with_name(p.name + ".ots")


# ---------------------------------------------------------------- explorer

HEADER_RE = re.compile(r"[0-9a-f]{160}")  # 80 bytes, hex


def _sha256d(b: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()


class BlockstreamExplorer:
    """Block lookups by height from the Blockstream Esplora API.

    block_at(h): GET /block-height/{h} gives the block hash; GET
    /block/{hash}/header gives the raw 80-byte header (hex). The header must
    hash to that block hash (double sha256, reversed to display order), so
    the explorer cannot hand back fields that disagree with the block it
    names. The merkle root (bytes 36..68, internal order) and time (bytes
    68..72, little-endian unix seconds) are read from the header bytes, not
    from the explorer's JSON. The trust left in the explorer is only that
    the hash it gives for height h is the main chain's block at h.
    ``get`` is injectable (requests.get by default) so tests stay offline.
    """

    def __init__(self, base: str = BLOCKSTREAM_API, get=None, timeout: float = 30.0):
        self.base = base.rstrip("/")
        self._get_fn = get
        self.timeout = timeout

    def _get(self, path: str):
        J = _J()
        url = self.base + path
        get = self._get_fn
        if get is None:
            import requests
            get = requests.get
        try:
            resp = get(url, timeout=self.timeout)
        except Exception as exc:  # network errors: the proof cannot be checked now
            raise J.StampError(f"cannot reach the block explorer ({url}): {exc}") from exc
        if resp.status_code != 200:
            raise J.StampError(f"block explorer {url} returned HTTP {resp.status_code}")
        return resp

    def block_at(self, height: int) -> dict:
        """{height, hash, merkle_root (hex, display order), timestamp (unix), header (hex)}."""
        J = _J()
        block_hash = self._get(f"/block-height/{int(height)}").text.strip()
        if not SHA256_RE.fullmatch(block_hash):
            raise J.StampError(f"block explorer gave no block hash for height {height}: {block_hash[:80]!r}")
        header_hex = self._get(f"/block/{block_hash}/header").text.strip()
        if not HEADER_RE.fullmatch(header_hex):
            raise J.StampError(
                f"block explorer gave no 80-byte header for block {block_hash} (height {height}): {header_hex[:200]!r}"
            )
        header = bytes.fromhex(header_hex)
        hashed = _sha256d(header)[::-1].hex()
        if hashed != block_hash:
            raise J.StampError(
                f"block explorer's header for height {height} does not hash to block {block_hash} (it hashes to {hashed})"
            )
        bits = int.from_bytes(header[72:76], "little")
        if int(height) >= MIN_ATTESTATION_HEIGHT:
            try:
                easy = bits_target(bits) > FLOOR_TARGET
            except ValueError:
                easy = True
            if easy:
                raise J.StampError(
                    f"block explorer's header for height {height} (block {block_hash}) has nBits {bits:#010x}, "
                    f"a difficulty below the {FLOOR_DIFFICULTY:.0e} floor for heights >= {MIN_ATTESTATION_HEIGHT}"
                )
        if not header_pow_ok(header):
            raise J.StampError(
                f"block explorer's header for height {height} (block {block_hash}) fails proof of work: "
                f"its hash is above the target in nBits {int.from_bytes(header[72:76], 'little'):#010x}, "
                "or that target is above Bitcoin's limit"
            )
        return {
            "height": int(height),
            "hash": block_hash,
            "merkle_root": header[36:68][::-1].hex(),
            "timestamp": int.from_bytes(header[68:72], "little"),
            "header": header_hex,
        }


def default_explorer() -> BlockstreamExplorer:
    """The explorer used when none is passed. Tests replace this."""
    return BlockstreamExplorer()


# ---------------------------------------------------------------- the proof

def parse_proof(blob: bytes, name: str) -> DetachedTimestampFile:
    """Deserialize a detached .ots proof; StampError if it is not one."""
    J = _J()
    try:
        return DetachedTimestampFile.deserialize(BytesDeserializationContext(blob))
    except (DeserializationError, ValueError, TypeError) as exc:
        raise J.StampError(f"{name} is not a valid OpenTimestamps proof: {exc}") from exc


def attestations(dtf: DetachedTimestampFile) -> tuple[list[str], list[tuple[int, bytes]]]:
    """(pending calendar URIs, [(block height, committed merkle root, internal byte order)]).

    The library computes each message by replaying the proof's operations
    from the file digest, so a Bitcoin attestation's message is the merkle
    root that path commits to. Other attestation types are ignored.
    """
    pending, bitcoin = set(), set()
    for msg, att in dtf.timestamp.all_attestations():
        if isinstance(att, PendingAttestation):
            pending.add(att.uri)
        elif isinstance(att, BitcoinBlockHeaderAttestation):
            bitcoin.add((int(att.height), bytes(msg)))
    return sorted(pending), sorted(bitcoin)


def _verify_bitcoin(height: int, root: bytes, explorer, name: str) -> dict:
    J = _J()
    if len(root) != 32:
        raise J.StampError(f"{name}: Bitcoin attestation at height {height} commits to a {len(root)}-byte message, not a merkle root")
    blk = explorer.block_at(height)
    if blk.get("height") != height:
        raise J.StampError(f"{name}: explorer returned block height {blk.get('height')!r} for height {height}")
    want = root[::-1].hex()  # internal byte order -> display order, as explorers print it
    got = str(blk.get("merkle_root") or "").lower()
    if got != want:
        raise J.StampError(
            f"{name}: Bitcoin attestation at height {height} commits to merkle root {want}, "
            f"but block {blk.get('hash')} has merkle_root {got or '(none)'}"
        )
    ts = blk.get("timestamp")
    if not isinstance(ts, int) or isinstance(ts, bool):
        raise J.StampError(f"{name}: explorer gave no header timestamp for block {height}")
    return {"block_height": height, "block_hash": blk.get("hash"), "block_time_utc": _utc_iso(ts)}


def verify_proof(
    blob: bytes, expected_sha256: str, *, require_bitcoin: bool, explorer=None, name: str = "proof",
    min_height: int | None = MIN_ATTESTATION_HEIGHT,
) -> dict:
    """Check a .ots proof's bytes against the sha256 it must stamp.

    The proof must be a sha256 detached proof of ``expected_sha256``.
    require_bitcoin=False: at least one attestation (a pending calendar one is
    enough); nothing is looked up, so block_* stay None and verified is False.
    require_bitcoin=True: at least one Bitcoin attestation, and every one must
    match its block's merkle root at the explorer (a proof with a false claim
    is refused). The earliest (lowest height) gives block_height and
    block_time_utc (the header time, UTC). Every Bitcoin attestation must be
    at height >= min_height (MIN_ATTESTATION_HEIGHT; None turns the floor
    off, for tests of old real blocks only), checked before any lookup.
    """
    J = _J()
    dtf = parse_proof(blob, name)
    if not isinstance(dtf.file_hash_op, OpSHA256):
        raise J.StampError(f"{name} stamps a {type(dtf.file_hash_op).__name__} digest; it must be an sha256 proof")
    digest = dtf.file_digest.hex()
    if digest != expected_sha256:
        raise J.StampError(f"{name} stamps sha256 {digest}, but the committed prereg has sha256 {expected_sha256}")
    pending, bitcoin = attestations(dtf)
    info = {
        "ots_sha256": hashlib.sha256(blob).hexdigest(),
        "digest_sha256": digest,
        "pending_calendars": pending,
        "bitcoin_heights": sorted({h for h, _ in bitcoin}),
        "block_height": None,
        "block_hash": None,
        "block_time_utc": None,
        "verified": False,
        "explorer": None,
    }
    if not pending and not bitcoin:
        raise J.StampError(f"{name} has no attestation at all; run `ots stamp` again")
    if not require_bitcoin:
        return info
    if not bitcoin:
        raise J.StampError(
            f"{name} has no Bitcoin attestation yet (pending at {', '.join(pending)}); "
            "once the calendars have one (the daily ots workflow posts a notice), run `ots upgrade` "
            "locally and commit the upgraded proof through a PR"
        )
    low = sorted({h for h, _ in bitcoin if min_height is not None and h < min_height})
    if low:
        raise J.StampError(
            f"{name} has Bitcoin attestation(s) at height {', '.join(map(str, low))}, below {min_height}: "
            "Final needs attestations in modern blocks"
        )
    explorer = explorer if explorer is not None else default_explorer()
    checked = [_verify_bitcoin(h, root, explorer, name) for h, root in bitcoin]
    earliest = min(checked, key=lambda c: c["block_height"])
    info.update(earliest, verified=True, explorer=getattr(explorer, "base", type(explorer).__name__))
    return info


def check_ots(prereg_path: Path, *, require_bitcoin: bool, explorer=None) -> dict:
    """Verify prereg/NN-name.json.ots for the committed prereg (see verify_proof).

    The .ots must be committed with working bytes equal to HEAD. It is exempt
    from the one-commit rule, because ``ots upgrade`` rewrites it. Returns
    verify_proof's dict plus "file" (repo-relative path), for the manifest.
    """
    J = _J()
    path = Path(prereg_path).resolve()
    root = J._repo_root(path.parent)
    ots = ots_path(path)
    rel = ots.relative_to(root).as_posix()
    if not ots.exists():
        raise J.StampError(f"{rel} is missing: run `ots stamp {path.name}` and commit the proof with the prereg")
    blob = J._committed_blob(root, ots, J.StampError)
    info = verify_proof(blob, J.committed_sha256(path), require_bitcoin=require_bitcoin, explorer=explorer, name=rel)
    return {"file": rel, **info}


# ---------------------------------------------------------------- announcement

def _no_duplicates(pairs):
    keys = [k for k, _ in pairs]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise ValueError(f"duplicate key(s) {', '.join(dup)}")
    return dict(pairs)


def _parse_announced(blob: bytes, where: str) -> dict:
    J = _J()
    try:
        obj = json.loads(blob.decode("utf-8"), object_pairs_hook=_no_duplicates)
    except ValueError as exc:  # also UnicodeDecodeError and duplicate keys
        raise J.AnnouncementError(f"{where} is not valid JSON without duplicate keys: {exc}") from exc
    if not isinstance(obj, dict):
        raise J.AnnouncementError(f"{where} must be a JSON object {{candidate_id: {{sha256, url}}}}")
    return obj


def check_announcement(prereg_path: Path, prereg_sha256: str, *, now: datetime | None = None) -> dict:
    """The candidate's committed, never-changed entry in prereg/announced.json.

    Refuses (AnnouncementError) unless:
      * prereg/announced.json is committed with working bytes equal to HEAD;
      * it holds an entry for this candidate_id (the prereg's file stem) that
        is exactly {"sha256", "url"};
      * sha256 equals the committed prereg's sha256;
      * url matches ^https://(x|twitter)\\.com/[A-Za-z0-9_]+/status/[0-9]+$;
      * every version of the file in any commit that touched it (all refs)
        either lacks the entry or holds exactly this value. Changing an entry
        and changing it back is still refused;
      * the post time (decoded from the status id, see snowflake_time) is
        (c) on or after ANNOUNCE_FLOOR (2026-11-01), (a) on or after the
        prereg's first commit date minus one day, and (b) before ``now``,
        the run's start (default: the current time);
      * the prereg's .json.ots was in the tree of the oldest commit holding
        the entry: a proof existed when the hash was announced.
    Returns {"file", "sha256", "url", "commit", "post_time_utc"}; commit is
    the oldest commit holding the entry.
    """
    J = _J()
    path = Path(prereg_path).resolve()
    root = J._repo_root(path.parent)
    ann = path.parent / ANNOUNCED_FILE
    rel = ann.relative_to(root).as_posix()
    cid = path.stem
    if not ann.exists():
        raise J.AnnouncementError(
            f"{rel} is missing: post the prereg's sha256 publicly, then commit "
            f'{{"{cid}": {{"sha256": "...", "url": "https://x.com/.../status/..."}}}} there before Test'
        )
    current = _parse_announced(J._committed_blob(root, ann, J.AnnouncementError), rel)
    entry = current.get(cid)
    if entry is None:
        raise J.AnnouncementError(f"{rel} has no entry for {cid}: post its sha256 publicly and commit the entry before Test")
    if not isinstance(entry, dict) or set(entry) != {"sha256", "url"} or not all(isinstance(v, str) for v in entry.values()):
        raise J.AnnouncementError(f"{rel}: the entry for {cid} must be exactly {{\"sha256\": str, \"url\": str}}, got {entry!r}")
    if entry["sha256"] != prereg_sha256:
        raise J.AnnouncementError(
            f"{rel}: {cid} was announced with sha256 {entry['sha256']}, but the committed prereg has sha256 {prereg_sha256}"
        )
    if not ANNOUNCE_URL_RE.fullmatch(entry["url"]):
        raise J.AnnouncementError(
            f"{rel}: {cid} url {entry['url']!r} must match ^https://(x|twitter)\\.com/[A-Za-z0-9_]+/status/[0-9]+$"
        )
    commits = J._git(root, "log", "--all", "--full-history", "--format=%H", "--", rel).split()
    holding = []
    for commit in commits:  # newest first
        res = J._git_run(root, "show", f"{commit}:{rel}")
        if res.returncode != 0:
            continue  # the file was deleted in this commit
        then = _parse_announced(res.stdout, f"{rel} at commit {commit[:12]}")
        if cid not in then:
            continue
        if then[cid] != entry:
            raise J.AnnouncementError(
                f"{rel}: the entry for {cid} was changed after it was committed "
                f"(commit {commit[:12]} holds {then[cid]!r}, HEAD holds {entry!r}); announcements are written once"
            )
        holding.append(commit)
    if not holding:
        raise J.AnnouncementError(f"{rel}: no commit in history holds the entry for {cid}")
    first_commit = holding[-1]

    post = snowflake_time(int(entry["url"].rsplit("/", 1)[1]))
    if post < ANNOUNCE_FLOOR:
        raise J.AnnouncementError(
            f"{rel}: {cid}'s post {entry['url']} was created {utc_ms_iso(post)}, before "
            f"{ANNOUNCE_FLOOR.date()}: it cannot be this prereg's announcement"
        )
    registered = J.prereg_first_commit_date(path)
    if post.date() < registered - POST_COMMIT_SLACK:
        raise J.AnnouncementError(
            f"{rel}: {cid}'s post was created {utc_ms_iso(post)}, more than a day before the prereg's "
            f"first commit ({registered}): a hash cannot be announced before its rules were registered"
        )
    now = now if now is not None else _now_utc()
    if post >= now:
        raise J.AnnouncementError(
            f"{rel}: {cid}'s post was created {utc_ms_iso(post)}, after the run's start ({utc_ms_iso(now)})"
        )
    ots_rel = ots_path(path).relative_to(root).as_posix()
    if J._git_run(root, "cat-file", "-e", f"{first_commit}:{ots_rel}").returncode != 0:
        raise J.AnnouncementError(
            f"{rel}: {cid} was first announced in commit {first_commit[:12]}, which has no {ots_rel}: "
            "the .ots proof must be committed no later than the announcement"
        )
    return {"file": rel, "sha256": entry["sha256"], "url": entry["url"], "commit": first_commit,
            "post_time_utc": utc_ms_iso(post)}


def check_stamp_near_post(info: dict, post_time: datetime) -> None:
    """Final: the earliest verified Bitcoin block of the stamp lies in
    [post_time - 7 days, post_time + 48 h]. A proof stamped again later (to
    move the Final window) lands outside it."""
    J = _J()
    block = parse_utc(info["block_time_utc"])
    if block < post_time - STAMP_BEFORE_POST:
        raise J.StampError(
            f"{info['file']}: the earliest Bitcoin attestation (block {info['block_height']}, {info['block_time_utc']}) "
            f"is more than 7 days before the announcement post ({utc_ms_iso(post_time)})"
        )
    if block > post_time + STAMP_AFTER_POST:
        raise J.StampError(
            f"{info['file']}: the earliest Bitcoin attestation (block {info['block_height']}, {info['block_time_utc']}) "
            f"is more than 48 h after the announcement post ({utc_ms_iso(post_time)}): the proof was stamped "
            "again after the announcement, or confirmed too late"
        )


# ---------------------------------------------------------------- ots-status

def _first_commit_unix(root: Path, rel: str) -> int | None:
    J = _J()
    out = J._git(root, "log", "--all", "--diff-filter=A", "--format=%ct", "--", rel).split()
    return int(out[-1]) if out else None  # newest first: the last line is the oldest


def ots_status(root: Path, *, fail_pending_hours: float | None = None) -> list[dict]:
    """One row per registered prereg (prereg/NN-name.json): its proof's state.

    state: "missing" (no .ots), "invalid" (unparsable or not sha256),
    "digest-mismatch" (stamps other bytes than the prereg file), "pending"
    (calendar attestations only) or "bitcoin" (at least one Bitcoin
    attestation; parsed, not checked against an explorer). age_hours counts
    from the .ots file's first commit (None while uncommitted). stale: pending
    and older than fail_pending_hours. problem: missing, invalid, mismatch or stale.
    """
    J = _J()
    root = Path(root).resolve()
    now = _now_utc().timestamp()
    rows = []
    for p in sorted((root / "prereg").glob("*.json")):
        if not J.PREREG_NAME_RE.match(p.name):
            continue
        ots = ots_path(p)
        rel = ots.relative_to(root).as_posix()
        row = {"prereg": p.stem, "ots": rel, "state": "missing", "detail": "", "pending_calendars": [],
               "bitcoin_heights": [], "first_commit_utc": None, "age_hours": None, "stale": False}
        if ots.exists():
            first = _first_commit_unix(root, rel)
            if first is not None:
                row["first_commit_utc"] = _utc_iso(first)
                row["age_hours"] = round((now - first) / 3600, 1)
            try:
                dtf = parse_proof(ots.read_bytes(), rel)
            except J.StampError as exc:
                row["state"], row["detail"] = "invalid", str(exc)
            else:
                pending, bitcoin = attestations(dtf)
                expected = J.file_sha256(p)
                if not isinstance(dtf.file_hash_op, OpSHA256):
                    row["state"], row["detail"] = "invalid", f"not an sha256 proof ({type(dtf.file_hash_op).__name__})"
                elif dtf.file_digest.hex() != expected:
                    row["state"] = "digest-mismatch"
                    row["detail"] = f"stamps sha256 {dtf.file_digest.hex()}, {p.name} has {expected}"
                elif not pending and not bitcoin:
                    row["state"], row["detail"] = "invalid", "no attestation"
                else:
                    row["pending_calendars"] = pending
                    row["bitcoin_heights"] = sorted({h for h, _ in bitcoin})
                    row["state"] = "bitcoin" if bitcoin else "pending"
                    row["stale"] = (
                        row["state"] == "pending" and fail_pending_hours is not None
                        and row["age_hours"] is not None and row["age_hours"] > fail_pending_hours
                    )
        row["problem"] = row["state"] in ("missing", "invalid", "digest-mismatch") or row["stale"]
        rows.append(row)
    return rows


def format_status(rows: list[dict], fail_pending_hours: float | None) -> str:
    if not rows:
        return "no registered prereg under prereg/ (prereg/NN-name.json)"
    out = []
    for r in rows:
        age = f"{r['age_hours']:.1f}h" if r["age_hours"] is not None else "uncommitted"
        if r["state"] == "bitcoin":
            what = f"heights {','.join(map(str, r['bitcoin_heights']))}"
        elif r["state"] == "pending":
            what = f"calendars {', '.join(r['pending_calendars'])}"
        elif r["state"] == "missing":
            what = f"no {r['ots']}: run `ots stamp` and commit it"
        else:
            what = r["detail"]
        flag = f"  STALE: pending over {fail_pending_hours:g}h" if r["stale"] else ""
        state = "bitcoin (unverified, run open-final to verify)" if r["state"] == "bitcoin" else r["state"]
        out.append(f"{r['prereg']}  {state}  age {age}  {what}{flag}")
    return "\n".join(out)
