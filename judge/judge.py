"""Scoring, the benchmark, luck odds, the pre-registration guard and ``run``.

Rules (Global Constraints):
  test, final_alone: Sharpe > 1.0 AND max_dd < 0.25 AND Sharpe > bench Sharpe
  final ("holds up"): max_dd < 0.25 AND Sharpe >= bench Sharpe
Sharpe = mean / std(ddof=1) * sqrt(365), risk-free 0; std 0 or fewer than
2 points gives 0.

Window timing: a window (start, end) scores the daily returns dated start..end
inclusive. The book is flat at $10,000 at 00:00 UTC on ``start``, which is the
close of start - 1 day, so the first trades are decided on that close. No
position is held before it. Indicators may read up to 450 days earlier.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from judge import data
from judge.costs import net_returns
from judge.strategies.donchian import (
    LOOKBACKS,
    PER_COIN_CAP,
    VOL_TARGET,
    VOL_WINDOW,
    _allocation_frame,
    _apply_allocation,
    allocate,
    portfolio_weights,
    signal,
    vol_scale,
)

DAYS_PER_YEAR = 365
SHARPE_MIN = 1.0
MAX_DD_LIMIT = 0.25
KINDS = ("test", "final", "final_alone")
FINAL_KINDS = ("final", "final_alone")

Kind = Literal["test", "final", "final_alone"]
Window = tuple[date, date]


def sharpe(daily: pd.Series) -> float:
    x = pd.Series(daily, dtype=float).dropna().to_numpy()
    if len(x) < 2:
        return 0.0
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-15:  # float noise on a constant series counts as 0
        return 0.0
    return float(x.mean() / sd * math.sqrt(DAYS_PER_YEAR))


def max_drawdown(daily: pd.Series) -> float:
    """Largest peak-to-trough fall of the compounded equity curve, starting
    from equity 1.0 before the first return. Positive fraction."""
    x = pd.Series(daily, dtype=float).fillna(0.0).to_numpy()
    if len(x) == 0:
        return 0.0
    equity = np.concatenate([[1.0], np.cumprod(1.0 + x)])
    peak = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peak))


# ---------------------------------------------------------------- windows on series

def start_flat(weights: pd.DataFrame, window: Window | None) -> pd.DataFrame:
    """Zero every decision dated before start - 1 day (see module doc)."""
    if window is None:
        return weights
    out = weights.copy()
    out.loc[out.index < pd.Timestamp(window[0]) - pd.Timedelta(days=1)] = 0.0
    return out


def window_slice(returns: pd.Series, window: Window | None) -> pd.Series:
    """Return days start..end inclusive (returns are indexed by return day)."""
    if window is None:
        return returns
    start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    return returns.loc[(returns.index >= start) & (returns.index <= end)]


def benchmark_returns(
    close: pd.DataFrame,
    funding: pd.DataFrame | None,
    universe_by_month: dict,
    cost_per_side: float,
    excluded: dict | None = None,
    window: Window | None = None,
) -> pd.Series:
    """Equal-weight 1x hold of each month's universe, through costs and funding.

    Target weights are 1/len(universe) per listed member, so trades happen only
    when the universe changes, a coin delists, or (with ``excluded``, as for
    the candidate) the daily exit check drops a coin. With ``window`` the hold
    starts flat and only the window's returns are returned.
    """
    held = pd.DataFrame(1.0, index=close.index, columns=close.columns).where(close.notna(), 0.0)
    weights, changed = allocate(held, close, universe_by_month, gross_cap=1.0, excluded=excluded)
    net = net_returns(start_flat(weights, window), changed, close, funding, cost_per_side, rebalance_threshold=None)
    return window_slice(net, window)


def verdict(strategy: pd.Series, bench: pd.Series, kind: Kind) -> dict:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    s, dd, b = sharpe(strategy), max_drawdown(strategy), sharpe(bench)
    if kind == "final":
        rules = {"max_dd_below_25pct": dd < MAX_DD_LIMIT, "sharpe_holds_vs_bench": s >= b}
    else:
        rules = {
            "sharpe_above_1": s > SHARPE_MIN,
            "max_dd_below_25pct": dd < MAX_DD_LIMIT,
            "sharpe_beats_bench": s > b,
        }
    return {"passed": bool(all(rules.values())), "sharpe": s, "max_dd": dd, "bench_sharpe": b, "rules": rules}


# ---------------------------------------------------------------- luck odds

LUCK_ODDS_METHOD = (
    "Luck odds are the share of n_sims no-skill strategies that pass the same verdict rule "
    "against the same benchmark. Each sim gives every coin one independent random state path "
    "per lookback (one path for each of the candidate's lookbacks) and averages them, as the "
    "combo averages its lookback signals. Path n mirrors the candidate's lookback-n signal "
    "inside the window (active coin-days, decision days start-1..end-1): its state "
    "distribution (p_long, p_flat, p_short) and its daily change rate. A path starts at 0; "
    "each day, with probability flip_prob, it redraws its state from that distribution "
    "(possibly the same state), so its stationary distribution is the candidate's and it "
    "changes state with probability flip_prob x (1 - sum p_i^2); hence flip_prob = rate / "
    "(1 - sum p_i^2), capped at 1. The per-lookback values used are in luck_odds_paths. "
    "Coin weight = vol_scale(vol_target, vol_window, per_coin_cap) x mean state, then the "
    "candidate's allocation (monthly universe, daily exit exclusions, delisting, gross cap), "
    "its rebalance threshold (a path state change counts as a signal change), costs and "
    "funding. The book starts flat at the window start and only window returns are scored. "
    "The rng is numpy default_rng(seed). Known gap: paths are independent across lookbacks "
    "and coins, while the candidate's lookbacks change on clustered days."
)


def _random_states(
    rng: np.random.Generator, shape: tuple[int, int], flip_prob: float, states: np.ndarray, p=None,
) -> np.ndarray:
    """State paths: start 0; each day with prob flip_prob draw a new state from
    ``states`` with probabilities ``p`` (uniform if None; it may equal the
    current one), else keep."""
    t_len, n_coins = shape
    flips = rng.random(shape) < flip_prob
    draws = rng.choice(states, size=shape, p=p)
    last = np.where(flips, np.arange(t_len)[:, None], -1)
    last = np.maximum.accumulate(last, axis=0)
    picked = np.take_along_axis(draws, np.clip(last, 0, None), axis=0)
    return np.where(last >= 0, picked, 0).astype(float)


def _per_path(value, n_paths: int, name: str) -> list:
    if isinstance(value, (list, tuple)):
        if len(value) != n_paths:
            raise ValueError(f"{name} needs one entry per lookback ({n_paths}), got {len(value)}")
        return list(value)
    return [value] * n_paths


def luck_sims(
    close: pd.DataFrame,
    funding: pd.DataFrame | None,
    universe_by_month: dict,
    long_short: bool,
    flip_prob,
    n_sims: int,
    seed: int,
    cost_per_side: float,
    gross_cap: float | None,
    *,
    state_probs=None,
    rebalance_threshold: float | None = None,
    lookbacks=LOOKBACKS,
    per_coin_cap: float = PER_COIN_CAP,
    vol_target: float = VOL_TARGET,
    vol_window: int = VOL_WINDOW,
    excluded: dict | None = None,
    window: Window | None = None,
    return_states: bool = False,
    return_weights: bool = False,
) -> list:
    """The n_sims random strategies of LUCK_ODDS_METHOD: their scored net
    returns, or (for tests) each sim's mean state or target weight frame.

    flip_prob: one float for every path, or one per lookback. state_probs:
    None (uniform redraws) or one probability vector per lookback over the
    states [0, 1] (long-only) or [-1, 0, 1] (long-short); see sim_path_args.
    """
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")
    n_paths = len(lookbacks)
    if n_paths == 0:
        raise ValueError("lookbacks must not be empty")
    flips = _per_path(flip_prob, n_paths, "flip_prob")
    probs = _per_path(state_probs, n_paths, "state_probs") if state_probs is not None else [None] * n_paths
    rng = np.random.default_rng(seed)
    states = np.array([-1, 0, 1] if long_short else [0, 1])
    idx, cols = close.index, close.columns

    # Everything that does not depend on the random paths is computed once.
    scale = np.column_stack(
        [vol_scale(close[c], vol_target, vol_window, per_coin_cap).to_numpy(dtype=float) for c in cols]
    ) if len(cols) else np.zeros((len(idx), 0))
    active, size, alloc_flags = _allocation_frame(close, universe_by_month, excluded)
    shape = close.shape

    out = []
    for _ in range(n_sims):
        total = np.zeros(shape)
        moved = np.zeros(shape, dtype=bool)
        for f, p in zip(flips, probs):
            path = _random_states(rng, shape, f, states, p)
            prev = np.vstack([np.zeros((1, shape[1])), path[:-1]])
            total += path
            moved |= path != prev
        state = total / n_paths
        if return_states:
            out.append(pd.DataFrame(state, index=idx, columns=cols))
            continue
        coin_w = pd.DataFrame(scale * state, index=idx, columns=cols)
        weights = start_flat(_apply_allocation(coin_w, active, size, gross_cap), window)
        if return_weights:
            out.append(weights)
            continue
        changed = alloc_flags | pd.DataFrame(moved, index=idx, columns=cols)
        sim = net_returns(weights, changed, close, funding, cost_per_side, rebalance_threshold)
        out.append(window_slice(sim, window))
    return out


def luck_odds(
    close: pd.DataFrame,
    funding: pd.DataFrame | None,
    universe_by_month: dict,
    long_short: bool,
    flip_prob,
    n_sims: int,
    seed: int,
    kind: Kind,
    cost_per_side: float,
    gross_cap: float | None,
    bench: pd.Series | None = None,
    *,
    state_probs=None,
    rebalance_threshold: float | None = None,
    lookbacks=LOOKBACKS,
    per_coin_cap: float = PER_COIN_CAP,
    vol_target: float = VOL_TARGET,
    vol_window: int = VOL_WINDOW,
    excluded: dict | None = None,
    window: Window | None = None,
) -> float:
    """Share of n_sims random-position strategies that pass ``verdict``
    (LUCK_ODDS_METHOD). Pass every setting exactly as the candidate uses it.

    ``bench`` defaults to benchmark_returns with the same excluded and window;
    a passed-in one is sliced to the window.
    """
    if bench is None:
        bench = benchmark_returns(close, funding, universe_by_month, cost_per_side, excluded=excluded, window=window)
    else:
        bench = window_slice(bench, window)
    sims = luck_sims(
        close, funding, universe_by_month, long_short, flip_prob, n_sims, seed, cost_per_side, gross_cap,
        state_probs=state_probs, rebalance_threshold=rebalance_threshold, lookbacks=lookbacks,
        per_coin_cap=per_coin_cap, vol_target=vol_target, vol_window=vol_window, excluded=excluded, window=window,
    )
    return sum(verdict(s, bench, kind)["passed"] for s in sims) / n_sims


def flip_prob_for_rate(rate: float, probs) -> float:
    """flip_prob at which a path redrawing from ``probs`` changes state at ``rate``.

    A redraw changes the state with probability 1 - sum(p_i^2) (in the
    stationary regime), so flip_prob = rate / (1 - sum p_i^2), capped at 1;
    0 when the path can never change (one state has probability 1).
    """
    p = np.asarray(probs, dtype=float)
    move = 1.0 - float(np.sum(p * p))
    if move <= 1e-12:
        return 0.0
    return float(min(1.0, rate / move))


def observed_path_models(
    close: pd.DataFrame,
    universe_by_month: dict,
    excluded: dict | None,
    lookbacks,
    long_short: bool,
    window: Window,
) -> list[dict]:
    """Per lookback: the candidate's signal statistics inside the window.

    Over active coin-days on the window's decision days (start-1 .. end-1,
    the decisions that earn the scored returns): change_rate (share of
    coin-days whose position differs from the day before), p_long, p_flat,
    p_short, and the matching flip_prob (flip_prob_for_rate). With no active
    coin-day a lookback is all flat with flip_prob 0.
    """
    active, _, _ = _allocation_frame(close, universe_by_month, excluded)
    idx = close.index
    first = pd.Timestamp(window[0]) - pd.Timedelta(days=1)
    last = pd.Timestamp(window[1]) - pd.Timedelta(days=1)
    in_win = np.asarray((idx >= first) & (idx <= last))
    mask = active.to_numpy(dtype=bool) & in_win[:, None]
    coin_days = int(mask.sum())
    out = []
    for n in lookbacks:
        if coin_days == 0:
            out.append({"lookback": int(n), "change_rate": 0.0, "p_long": 0.0, "p_flat": 1.0, "p_short": 0.0,
                        "flip_prob": 0.0})
            continue
        pos = np.column_stack(
            [signal(close[c], n, long_short).to_numpy(dtype=float) for c in close.columns]
        )
        prev = np.vstack([np.zeros((1, pos.shape[1])), pos[:-1]])
        sel = pos[mask]
        rate = float((pos != prev)[mask].mean())
        p_long, p_short = float((sel == 1).mean()), float((sel == -1).mean())
        p_flat = 1.0 - p_long - p_short
        probs = [p_short, p_flat, p_long] if long_short else [p_flat, p_long]
        out.append({"lookback": int(n), "change_rate": rate, "p_long": p_long, "p_flat": p_flat,
                    "p_short": p_short, "flip_prob": flip_prob_for_rate(rate, probs)})
    return out


def sim_path_args(models: list[dict], long_short: bool) -> tuple[list[float], list[list[float]]]:
    """(flip_prob per lookback, state_probs per lookback) for luck_sims/luck_odds."""
    flips = [m["flip_prob"] for m in models]
    raw = (
        [[m["p_short"], m["p_flat"], m["p_long"]] for m in models] if long_short
        else [[m["p_flat"], m["p_long"]] for m in models]
    )
    probs = []
    for p in raw:  # guard float rounding so numpy accepts the vectors
        a = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
        probs.append(list(a / a.sum()))
    return flips, probs


# ---------------------------------------------------------------- pre-registration

REQUIRED_KEYS = (
    "candidate_id", "paper", "long_short", "lookbacks", "vol_target", "vol_window",
    "per_coin_cap_repro", "gross_cap", "cost_per_side", "rebalance_threshold",
    "universe_size", "paper_sample_end", "seed", "n_sims", "judge_tree_hash",
)
# Fixed by the judge for test, final and final_alone (Global Constraints). A
# prereg must state exactly these values; they are not candidate choices.
JUDGING = {
    "cost_per_side": 0.0011,  # fee 0.06% + slippage 0.05% per side
    "rebalance_threshold": 0.2,
    "gross_cap": 1.0,
    "n_sims": 1000,
    "universe_size": 20,
}
# Paper reproduction (Task 8) only: the paper's own 10 bps and 20% threshold.
# run() never uses these; reproduction must select them explicitly.
REPRO = {
    "cost_per_side": 0.0010,
    "rebalance_threshold": 0.2,
}
CANDIDATE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
PREREG_NAME_RE = re.compile(r"^\d{2}-[A-Za-z0-9][A-Za-z0-9._-]*\.json$")
TREE_HASH_RE = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")
TEST_FLOOR = date(2023, 1, 1)
TEST_END = date(2026, 10, 19)
FINAL_DAYS = 90
FINAL_ALONE_UNDER_DAYS = 365  # a Test window shorter than 12 months
LOOKBACK_DAYS = 450
# The newest archive days may be unpublished (data.daily_klines), so Final
# may only open once its last day is this many days old.
FINAL_DATA_LAG_DAYS = 3
MARKET = "um"
UNIVERSE_MIN_AGE_DAYS = 365
UNIVERSE_MIN_MEDIAN_VOLUME = 2_000_000
EXIT_ROWS = data.VOLUME_WINDOW + 1  # exit_flags reads 30 daily changes: 31 closes

CODE_ROOT = Path(__file__).resolve().parent.parent  # repo holding this judge/ code


class JudgeRefusal(Exception):
    """The judge refuses to run. Base of every refusal below."""


class PreregError(JudgeRefusal):
    """The prereg file (or its stamp_confirmed companion) is unreadable, incomplete or invalid."""


class PreregMismatch(JudgeRefusal):
    """A stamped file (prereg or judge code) differs from what was registered."""


class DataNotReady(JudgeRefusal):
    """The data does not cover the window."""


class FinalNotReady(DataNotReady):
    """Final cannot run yet: no confirmed stamp date, or data short of the window."""


class FinalGateError(JudgeRefusal):
    """Final may not be opened (no matching Test pass, already opened, wrong kind)."""


class ResultExists(JudgeRefusal):
    """A result file for this candidate and kind already exists; never overwritten."""


class ReproGateError(JudgeRefusal):
    """Test (or Final-alone) needs a committed, passing reproduction of the same strategy."""


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _validate(raw: dict, name: str) -> None:
    def bad(key: str, want: str) -> PreregError:
        return PreregError(f"{name}: {key} must be {want}, got {raw[key]!r}")

    if not isinstance(raw["candidate_id"], str) or not CANDIDATE_ID_RE.match(raw["candidate_id"]):
        raise bad("candidate_id", f"a string matching {CANDIDATE_ID_RE.pattern}")
    if not isinstance(raw["paper"], str) or not raw["paper"].strip():
        raise bad("paper", "a non-empty string")
    if not isinstance(raw["long_short"], bool):
        raise bad("long_short", "a JSON bool (true/false)")
    lbs = raw["lookbacks"]
    if not isinstance(lbs, list) or not lbs or not all(_is_int(n) and n > 0 for n in lbs) or len(set(lbs)) != len(lbs):
        raise bad("lookbacks", "a non-empty list of distinct positive integers")
    for key in ("vol_target", "per_coin_cap_repro"):
        if not _is_num(raw[key]) or raw[key] <= 0:
            raise bad(key, "a positive number")
    if not _is_int(raw["vol_window"]) or raw["vol_window"] < 2:
        raise bad("vol_window", "an integer >= 2")
    if not _is_int(raw["seed"]) or raw["seed"] < 0:
        raise bad("seed", "a non-negative integer")
    if not isinstance(raw["judge_tree_hash"], str) or not TREE_HASH_RE.match(raw["judge_tree_hash"]):
        raise bad("judge_tree_hash", "a git tree hash (hex string)")
    if not isinstance(raw["paper_sample_end"], str):
        raise bad("paper_sample_end", "an ISO date string")
    _paper_end(raw, name)
    for key, fixed in JUDGING.items():
        v = raw[key]
        ok = (_is_int(v) if isinstance(fixed, int) else _is_num(v)) and v == fixed
        if not ok:
            raise bad(key, f"{fixed!r} ({type(fixed).__name__}, fixed by the judge)")


def load_prereg(path: Path) -> dict:
    """Read and validate a prereg: required keys, strict types, judging constants."""
    path = Path(path)
    try:
        raw = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise PreregError(f"{path}: cannot read prereg: {exc}") from exc
    if not isinstance(raw, dict):
        raise PreregError(f"{path.name}: prereg must be a JSON object")
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise PreregError(f"{path.name}: missing required key(s): {', '.join(missing)}")
    _validate(raw, path.name)
    return raw


def load_prereg_params(path: Path) -> dict:
    """Read a prereg draft (prereg/drafts/NN-name.json) for reproduction.

    The same key and type validation as load_prereg, and NO integrity check:
    a draft is not registered yet and cannot carry the final judge_tree_hash.
    """
    return load_prereg(path)


def _paper_end(prereg: dict, name: str = "prereg") -> date:
    try:
        return date.fromisoformat(str(prereg["paper_sample_end"]))
    except ValueError as exc:
        raise PreregError(f"{name}: paper_sample_end is not an ISO date: {prereg['paper_sample_end']!r}") from exc


def _git_run(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True)
    except OSError as exc:
        raise PreregMismatch(f"cannot run git in {cwd}: {exc}") from exc


def _git(cwd: Path, *args: str) -> str:
    res = _git_run(cwd, *args)
    if res.returncode != 0:
        raise PreregMismatch(f"git {' '.join(args)} failed in {cwd}: {res.stderr.decode(errors='replace').strip()}")
    return res.stdout.decode("utf-8").strip()


def _repo_root(path: Path) -> Path:
    return Path(_git(Path(path), "rev-parse", "--show-toplevel")).resolve()


def file_sha256(path: Path) -> str:
    """sha256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _committed_blob(root: Path, path: Path, err=PreregMismatch) -> bytes:
    """The HEAD blob of a tracked file whose working bytes equal it exactly.

    Refuses (``err``) an untracked or uncommitted file, and a working file
    whose bytes differ from the blob, which ``git status`` can hide (e.g. CRLF
    in the working copy under ``eol=lf``).
    """
    rel = Path(path).resolve().relative_to(root).as_posix()
    res = _git_run(root, "cat-file", "-p", f"HEAD:{rel}")
    if res.returncode != 0:
        raise err(f"{rel} is not committed (no HEAD:{rel}); commit it first")
    blob = res.stdout
    work = Path(path).read_bytes()
    if work != blob:
        raise err(
            f"{rel}: working file bytes differ from the committed blob "
            f"(sha256 {hashlib.sha256(work).hexdigest()} vs HEAD {hashlib.sha256(blob).hexdigest()}); "
            "check line endings, then restore or recommit"
        )
    return blob


def _single_commit(root: Path, path: Path, err, why: str) -> tuple[str, date]:
    """(commit hash, committer date) of the only commit, on any ref, touching ``path``.

    Refuses (``err``) when the file was touched by more than one commit (edited,
    deleted, re-added) or by none."""
    rel = Path(path).resolve().relative_to(root).as_posix()
    lines = [ln for ln in _git(root, "log", "--all", "--format=%H %cs", "--", rel).splitlines() if ln.strip()]
    if len(lines) != 1:
        raise err(f"{rel} must have exactly one commit touching it, found {len(lines)}: {why}")
    commit, day = lines[0].split()
    return commit, date.fromisoformat(day)


def committed_sha256(path: Path) -> str:
    """sha256 of the committed blob of a file (refuses if the working copy differs)."""
    path = Path(path).resolve()
    return hashlib.sha256(_committed_blob(_repo_root(path.parent), path)).hexdigest()


def check_integrity(prereg_path: Path) -> None:
    """Refuse to run unless the prereg and judge code are exactly as stamped.

    1. The prereg is <repo>/prereg/NN-name.json and its candidate_id is NN-name.
    2. The prereg, its .sha256 and (if present) .stamp_confirmed are committed,
       and each working file's bytes equal its HEAD blob.
    3. sha256 of the committed prereg blob equals the .sha256 companion
       (its first token, so ``sha256sum`` output works).
    4. The prereg's judge_tree_hash equals ``git rev-parse HEAD:judge``.
    5. The prereg and its .sha256 each have exactly one commit touching them
       (on any ref): a registered prereg is never edited, only replaced by a
       new candidate number.
    6. ``git status`` shows nothing, tracked or untracked, under judge/ or prereg/.
    Each failure raises PreregMismatch naming the file (and both hashes).
    """
    path = Path(prereg_path).resolve()
    root = _repo_root(path.parent)
    rel = path.relative_to(root).as_posix()
    if path.parent != root / "prereg" or not PREREG_NAME_RE.match(path.name):
        raise PreregMismatch(f"{rel}: a prereg must be prereg/NN-name.json in the judge repo")
    prereg = load_prereg(path)
    if prereg["candidate_id"] != path.stem:
        raise PreregMismatch(f"{rel}: candidate_id {prereg['candidate_id']!r} must equal the file name {path.stem!r}")
    sha_file = path.with_suffix(".sha256")
    if not sha_file.exists():
        raise PreregMismatch(f"{path.name}: no companion {sha_file.name} holding the registered sha256")
    blob = _committed_blob(root, path)
    _committed_blob(root, sha_file)
    stamp_file = path.with_suffix(".stamp_confirmed")
    if stamp_file.exists():
        _committed_blob(root, stamp_file)
    parts = sha_file.read_text().split()
    registered = parts[0].lower() if parts else "(empty)"
    actual = hashlib.sha256(blob).hexdigest()
    if actual != registered:
        raise PreregMismatch(
            f"{path.name} differs from its registration: sha256 is {actual}, "
            f"registered in {sha_file.name} is {registered}"
        )
    recorded = str(prereg["judge_tree_hash"])
    head = _git(root, "rev-parse", "HEAD:judge")
    if recorded != head:
        raise PreregMismatch(
            f"judge/ code differs from {path.name}: git rev-parse HEAD:judge is {head}, "
            f"judge_tree_hash recorded in {path.name} is {recorded}"
        )
    for f in (path, sha_file):
        _single_commit(root, f, PreregMismatch, "a registered prereg is written once; a change needs a new candidate number")
    dirty = _git(root, "status", "--porcelain", "--untracked-files=all", "--", "judge", "prereg")
    if dirty:
        files = ", ".join(line[3:] for line in dirty.splitlines())
        raise PreregMismatch(f"uncommitted changes under judge/ or prereg/: {files}")


def _test_window(prereg: dict) -> Window:
    start = max(TEST_FLOOR, _paper_end(prereg) + timedelta(days=1))
    if start > TEST_END:
        raise PreregError(f"paper_sample_end {prereg['paper_sample_end']} leaves no Test window")
    return start, TEST_END


def is_final_alone(prereg: dict) -> bool:
    """A candidate whose Test window is under 12 months is judged on Final alone."""
    start, end = _test_window(prereg)
    return (end - start).days + 1 < FINAL_ALONE_UNDER_DAYS


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def window(kind: Kind, prereg: dict, stamp_confirmed: date | None, latest_data_date: date | None = None) -> Window:
    """(start, end) of the window, both inclusive.

    test: max(2023-01-01, paper_sample_end + 1 day) .. 2026-10-19.
    final, final_alone: the 90 days starting the day after stamp_confirmed
    (the day the prereg's .ots stamp was confirmed in a Bitcoin block). The end
    must be on or before latest_data_date (default: today UTC minus
    FINAL_DATA_LAG_DAYS), else FinalNotReady.
    """
    if kind == "test":
        return _test_window(prereg)
    if kind not in FINAL_KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if stamp_confirmed is None:
        raise FinalNotReady(
            f"{kind} needs stamp_confirmed, the date the prereg's .ots stamp was confirmed in a "
            "Bitcoin block: commit it as prereg/NN-name.stamp_confirmed (one ISO date line)"
        )
    start = stamp_confirmed + timedelta(days=1)
    end = start + timedelta(days=FINAL_DAYS - 1)
    latest = latest_data_date if latest_data_date is not None else _today_utc() - timedelta(days=FINAL_DATA_LAG_DAYS)
    if end > latest:
        raise FinalNotReady(f"Final window {start}..{end} ends after the latest data date {latest}")
    return start, end


def resolve_stamp_confirmed(prereg_path: Path, flag: date | None) -> tuple[date | None, str | None]:
    """(stamp_confirmed, source) for Final. Only the committed file counts.

    prereg/NN-name.stamp_confirmed holds a single ISO date line; check_integrity
    makes sure it is committed. The --stamp-confirmed flag may only confirm it:
    a flag without the file, or one that disagrees with it, is refused.
    source is "file" or "file+flag"; (None, None) when neither is given.
    """
    side = Path(prereg_path).with_suffix(".stamp_confirmed")
    if not side.exists():
        if flag is not None:
            raise FinalNotReady(
                f"--stamp-confirmed may only confirm the committed {side.name}, which does not exist: "
                "commit the stamp_confirmed file first"
            )
        return None, None
    lines = [ln.strip() for ln in side.read_text().splitlines() if ln.strip()]
    if len(lines) != 1:
        raise PreregError(f"{side.name} must hold a single ISO date line, found {len(lines)} lines")
    try:
        from_file = date.fromisoformat(lines[0])
    except ValueError as exc:
        raise PreregError(f"{side.name}: not an ISO date: {lines[0]!r}") from exc
    if flag is not None and flag != from_file:
        raise PreregError(f"--stamp-confirmed {flag} conflicts with {side.name} ({from_file})")
    return from_file, ("file+flag" if flag is not None else "file")


def prereg_first_commit_date(prereg_path: Path) -> date:
    """Committer date (%cs, the committer's own timezone) of the oldest commit that added the prereg."""
    path = Path(prereg_path).resolve()
    root = _repo_root(path.parent)
    rel = path.relative_to(root).as_posix()
    dates = _git(root, "log", "--all", "--diff-filter=A", "--format=%cs", "--", rel).split()
    if not dates:
        raise PreregMismatch(f"{rel} has no commit that added it")
    return date.fromisoformat(dates[-1])  # git log lists newest first


def check_stamp(prereg_path: Path, stamp_confirmed: date) -> None:
    """Refuse a stamp date that could put Final on already-seen data.

    (a) The Final window start (stamp + 1 day) must be after the Test window
        end (TEST_END), so Final never overlaps Test.
    (b) The stamp date must be on or after the prereg's first commit date: a
        stamp cannot be confirmed before the file it stamps was committed.
    """
    start = stamp_confirmed + timedelta(days=1)
    if not start > TEST_END:
        raise FinalGateError(
            f"stamp_confirmed {stamp_confirmed} puts the Final start ({start}) on or before the end of "
            f"the Test window ({TEST_END}); Final must start after the Test window"
        )
    first = prereg_first_commit_date(prereg_path)
    if stamp_confirmed < first:
        raise FinalGateError(
            f"stamp_confirmed {stamp_confirmed} is before the prereg's first commit ({first}); "
            "a stamp cannot predate the file it stamps"
        )


def check_stamp_file(prereg_path: Path, final_start: date) -> None:
    """The committed .stamp_confirmed has exactly one commit (on any ref),
    dated on or before the Final window start, so the window is fixed before
    its first day and never moved afterwards."""
    path = Path(prereg_path).resolve()
    root = _repo_root(path.parent)
    side = path.with_suffix(".stamp_confirmed")
    _, committed = _single_commit(root, side, FinalGateError, "stamp_confirmed is written once and never changed")
    if committed > final_start:
        raise FinalGateError(
            f"{side.name} was committed on {committed}, after the Final window start {final_start}: "
            "the stamp date must be committed on or before the window starts"
        )


def _window_json(win: Window) -> dict:
    return {"start": win[0].isoformat(), "end": win[1].isoformat()}


def _check_test_pass(root: Path, rd: Path, candidate_id: str, prereg_sha256: str, judge_tree_hash: str) -> None:
    test_file = rd / f"{candidate_id}-test.json"
    if not test_file.exists():
        raise FinalGateError(f"no Test result {test_file.name}: Final opens only after a committed Test pass")
    _committed_blob(root, test_file, FinalGateError)
    try:
        res = json.loads(test_file.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise FinalGateError(f"Test result {test_file.name} is not valid JSON: {exc}") from exc
    if res.get("kind") != "test" or res.get("candidate_id") != candidate_id:
        raise FinalGateError(f"{test_file.name} is not the Test result of {candidate_id}")
    if res.get("passed") is not True:
        raise FinalGateError(f"Test result {test_file.name} says passed: false; Final stays closed")
    m = res.get("manifest") or {}
    if m.get("prereg_sha256") != prereg_sha256:
        raise FinalGateError(
            f"{test_file.name} was run on another prereg: prereg_sha256 {m.get('prereg_sha256')} vs current {prereg_sha256}"
        )
    if m.get("judge_tree_hash") != judge_tree_hash:
        raise FinalGateError(
            f"{test_file.name} was run by other judge code: judge_tree_hash {m.get('judge_tree_hash')} vs current {judge_tree_hash}"
        )


def _marker_path(rd: Path, candidate_id: str) -> Path:
    return rd / f"{candidate_id}.final_opened"


def _final_history_guard(root: Path, rd: Path, candidate_id: str) -> None:
    """Refuse if any Final result for this candidate was ever in git history, or exists."""
    finals = [rd / f"{candidate_id}-{k}.json" for k in FINAL_KINDS]
    rels = [f.relative_to(root).as_posix() for f in finals]
    if _git(root, "log", "--all", "--format=%H", "--", *rels):
        raise FinalGateError(f"git history already holds a Final result for {candidate_id} ({', '.join(rels)}); Final runs once")
    if any(f.exists() for f in finals):
        raise FinalGateError(f"a Final result for {candidate_id} already exists; Final runs once")


def _results_clean(root: Path, rd: Path, allow: str | None = None) -> None:
    dirty = _git(root, "status", "--porcelain", "--untracked-files=all", "--", rd.relative_to(root).as_posix())
    others = [ln for ln in dirty.splitlines() if ln and ln != allow]
    if others:
        raise FinalGateError(f"uncommitted changes under results/: {', '.join(ln[3:] for ln in others)}; commit earlier results first")


def _marker_mismatch(marker: Path, expected: dict) -> list[str]:
    try:
        rec = json.loads(marker.read_text(encoding="utf-8"))
    except ValueError:
        rec = {}
    return [k for k, v in expected.items() if rec.get(k) != v]


def open_final(
    candidate_id: str,
    *,
    prereg_sha256: str,
    judge_tree_hash: str,
    window: Window,
    final_alone: bool = False,
    results_dir: Path | None = None,
) -> Path:
    """Final step 1: run the gate checks and write results/{id}.final_opened.

    No data is loaded. The marker holds the prereg sha256, judge tree hash and
    window; the operator commits it, and only then may ``run`` load Final data
    (check_final_opened). Refuses (FinalGateError) when:
      * a Final result was ever in git history, or exists;
      * the marker was ever committed (Final is already open);
      * results/ has uncommitted changes (other than this marker);
      * unless Final-alone: there is no committed, unmodified Test result for
        this prereg_sha256 and judge_tree_hash that says passed: true;
      * an uncommitted marker exists that does not match.
    An uncommitted matching marker is left as is (idempotent).
    """
    rd = (Path(results_dir) if results_dir is not None else CODE_ROOT / "results").resolve()
    root = _repo_root(rd.parent)
    marker = _marker_path(rd, candidate_id)
    marker_rel = marker.relative_to(root).as_posix()
    _final_history_guard(root, rd, candidate_id)
    if _git(root, "log", "--all", "--format=%H", "--", marker_rel):
        raise FinalGateError(f"git history already holds {marker_rel}: Final was already opened; run --kind final, do not reopen")
    _results_clean(root, rd, allow=f"?? {marker_rel}" if marker.exists() else None)
    if not final_alone:
        _check_test_pass(root, rd, candidate_id, prereg_sha256, judge_tree_hash)
    expected = {"prereg_sha256": prereg_sha256, "judge_tree_hash": judge_tree_hash, "window": _window_json(window)}
    if marker.exists():
        diff = _marker_mismatch(marker, expected)
        if diff:
            raise FinalGateError(f"{marker.name} does not match this Final ({', '.join(diff)}); it was opened differently")
        return marker
    rd.mkdir(parents=True, exist_ok=True)
    rec = {"candidate_id": candidate_id, "final_alone": final_alone, "opened_utc": _utc_now(), **expected}
    try:
        with open(marker, "x", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec) + "\n")
    except FileExistsError as exc:
        raise FinalGateError(f"Final for {candidate_id} was opened concurrently ({marker.name})") from exc
    return marker


def check_final_opened(
    candidate_id: str,
    *,
    prereg_sha256: str,
    judge_tree_hash: str,
    window: Window,
    final_alone: bool = False,
    results_dir: Path | None = None,
) -> dict:
    """Final step 2 gate, checked by ``run`` before any Final data loads.

    The marker must be committed (tracked, working bytes equal to HEAD), touched
    by exactly one commit, and match (prereg_sha256, judge_tree_hash, window).
    No Final result may exist or ever have been in history, results/ must be
    clean, and (unless Final-alone) the committed Test pass must still match.
    A rerun after a failure passes these checks again: that is the resume.
    Returns {"path", "commit"} of the marker for the manifest.
    """
    rd = (Path(results_dir) if results_dir is not None else CODE_ROOT / "results").resolve()
    root = _repo_root(rd.parent)
    marker = _marker_path(rd, candidate_id)
    marker_rel = marker.relative_to(root).as_posix()
    _final_history_guard(root, rd, candidate_id)
    if not marker.exists():
        raise FinalGateError(
            f"{marker_rel} is missing: run `python -m judge open-final <prereg>`, commit the marker, then run Final"
        )
    _committed_blob(root, marker, FinalGateError)
    commit, _ = _single_commit(root, marker, FinalGateError, "the Final marker is written once by open-final")
    _results_clean(root, rd)
    if not final_alone:
        _check_test_pass(root, rd, candidate_id, prereg_sha256, judge_tree_hash)
    expected = {"prereg_sha256": prereg_sha256, "judge_tree_hash": judge_tree_hash, "window": _window_json(window)}
    diff = _marker_mismatch(marker, expected)
    if diff:
        raise FinalGateError(f"committed {marker.name} does not match this Final ({', '.join(diff)})")
    return {"path": marker_rel, "commit": commit}


def _final_context(path: Path, prereg: dict, kind: Kind, stamp_flag: date | None, latest_data_date: date | None = None):
    """Shared Final preamble: kind vs candidate type, stamp, window, stamp rules."""
    cid = prereg["candidate_id"]
    alone = is_final_alone(prereg)
    if kind == "final" and alone:
        raise FinalGateError(f"{cid} has a Test window under 12 months: it is Final-alone, run --kind final_alone")
    if kind == "final_alone" and not alone:
        raise FinalGateError(f"{cid} has a Test window of 12 months or more: run --kind final after a Test pass, not final_alone")
    stamp, source = resolve_stamp_confirmed(path, stamp_flag)
    win = window(kind, prereg, stamp, latest_data_date=latest_data_date)
    check_stamp(path, stamp)
    check_stamp_file(path, win[0])
    return stamp, source, win


def prepare_final(prereg_path: Path, *, stamp_confirmed: date | None = None, results_dir: Path | None = None) -> Path:
    """``python -m judge open-final``: every Final gate check, then write the
    marker and stop. Loads no data. The kind (final or final_alone) follows
    from the candidate's Test window. The window's data need not exist yet.
    A Final-alone candidate (no Test) also needs its reproduction pass."""
    path = Path(prereg_path).resolve()
    prereg = load_prereg(path)
    check_integrity(path)
    root = _repo_root(path.parent)
    if root != Path(CODE_ROOT).resolve():
        raise PreregMismatch(f"judge code runs from {Path(CODE_ROOT).resolve()} but {path.name} is in {root}")
    rd = Path(results_dir) if results_dir is not None else root / "results"
    kind = "final_alone" if is_final_alone(prereg) else "final"
    _, _, win = _final_context(path, prereg, kind, stamp_confirmed, latest_data_date=date.max)
    if kind == "final_alone":
        check_repro_pass(root, rd, prereg)
    return open_final(
        prereg["candidate_id"], prereg_sha256=committed_sha256(path), judge_tree_hash=_git(root, "rev-parse", "HEAD:judge"),
        window=win, final_alone=kind == "final_alone", results_dir=rd,
    )


# ---------------------------------------------------------------- reproduction (Task 8)
#
# Before a candidate is judged, the judge must reproduce the paper's own
# single-asset Sharpe ratios with the paper's own costs (REPRO). A miss of more
# than REPRO_MAX_REL_ERROR on any check fails the reproduction; nobody tunes.

exchange_daily = data.exchange_daily  # the default reproduction price loader

REPRO_MAX_REL_ERROR = 0.20
REPRO_SAMPLE_END = date(2025, 3, 19)  # SSRN 5209907 sample end
REPRO_START = {"BTC": date(2015, 1, 1), "ETH": date(2015, 8, 8)}
REPRO_PERIOD_NOTES = {
    "BTC": "2015-01-01..2025-03-19, as the paper",
    "ETH": "starts 2015-08-08 (Poloniex listing), paper 2015-08-01",
}
REPRO_SOURCES = {
    "BTC": f"Bitstamp btcusd daily close, candle opening 00:00 UTC ({data.BITSTAMP_OHLC})",
    "ETH": f"Poloniex ETH_BTC DAY_1 close ({data.POLONIEX_CANDLES}) x Bitstamp btcusd close, same UTC day",
}
# (name, asset, long_short, gross of costs, paper Sharpe, paper table)
REPRO_LONG_ONLY_CHECKS = (
    ("btc_long_only_net", "BTC", False, False, 1.56, "Table 3"),
    ("eth_long_only_net", "ETH", False, False, 1.51, "Table 3"),
)
REPRO_LONG_SHORT_CHECKS = (
    ("btc_long_short_gross", "BTC", True, True, 1.31, "Table 5 Combo"),
)


def strategy_canonical(params: dict) -> dict:
    """The strategy rule values a reproduction vouches for, with normalised types."""
    return {
        "long_short": bool(params["long_short"]),
        "lookbacks": [int(n) for n in params["lookbacks"]],
        "vol_target": float(params["vol_target"]),
        "vol_window": int(params["vol_window"]),
        "per_coin_cap_repro": float(params["per_coin_cap_repro"]),
    }


def strategy_sha256(params: dict) -> str:
    """sha256 of the canonical JSON (sorted keys, no spaces) of strategy_canonical."""
    canon = json.dumps(strategy_canonical(params), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def repro_check(target: float, ours: float) -> dict:
    """{target, ours, rel_error, passed}: rel_error = |ours - target| / |target|,
    passed when rel_error <= REPRO_MAX_REL_ERROR."""
    rel = abs(ours - target) / abs(target)
    return {"target": float(target), "ours": float(ours), "rel_error": float(rel), "passed": bool(rel <= REPRO_MAX_REL_ERROR)}


def repro_returns(
    close: pd.Series,
    *,
    long_short: bool,
    lookbacks,
    vol_target: float,
    vol_window: int,
    per_coin_cap: float,
    cost_per_side: float,
    rebalance_threshold: float | None,
) -> pd.Series:
    """Daily returns of the single-asset Combo on ``close`` (spot: no funding).

    The asset is the whole universe every month and the gross cap is the
    per-coin cap. Indicators need their history inside ``close``: until a
    lookback's channel (or the vol window) is full it holds no position, so
    early returns are flat. Returns run from the second day to the last.
    """
    frame = close.astype(float).to_frame(close.name or "ASSET")
    col = frame.columns[0]
    months = frame.index.to_period("M").unique()
    universe = {(m - 1).to_timestamp(how="end").normalize().date(): [col] for m in months}
    weights, changed = portfolio_weights(
        frame, universe, long_short, per_coin_cap, per_coin_cap, None,
        lookbacks=tuple(lookbacks), vol_target=vol_target, vol_window=vol_window,
    )
    return net_returns(weights, changed, frame, None, cost_per_side, rebalance_threshold)


def check_repro_pass(root: Path, rd: Path, prereg: dict) -> dict:
    """Refuse (ReproGateError) unless results/{id}-repro.json is committed and
    unmodified, is this candidate's repro result, says passed: true, and its
    strategy_sha256 equals the one computed from ``prereg``.
    Returns {"path", "strategy_sha256"} for the manifest."""
    cid = prereg["candidate_id"]
    root = Path(root).resolve()
    f = Path(rd).resolve() / f"{cid}-repro.json"
    rel = f.relative_to(root).as_posix()
    if not f.exists():
        raise ReproGateError(
            f"no reproduction result {rel}: run `python -m judge reproduce prereg/drafts/{cid}.json` "
            "and commit its result before judging"
        )
    _committed_blob(root, f, ReproGateError)
    try:
        res = json.loads(f.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ReproGateError(f"{rel} is not valid JSON: {exc}") from exc
    if not isinstance(res, dict) or res.get("kind") != "repro" or res.get("candidate_id") != cid:
        raise ReproGateError(f"{rel} is not the repro result of {cid}")
    if res.get("passed") is not True:
        raise ReproGateError(f"{rel} says passed: false; {cid} did not reproduce the paper and is not judged")
    expected = strategy_sha256(prereg)
    if res.get("strategy_sha256") != expected:
        raise ReproGateError(
            f"{rel} reproduced other rules: strategy_sha256 {res.get('strategy_sha256')} vs this prereg's {expected}"
        )
    return {"path": rel, "strategy_sha256": expected}


def reproduce(
    prereg_path: Path,
    *,
    loader=exchange_daily,
    cache_dir: Path | None = None,
    results_dir: Path | None = None,
) -> dict:
    """Reproduce the paper's single-asset Sharpe ratios for a prereg draft and
    write results/{candidate_id}-repro.json (pass or fail; never overwritten).

    Checks (REPRO costs: 10 bps per side, 20% threshold, no funding):
      long-only Combo, net: BTC 2015-01-01..2025-03-19 vs 1.56, ETH
      2015-08-08..2025-03-19 vs 1.51 (Table 3). A long-short candidate adds
      long-short BTC gross of costs vs 1.31 (Table 5 Combo).
    Refuses before loading data when the draft's candidate_id is not its file
    name, its paper_sample_end is not 2025-03-19, the draft is not committed
    as is, or judge/ has uncommitted changes (the recorded judge tree hash
    must be the code that ran).
    """
    path = Path(prereg_path).resolve()
    params = load_prereg_params(path)
    cid = params["candidate_id"]
    if cid != path.stem:
        raise PreregError(f"{path.name}: candidate_id {cid!r} must equal the file name {path.stem!r}")
    if _paper_end(params, path.name) != REPRO_SAMPLE_END:
        raise PreregError(
            f"{path.name}: paper_sample_end {params['paper_sample_end']} is not {REPRO_SAMPLE_END}, "
            "the sample the reproduction targets come from"
        )
    root = _repo_root(path.parent)
    if root != Path(CODE_ROOT).resolve():
        raise PreregMismatch(f"judge code runs from {Path(CODE_ROOT).resolve()} but {path.name} is in {root}")
    results_dir = Path(results_dir) if results_dir is not None else root / "results"
    cache_dir = Path(cache_dir) if cache_dir is not None else root / "data_cache" / "repro"
    out = results_dir / f"{cid}-repro.json"
    if out.exists():
        raise ResultExists(f"{out.name} already exists; results are never overwritten")
    blob = _committed_blob(root, path)
    dirty = _git(root, "status", "--porcelain", "--untracked-files=all", "--", "judge")
    if dirty:
        raise PreregMismatch(f"uncommitted changes under judge/: {', '.join(ln[3:] for ln in dirty.splitlines())}")
    tree_hash = _git(root, "rev-parse", "HEAD:judge")
    git_head = _git(root, "rev-parse", "HEAD")
    freeze = _pip_freeze()
    run_utc = _utc_now()

    with data.record_api_prices() as prices:
        closes = {}
        for asset in ("BTC", "ETH"):
            s = loader(asset, REPRO_START[asset], REPRO_SAMPLE_END, cache_dir)
            first, last = s.index[0].date(), s.index[-1].date()
            if first != REPRO_START[asset] or last != REPRO_SAMPLE_END or s.isna().any():
                raise data.DataIntegrityError(
                    f"{asset}: loaded {first}..{last} (NaN: {bool(s.isna().any())}), "
                    f"need {REPRO_START[asset]}..{REPRO_SAMPLE_END} without gaps"
                )
            closes[asset] = s

    plan = REPRO_LONG_ONLY_CHECKS + (REPRO_LONG_SHORT_CHECKS if params["long_short"] else ())
    strat = strategy_canonical(params)
    per_coin_cap = strat["per_coin_cap_repro"]
    threshold = float(REPRO["rebalance_threshold"])
    checks = {}
    for name, asset, ls, gross, target, table in plan:
        cost = 0.0 if gross else float(REPRO["cost_per_side"])
        r = repro_returns(
            closes[asset], long_short=ls, lookbacks=strat["lookbacks"], vol_target=strat["vol_target"],
            vol_window=strat["vol_window"], per_coin_cap=per_coin_cap, cost_per_side=cost,
            rebalance_threshold=threshold,
        )
        c = repro_check(target, sharpe(r))
        c.update({
            "asset": asset, "long_short": ls, "gross_of_costs": gross, "cost_per_side": cost,
            "rebalance_threshold": threshold, "paper_table": table,
            "period": {"start": REPRO_START[asset].isoformat(), "end": REPRO_SAMPLE_END.isoformat()},
            "scored_days": int(len(r)), "total_return": _total_return(r), "max_dd": max_drawdown(r),
        })
        checks[name] = c

    result = {
        "candidate_id": cid,
        "kind": "repro",
        "passed": bool(all(c["passed"] for c in checks.values())),
        "checks": checks,
        "max_rel_error": REPRO_MAX_REL_ERROR,
        "period_notes": dict(REPRO_PERIOD_NOTES),
        "strategy_sha256": strategy_sha256(params),
        "strategy": strat,
        "settings": {
            "cost_per_side": float(REPRO["cost_per_side"]), "rebalance_threshold": threshold,
            "gross_cap": per_coin_cap, "per_coin_cap": per_coin_cap, "funding": "none (spot)",
            "gross_checks_cost_per_side": 0.0,
        },
        "sources": dict(REPRO_SOURCES),
        "manifest": {
            "draft_file": path.relative_to(root).as_posix(),
            "draft_sha256": hashlib.sha256(blob).hexdigest(),
            "judge_tree_hash": tree_hash,
            "git_head": git_head,
            "run_utc": run_utc,
            "api_price_sha256": dict(sorted(prices.items())),
            "python": sys.version,
            "pip_freeze": freeze,
        },
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(out, "x", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(result, indent=2) + "\n")
    except FileExistsError as exc:
        raise ResultExists(f"{out.name} already exists; results are never overwritten") from exc
    return result


# ---------------------------------------------------------------- run

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_data(market: str, start: date, end: date, cache_dir: Path):
    """(close, quote_volume, funding) for every listed symbol. Tests replace this."""
    symbols = data.list_symbols(market)
    close, qv, funding = data.panel(symbols, market, start, end, cache_dir)
    close.attrs["listed_symbols"] = list(symbols)
    return close, qv, funding


def _pip_freeze() -> list[str]:
    res = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"pip freeze failed: {res.stderr.strip()}")
    return [ln for ln in res.stdout.splitlines() if ln.strip()]


def archive_checksums(paths, cache_dir: Path) -> dict[str, str]:
    """{path relative to cache_dir: sha256} from each file's cached .CHECKSUM."""
    cache_dir = Path(cache_dir).resolve()
    out: dict[str, str] = {}
    for p in paths:
        side = data.checksum_path(p)
        if not side.exists():
            raise data.DataIntegrityError(f"no cached checksum for {p}")
        parts = side.read_text().split()
        key = Path(p).resolve().relative_to(cache_dir).as_posix()
        out[key] = parts[0].lower() if parts else ""
    return dict(sorted(out.items()))


def _data_last_date(close: pd.DataFrame) -> date:
    last = close.attrs.get("last_date") or {}
    if last:
        return max(pd.Timestamp(v) for v in last.values()).date()
    valid = close.dropna(how="all")
    if valid.empty:
        raise DataNotReady("no data loaded")
    return valid.index[-1].date()


def _monthly_universe(close, qv, size: int) -> dict[date, list[str]]:
    month_ends = [ts for ts in close.index if (ts + pd.Timedelta(days=1)).day == 1]
    return {
        ts.date(): data.top_universe(close, qv, ts.date(), size, UNIVERSE_MIN_AGE_DAYS, UNIVERSE_MIN_MEDIAN_VOLUME)
        for ts in month_ends
    }


def _daily_exclusions(close, qv, win: Window) -> dict[date, set[str]]:
    """exit_flags for every decision day of the window (start - 1 .. end)."""
    first = pd.Timestamp(win[0]) - pd.Timedelta(days=1)
    last = pd.Timestamp(win[1])
    out: dict[date, set[str]] = {}
    for i, ts in enumerate(close.index):
        if ts < first or ts > last:
            continue
        lo = max(0, i - EXIT_ROWS + 1)  # exit_flags reads only the last 31 closes
        flagged = data.exit_flags(close.iloc[lo:i + 1], qv.iloc[lo:i + 1], ts.date())
        if flagged:
            out[ts.date()] = set(flagged)
    return out


def _total_return(r: pd.Series) -> float:
    return float(np.prod(1.0 + r.to_numpy(dtype=float)) - 1.0) if len(r) else 0.0


def run(
    prereg_path: Path,
    kind: Kind,
    *,
    stamp_confirmed: date | None = None,
    cache_dir: Path | None = None,
    results_dir: Path | None = None,
) -> Path:
    """Judge one candidate on one window and write results/{id}-{kind}.json.

    Order: prereg integrity, kind checks, window, the reproduction gate (test
    and final_alone: check_repro_pass), manifest inputs that can fail,
    then (Final only) check_final_opened, which needs the marker written by
    ``open-final`` and committed, and only then the data load. The result file
    is written for passes and failures, and never overwritten. Costs, threshold,
    gross cap, sims and universe size are the judge's JUDGING constants (the
    prereg must state the same values). stamp_confirmed is the
    --stamp-confirmed flag, which may only confirm the committed file.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    path = Path(prereg_path).resolve()
    prereg = load_prereg(path)
    check_integrity(path)
    root = _repo_root(path.parent)
    if root != Path(CODE_ROOT).resolve():
        raise PreregMismatch(f"judge code runs from {Path(CODE_ROOT).resolve()} but {path.name} is in {root}")
    cid = prereg["candidate_id"]
    results_dir = Path(results_dir) if results_dir is not None else root / "results"
    cache_dir = Path(cache_dir) if cache_dir is not None else root / "data_cache"
    out = results_dir / f"{cid}-{kind}.json"
    if out.exists():
        raise ResultExists(f"{out.name} already exists; results are never overwritten")

    if kind in FINAL_KINDS:
        stamp, stamp_source, win = _final_context(path, prereg, kind, stamp_confirmed)
    else:
        stamp, stamp_source, win = None, None, window(kind, prereg, None)
    # Test and Final-alone judge only rules that reproduced the paper (Final
    # relies on the Test pass, which already needed this).
    repro = check_repro_pass(root, results_dir, prereg) if kind in ("test", "final_alone") else None
    prereg_sha = committed_sha256(path)
    tree_hash = _git(root, "rev-parse", "HEAD:judge")
    git_head = _git(root, "rev-parse", "HEAD")
    freeze = _pip_freeze()
    run_utc = _utc_now()
    results_status = [ln for ln in _git(root, "status", "--porcelain", "--untracked-files=all", "--", "results").splitlines() if ln]

    final_marker = None
    if kind in FINAL_KINDS:
        final_marker = check_final_opened(cid, prereg_sha256=prereg_sha, judge_tree_hash=tree_hash, window=win,
                                          final_alone=kind == "final_alone", results_dir=results_dir)

    load_start = win[0] - timedelta(days=LOOKBACK_DAYS)
    with data.record_fetches() as fetched, data.record_api_funding() as api_funding:
        close, qv, funding = _load_data(MARKET, load_start, win[1], cache_dir)
    last = _data_last_date(close)
    if last < win[1]:
        err = FinalNotReady if kind in FINAL_KINDS else DataNotReady
        raise err(f"data ends {last}, before the window end {win[1]}")
    checksums = archive_checksums(fetched, cache_dir)
    skipped = list(close.attrs.get("skipped", []))
    listed = close.attrs.get("listed_symbols")
    listed = list(listed) if listed is not None else None

    universe = _monthly_universe(close, qv, JUDGING["universe_size"])
    ever = set().union(*universe.values()) if universe else set()
    members = [c for c in close.columns if c in ever]  # other coins never get weight
    close, qv = close[members], qv[members]
    funding = funding.reindex(columns=members) if funding is not None else None
    excluded = _daily_exclusions(close, qv, win)

    long_short = prereg["long_short"]
    lookbacks = tuple(prereg["lookbacks"])
    vol_target, vol_window = float(prereg["vol_target"]), int(prereg["vol_window"])
    per_coin_cap = float(prereg["per_coin_cap_repro"])
    gross_cap = float(JUDGING["gross_cap"])
    cost = float(JUDGING["cost_per_side"])
    threshold = float(JUDGING["rebalance_threshold"])
    seed, n_sims = prereg["seed"], JUDGING["n_sims"]

    weights, changed = portfolio_weights(
        close, universe, long_short, per_coin_cap, gross_cap, excluded,
        lookbacks=lookbacks, vol_target=vol_target, vol_window=vol_window,
    )
    strat = window_slice(net_returns(start_flat(weights, win), changed, close, funding, cost, threshold), win)
    bench = benchmark_returns(close, funding, universe, cost, excluded=excluded, window=win)
    models = observed_path_models(close, universe, excluded, lookbacks, long_short, win)
    flips, probs = sim_path_args(models, long_short)
    odds = luck_odds(
        close, funding, universe, long_short, flips, n_sims, seed, kind, cost, gross_cap, bench=bench,
        state_probs=probs, rebalance_threshold=threshold, lookbacks=lookbacks, per_coin_cap=per_coin_cap,
        vol_target=vol_target, vol_window=vol_window, excluded=excluded, window=win,
    )
    v = verdict(strat, bench, kind)

    win_json = _window_json(win)
    result = {
        "candidate_id": cid,
        "kind": kind,
        "passed": v["passed"],
        "verdict": v,
        "luck_odds": odds,
        "window": win_json,
        "scored_days": int(len(strat)),
        "total_return": _total_return(strat),
        "bench_total_return": _total_return(bench),
        "manifest": {
            "prereg_file": path.relative_to(root).as_posix(),
            "prereg_sha256": prereg_sha,
            "judge_tree_hash": tree_hash,
            "git_head": git_head,
            "seed": seed,
            "n_sims": n_sims,
            "run_utc": run_utc,
            "window": win_json,
            "stamp_confirmed": stamp.isoformat() if stamp else None,
            "stamp_source": stamp_source,
            "final_marker": final_marker,
            "repro": repro,
            "results_git_status": results_status,
            "market": MARKET,
            "data_loaded": {"start": load_start.isoformat(), "end": win[1].isoformat()},
            "data_last_date": last.isoformat(),
            "listed_symbols": listed,
            "skipped_symbols": skipped,
            "universe_members": members,
            "universe_by_month": {d.isoformat(): coins for d, coins in sorted(universe.items())},
            "excluded": {d.isoformat(): sorted(coins) for d, coins in sorted(excluded.items())},
            "settings": {
                "per_coin_cap": per_coin_cap, "gross_cap": gross_cap, "cost_per_side": cost,
                "rebalance_threshold": threshold, "universe_size": JUDGING["universe_size"],
                "universe_min_age_days": UNIVERSE_MIN_AGE_DAYS,
                "universe_min_median_volume": UNIVERSE_MIN_MEDIAN_VOLUME,
                "lookbacks": list(lookbacks), "vol_target": vol_target, "vol_window": vol_window,
                "long_short": long_short,
            },
            "luck_odds_method": LUCK_ODDS_METHOD,
            "luck_odds_paths": models,
            "archive_checksums": checksums,
            "api_funding_sha256": {
                sym: data.funding_rows_sha256(sym, rows) for sym, rows in sorted(api_funding.items())
            },
            "python": sys.version,
            "pip_freeze": freeze,
        },
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(out, "x", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(result, indent=2) + "\n")
    except FileExistsError as exc:
        raise ResultExists(f"{out.name} already exists; results are never overwritten") from exc
    return out
