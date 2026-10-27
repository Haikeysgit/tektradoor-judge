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
    "combo averages its lookback signals. A path starts at 0; each day, with probability "
    "flip_prob, it redraws its state uniformly from {0,1} (long-only) or {-1,0,1} (long-short), "
    "possibly the same state. The target rate is the candidate's own signal change rate: changes "
    "summed over lookbacks divided by (lookbacks x active coin-days) inside the window. A redraw "
    "keeps the state with probability 1/k (k states), so flip_prob = rate x k/(k-1), capped at 1, "
    "which makes each path change state at the candidate's observed rate. Coin "
    "weight = vol_scale(vol_target, vol_window, per_coin_cap) x mean state, then the "
    "candidate's allocation (monthly universe, daily exit exclusions, delisting, gross cap), "
    "its rebalance threshold (a path state change counts as a signal change), costs and "
    "funding. The book starts flat at the window start and only window returns are scored. "
    "The rng is numpy default_rng(seed)."
)


def _random_states(rng: np.random.Generator, shape: tuple[int, int], flip_prob: float, states: np.ndarray) -> np.ndarray:
    """State paths: start 0; each day with prob flip_prob draw a new state
    uniformly from ``states`` (it may equal the current one), else keep."""
    t_len, n_coins = shape
    flips = rng.random(shape) < flip_prob
    draws = rng.choice(states, size=shape)
    last = np.where(flips, np.arange(t_len)[:, None], -1)
    last = np.maximum.accumulate(last, axis=0)
    picked = np.take_along_axis(draws, np.clip(last, 0, None), axis=0)
    return np.where(last >= 0, picked, 0).astype(float)


def luck_sims(
    close: pd.DataFrame,
    funding: pd.DataFrame | None,
    universe_by_month: dict,
    long_short: bool,
    flip_prob: float,
    n_sims: int,
    seed: int,
    cost_per_side: float,
    gross_cap: float | None,
    *,
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
    returns, or (for tests) each sim's mean state or target weight frame."""
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")
    n_paths = len(lookbacks)
    if n_paths == 0:
        raise ValueError("lookbacks must not be empty")
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
        for _ in range(n_paths):
            path = _random_states(rng, shape, flip_prob, states)
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
    flip_prob: float,
    n_sims: int,
    seed: int,
    kind: Kind,
    cost_per_side: float,
    gross_cap: float | None,
    bench: pd.Series | None = None,
    *,
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
        rebalance_threshold=rebalance_threshold, lookbacks=lookbacks, per_coin_cap=per_coin_cap,
        vol_target=vol_target, vol_window=vol_window, excluded=excluded, window=window,
    )
    return sum(verdict(s, bench, kind)["passed"] for s in sims) / n_sims


def flip_prob_for_rate(rate: float, n_states: int) -> float:
    """flip_prob at which a _random_states path changes state at ``rate``.

    A flip redraws uniformly from n_states states, so it changes the state
    with probability (k-1)/k; hence flip_prob = rate * k/(k-1), capped at 1.
    """
    if n_states < 2:
        return 0.0
    return float(min(1.0, rate * n_states / (n_states - 1)))


def observed_flip_prob(
    close: pd.DataFrame,
    universe_by_month: dict,
    excluded: dict | None,
    lookbacks,
    long_short: bool,
    window: Window,
) -> float:
    """flip_prob that makes each sim path change state as often as one of the
    candidate's lookback signals did: the mean daily change rate per lookback
    per active coin-day inside the window (decision days start..end), through
    flip_prob_for_rate. 0 if no coin-day is active."""
    active, _, _ = _allocation_frame(close, universe_by_month, excluded)
    idx = close.index
    in_win = np.asarray((idx >= pd.Timestamp(window[0])) & (idx <= pd.Timestamp(window[1])))
    mask = active.to_numpy(dtype=bool) & in_win[:, None]
    coin_days = int(mask.sum())
    if coin_days == 0 or not len(lookbacks):
        return 0.0
    changes = 0
    for n in lookbacks:
        for j, c in enumerate(close.columns):
            pos = signal(close[c], n, long_short)
            moved = (pos != pos.shift(fill_value=0)).to_numpy(dtype=bool)
            changes += int((moved & mask[:, j]).sum())
    return flip_prob_for_rate(changes / (coin_days * len(lookbacks)), 3 if long_short else 2)


# ---------------------------------------------------------------- pre-registration

REQUIRED_KEYS = (
    "candidate_id", "paper", "long_short", "lookbacks", "vol_target", "vol_window",
    "per_coin_cap_repro", "gross_cap", "cost_per_side", "rebalance_threshold",
    "universe_size", "paper_sample_end", "seed", "n_sims", "judge_tree_hash",
)
CANDIDATE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
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
    """The prereg file (or its stamp_confirmed companion) is unreadable or incomplete."""


class PreregMismatch(JudgeRefusal):
    """A stamped file (prereg or judge code) differs from what was registered."""


class DataNotReady(JudgeRefusal):
    """The data does not cover the window."""


class FinalNotReady(DataNotReady):
    """Final cannot run yet: no confirmed stamp date, or data short of the window."""


class FinalGateError(JudgeRefusal):
    """Final may not be opened (no Test pass, already opened, wrong kind)."""


class ResultExists(JudgeRefusal):
    """A result file for this candidate and kind already exists; never overwritten."""


def load_prereg(path: Path) -> dict:
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
    if not isinstance(raw["candidate_id"], str) or not CANDIDATE_ID_RE.match(raw["candidate_id"]):
        raise PreregError(f"{path.name}: candidate_id must match {CANDIDATE_ID_RE.pattern}: {raw['candidate_id']!r}")
    _paper_end(raw, path.name)
    return raw


def _paper_end(prereg: dict, name: str = "prereg") -> date:
    try:
        return date.fromisoformat(str(prereg["paper_sample_end"]))
    except ValueError as exc:
        raise PreregError(f"{name}: paper_sample_end is not an ISO date: {prereg['paper_sample_end']!r}") from exc


def _git(cwd: Path, *args: str) -> str:
    try:
        res = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    except OSError as exc:
        raise PreregMismatch(f"cannot run git in {cwd}: {exc}") from exc
    if res.returncode != 0:
        raise PreregMismatch(f"git {' '.join(args)} failed in {cwd}: {res.stderr.strip()}")
    return res.stdout.strip()


def _repo_root(path: Path) -> Path:
    return Path(_git(Path(path), "rev-parse", "--show-toplevel")).resolve()


def file_sha256(path: Path) -> str:
    """sha256 of the raw bytes (the repo forces LF, so this is the same on every OS)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_integrity(prereg_path: Path) -> None:
    """Refuse to run unless the prereg and judge code are exactly as stamped.

    1. sha256(prereg) equals the committed companion prereg/NN-name.sha256
       (its first token, so ``sha256sum`` output works).
    2. The prereg's judge_tree_hash equals ``git rev-parse HEAD:judge``.
    3. ``git status`` shows nothing, tracked or untracked, under judge/ or prereg/.
    Each failure raises PreregMismatch naming the file (and both hashes).
    """
    path = Path(prereg_path).resolve()
    root = _repo_root(path.parent)
    sha_file = path.with_suffix(".sha256")
    if not sha_file.exists():
        raise PreregMismatch(f"{path.name}: no companion {sha_file.name} holding the registered sha256")
    parts = sha_file.read_text().split()
    registered = parts[0].lower() if parts else "(empty)"
    actual = file_sha256(path)
    if actual != registered:
        raise PreregMismatch(
            f"{path.name} differs from its registration: sha256 is {actual}, "
            f"registered in {sha_file.name} is {registered}"
        )
    recorded = str(load_prereg(path)["judge_tree_hash"])
    head = _git(root, "rev-parse", "HEAD:judge")
    if recorded != head:
        raise PreregMismatch(
            f"judge/ code differs from {path.name}: git rev-parse HEAD:judge is {head}, "
            f"judge_tree_hash recorded in {path.name} is {recorded}"
        )
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
            "Bitcoin block: commit it as prereg/NN-name.stamp_confirmed (one ISO date line) "
            "or pass --stamp-confirmed YYYY-MM-DD"
        )
    start = stamp_confirmed + timedelta(days=1)
    end = start + timedelta(days=FINAL_DAYS - 1)
    latest = latest_data_date if latest_data_date is not None else _today_utc() - timedelta(days=FINAL_DATA_LAG_DAYS)
    if end > latest:
        raise FinalNotReady(f"Final window {start}..{end} ends after the latest data date {latest}")
    return start, end


def resolve_stamp_confirmed(prereg_path: Path, flag: date | None) -> date | None:
    """The stamp_confirmed date: the committed companion file wins.

    prereg/NN-name.stamp_confirmed (a single ISO date line) is used when it
    exists, because it is committed and so part of the audited record. The
    --stamp-confirmed flag is used only when there is no such file; a flag
    that disagrees with the file is refused rather than silently ignored.
    """
    side = Path(prereg_path).with_suffix(".stamp_confirmed")
    if not side.exists():
        return flag
    lines = [ln.strip() for ln in side.read_text().splitlines() if ln.strip()]
    if len(lines) != 1:
        raise PreregError(f"{side.name} must hold a single ISO date line, found {len(lines)} lines")
    try:
        from_file = date.fromisoformat(lines[0])
    except ValueError as exc:
        raise PreregError(f"{side.name}: not an ISO date: {lines[0]!r}") from exc
    if flag is not None and flag != from_file:
        raise PreregError(f"--stamp-confirmed {flag} conflicts with {side.name} ({from_file}); the file wins, so drop the flag")
    return from_file


def open_final(candidate_id: str, *, final_alone: bool = False, results_dir: Path | None = None) -> None:
    """Record that the Final window was opened. Call BEFORE loading Final data.

    Writes results/{id}.final_opened. Refuses (FinalGateError) if it already
    exists, or, unless the candidate is Final-alone, if the Test result file
    is missing or does not say ``passed: true``.
    """
    rd = Path(results_dir) if results_dir is not None else CODE_ROOT / "results"
    marker = rd / f"{candidate_id}.final_opened"
    if marker.exists():
        raise FinalGateError(f"Final for {candidate_id} was already opened ({marker.name}); it opens once")
    if not final_alone:
        test_file = rd / f"{candidate_id}-test.json"
        if not test_file.exists():
            raise FinalGateError(f"no Test result {test_file.name}: Final opens only after a Test pass")
        passed = json.loads(test_file.read_text()).get("passed")
        if passed is not True:
            raise FinalGateError(f"Test result {test_file.name} says passed: false; Final stays closed")
    rd.mkdir(parents=True, exist_ok=True)
    stamp = {"candidate_id": candidate_id, "final_alone": final_alone, "opened_utc": _utc_now()}
    try:
        with open(marker, "x", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(stamp) + "\n")
    except FileExistsError as exc:
        raise FinalGateError(f"Final for {candidate_id} was already opened ({marker.name}); it opens once") from exc


# ---------------------------------------------------------------- run

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_data(market: str, start: date, end: date, cache_dir: Path):
    """(close, quote_volume, funding) for every listed symbol. Tests replace this."""
    return data.panel(data.list_symbols(market), market, start, end, cache_dir)


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

    Order: prereg integrity, kind checks, window, manifest inputs that can fail,
    then (Final only) open_final, and only then the data load. The result file
    is written for passes and failures, and never overwritten.
    stamp_confirmed is the --stamp-confirmed flag; resolve_stamp_confirmed
    decides between it and the committed companion file.
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

    alone = is_final_alone(prereg)
    if kind == "final" and alone:
        raise FinalGateError(f"{cid} has a Test window under 12 months: it is Final-alone, run --kind final_alone")
    if kind == "final_alone" and not alone:
        raise FinalGateError(f"{cid} has a Test window of 12 months or more: run --kind final after a Test pass, not final_alone")
    stamp = resolve_stamp_confirmed(path, stamp_confirmed) if kind in FINAL_KINDS else None
    win = window(kind, prereg, stamp)
    prereg_sha = file_sha256(path)
    tree_hash = _git(root, "rev-parse", "HEAD:judge")
    git_head = _git(root, "rev-parse", "HEAD")
    freeze = _pip_freeze()
    run_utc = _utc_now()

    if kind in FINAL_KINDS:
        open_final(cid, final_alone=kind == "final_alone", results_dir=results_dir)

    load_start = win[0] - timedelta(days=LOOKBACK_DAYS)
    with data.record_fetches() as fetched, data.record_api_funding() as api_funding:
        close, qv, funding = _load_data(MARKET, load_start, win[1], cache_dir)
    last = _data_last_date(close)
    if last < win[1]:
        err = FinalNotReady if kind in FINAL_KINDS else DataNotReady
        raise err(f"data ends {last}, before the window end {win[1]}")
    checksums = archive_checksums(fetched, cache_dir)

    universe = _monthly_universe(close, qv, int(prereg["universe_size"]))
    ever = set().union(*universe.values()) if universe else set()
    members = [c for c in close.columns if c in ever]  # other coins never get weight
    close, qv = close[members], qv[members]
    funding = funding.reindex(columns=members) if funding is not None else None
    excluded = _daily_exclusions(close, qv, win)

    long_short = bool(prereg["long_short"])
    lookbacks = tuple(int(n) for n in prereg["lookbacks"])
    vol_target, vol_window = float(prereg["vol_target"]), int(prereg["vol_window"])
    per_coin_cap = float(prereg["per_coin_cap_repro"])
    gross_cap = None if prereg["gross_cap"] is None else float(prereg["gross_cap"])
    cost = float(prereg["cost_per_side"])
    threshold = None if prereg["rebalance_threshold"] is None else float(prereg["rebalance_threshold"])
    seed, n_sims = int(prereg["seed"]), int(prereg["n_sims"])

    weights, changed = portfolio_weights(
        close, universe, long_short, per_coin_cap, gross_cap, excluded,
        lookbacks=lookbacks, vol_target=vol_target, vol_window=vol_window,
    )
    strat = window_slice(net_returns(start_flat(weights, win), changed, close, funding, cost, threshold), win)
    bench = benchmark_returns(close, funding, universe, cost, excluded=excluded, window=win)
    flip = observed_flip_prob(close, universe, excluded, lookbacks, long_short, win)
    odds = luck_odds(
        close, funding, universe, long_short, flip, n_sims, seed, kind, cost, gross_cap, bench=bench,
        rebalance_threshold=threshold, lookbacks=lookbacks, per_coin_cap=per_coin_cap,
        vol_target=vol_target, vol_window=vol_window, excluded=excluded, window=win,
    )
    v = verdict(strat, bench, kind)

    win_json = {"start": win[0].isoformat(), "end": win[1].isoformat()}
    result = {
        "candidate_id": cid,
        "kind": kind,
        "passed": v["passed"],
        "verdict": v,
        "luck_odds": odds,
        "luck_odds_flip_prob": flip,
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
            "market": MARKET,
            "data_loaded": {"start": load_start.isoformat(), "end": win[1].isoformat()},
            "data_last_date": last.isoformat(),
            "skipped_symbols": list(close.attrs.get("skipped", [])),
            "universe_members": members,
            "settings": {
                "per_coin_cap": per_coin_cap, "gross_cap": gross_cap, "cost_per_side": cost,
                "rebalance_threshold": threshold, "universe_size": int(prereg["universe_size"]),
                "universe_min_age_days": UNIVERSE_MIN_AGE_DAYS,
                "universe_min_median_volume": UNIVERSE_MIN_MEDIAN_VOLUME,
                "lookbacks": list(lookbacks), "vol_target": vol_target, "vol_window": vol_window,
                "long_short": long_short,
            },
            "luck_odds_method": LUCK_ODDS_METHOD,
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
