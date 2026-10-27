"""Paper reproduction (Task 8): SSRN 5209907's single-asset Sharpe ratios,
recomputed with the paper's own costs before any candidate is judged.

Everything that decides a reproduction number lives in this module,
metrics.py, prices.py, costs.py and strategies/. The Test-time gate
(judge.check_repro_pass) requires those files to be byte-identical to the
judge tree that ran the reproduction, and recomputes every check's verdict
from the result file with the constants here. A miss of more than
REPRO_REL_ERROR_LIMIT on any check fails the reproduction; nobody tunes.

This module imports judge.judge only inside functions (judge.judge
re-exports the names here).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from judge import data, prices
from judge.costs import net_returns
from judge.metrics import max_drawdown, sharpe, total_return
from judge.prices import exchange_daily
from judge.strategies.donchian import portfolio_weights

# The paper's own 10 bps per side and 20% rebalance threshold. run() never
# uses these; it judges with judge.JUDGING.
REPRO = {
    "cost_per_side": 0.0010,
    "rebalance_threshold": 0.2,
}
REPRO_REL_ERROR_LIMIT = 0.20
REPRO_MAX_REL_ERROR = REPRO_REL_ERROR_LIMIT  # earlier name
REPRO_SAMPLE_END = date(2025, 3, 19)  # SSRN 5209907 sample end
REPRO_START = {"BTC": date(2015, 1, 1), "ETH": date(2015, 8, 8)}
REPRO_PERIOD_NOTES = {
    "BTC": "2015-01-01..2025-03-19, as the paper",
    "ETH": "starts 2015-08-08 (Poloniex listing), paper 2015-08-01",
}
REPRO_SOURCES = {
    "BTC": f"Bitstamp btcusd daily close, candle opening 00:00 UTC ({prices.BITSTAMP_OHLC})",
    "ETH": f"Poloniex ETH_BTC DAY_1 close ({prices.POLONIEX_CANDLES}) x Bitstamp btcusd close, same UTC day",
}
# (name, asset, long_short, gross of costs, paper Sharpe, paper table)
REPRO_LONG_ONLY_CHECKS = (
    ("btc_long_only_net", "BTC", False, False, 1.56, "Table 3"),
    ("eth_long_only_net", "ETH", False, False, 1.51, "Table 3"),
)
REPRO_LONG_SHORT_CHECKS = (
    ("btc_long_short_gross", "BTC", True, True, 1.31, "Table 5 Combo"),
)
# Canonical price series used by a reproduction are committed here, under results/.
PRICE_DATA_DIR = "repro-data"
_PRICE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def repro_plan(long_short: bool) -> tuple:
    """The checks a candidate must pass: the long-only Combo core for every
    candidate, plus the long-short gross check for a long-short one."""
    return REPRO_LONG_ONLY_CHECKS + (REPRO_LONG_SHORT_CHECKS if long_short else ())


def _expected_fields(asset: str, ls: bool, gross: bool, target: float) -> dict:
    return {
        "target": target,
        "asset": asset,
        "long_short": ls,
        "gross_of_costs": gross,
        "cost_per_side": 0.0 if gross else float(REPRO["cost_per_side"]),
        "rebalance_threshold": float(REPRO["rebalance_threshold"]),
        "period": {"start": REPRO_START[asset].isoformat(), "end": REPRO_SAMPLE_END.isoformat()},
    }


def load_prereg_params(path: Path) -> dict:
    """Read a prereg draft (prereg/drafts/NN-name.json) for reproduction.

    The same key and type validation as load_prereg, and NO integrity check:
    a draft is not registered yet and cannot carry the final judge_tree_hash.
    """
    from judge import judge as J
    return J.load_prereg(path)


def strategy_canonical(params: dict) -> dict:
    """The rule values a reproduction vouches for, with normalised types.
    paper_sample_end is included: it fixes the Test window's start."""
    return {
        "long_short": bool(params["long_short"]),
        "lookbacks": [int(n) for n in params["lookbacks"]],
        "vol_target": float(params["vol_target"]),
        "vol_window": int(params["vol_window"]),
        "per_coin_cap_repro": float(params["per_coin_cap_repro"]),
        "paper_sample_end": date.fromisoformat(str(params["paper_sample_end"])).isoformat(),
    }


def strategy_sha256(params: dict) -> str:
    """sha256 of the canonical JSON (sorted keys, no spaces) of strategy_canonical."""
    canon = json.dumps(strategy_canonical(params), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def repro_check(target: float, ours: float) -> dict:
    """{target, ours, rel_error, passed}: rel_error = |ours - target| / |target|,
    passed when rel_error <= REPRO_REL_ERROR_LIMIT."""
    rel = abs(ours - target) / abs(target)
    return {"target": float(target), "ours": float(ours), "rel_error": float(rel),
            "passed": bool(rel <= REPRO_REL_ERROR_LIMIT)}


def verify_checks(res: dict, long_short: bool) -> list[str]:
    """Problems found recomputing a repro result's verdict (empty: a pass).

    The checks must be exactly repro_plan(long_short); each must carry this
    module's target, asset, long_short, gross flag, cost, threshold and
    period; its rel_error must equal |ours - target| / |target| recomputed,
    and be within REPRO_REL_ERROR_LIMIT. The recorded flags are not trusted.
    """
    checks = res.get("checks")
    if not isinstance(checks, dict):
        return ["no checks"]
    plan = repro_plan(long_short)
    problems = []
    names = sorted(p[0] for p in plan)
    if sorted(checks) != names:
        problems.append(f"checks {sorted(checks)} are not the expected {names}")
    if res.get("rel_error_limit") != REPRO_REL_ERROR_LIMIT:
        problems.append(f"rel_error_limit is {res.get('rel_error_limit')!r}, the judge's is {REPRO_REL_ERROR_LIMIT}")
    for name, asset, ls, gross, target, _ in plan:
        c = checks.get(name)
        if not isinstance(c, dict):
            continue
        for key, want in _expected_fields(asset, ls, gross, target).items():
            if c.get(key) != want or type(c.get(key)) is not type(want):
                problems.append(f"{name}.{key} is {c.get(key)!r}, expected {want!r}")
        ours = c.get("ours")
        if not isinstance(ours, (int, float)) or isinstance(ours, bool) or not math.isfinite(ours):
            problems.append(f"{name}.ours is {ours!r}, not a number")
            continue
        rel = abs(ours - target) / abs(target)
        recorded = c.get("rel_error")
        if not isinstance(recorded, (int, float)) or isinstance(recorded, bool) or abs(recorded - rel) > 1e-12:
            problems.append(f"{name}.rel_error is {recorded!r}, recomputed {rel!r}")
        if rel > REPRO_REL_ERROR_LIMIT:
            problems.append(f"{name}: ours {ours} misses target {target} by {rel:.4f} > {REPRO_REL_ERROR_LIMIT}")
    return problems


def price_data_name(key: str) -> str:
    """File stem under results/repro-data/ for an api_price_sha256 key:
    usd:BTC:2015-01-01..2025-03-19 -> BTC-2015-01-01..2025-03-19,
    bitstamp:btcusd:<range> -> bitstamp-btcusd-<range>."""
    stem = key[len("usd:"):] if key.startswith("usd:") else key
    stem = stem.replace(":", "-")
    if not _PRICE_NAME_RE.match(stem) or stem.startswith("."):
        raise ValueError(f"price series name {key!r} does not make a safe file name")
    return stem


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


def _write_price_data(results_dir: Path, rec: prices.PriceRecord) -> dict[str, str]:
    """Write each recorded series' canonical CSV to results/repro-data/<name>.csv.
    An existing file must hold exactly the same bytes (the same series from an
    earlier reproduction); otherwise DataIntegrityError. Returns {key: file name}."""
    out_dir = Path(results_dir) / PRICE_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for key in sorted(rec):
        name = f"{price_data_name(key)}.csv"
        f = out_dir / name
        blob = rec.csv[key].encode("utf-8")
        if f.exists():
            if f.read_bytes() != blob:
                raise data.DataIntegrityError(f"{f} exists with different content for {key}; refusing to overwrite")
        else:
            with open(f, "xb") as fh:
                fh.write(blob)
        written[key] = f"results/{PRICE_DATA_DIR}/{name}"
    return written


def reproduce(
    prereg_path: Path,
    *,
    loader=exchange_daily,
    cache_dir: Path | None = None,
    results_dir: Path | None = None,
) -> dict:
    """Reproduce the paper's single-asset Sharpe ratios for a prereg draft and
    write results/{candidate_id}-repro.json (pass or fail; never overwritten),
    plus the canonical CSV of every price series used, under results/repro-data/.

    Checks (REPRO costs: 10 bps per side, 20% threshold, no funding):
      long-only Combo, net: BTC 2015-01-01..2025-03-19 vs 1.56, ETH
      2015-08-08..2025-03-19 vs 1.51 (Table 3). A long-short candidate adds
      long-short BTC gross of costs vs 1.31 (Table 5 Combo).
    Refuses before loading data when the draft's candidate_id is not its file
    name, its paper_sample_end is not 2025-03-19, the draft is not committed
    as is, or judge/ has uncommitted changes (the recorded judge tree hash
    must be the code that ran).
    """
    from judge import judge as J

    path = Path(prereg_path).resolve()
    params = load_prereg_params(path)
    cid = params["candidate_id"]
    if cid != path.stem:
        raise J.PreregError(f"{path.name}: candidate_id {cid!r} must equal the file name {path.stem!r}")
    if J._paper_end(params, path.name) != REPRO_SAMPLE_END:
        raise J.PreregError(
            f"{path.name}: paper_sample_end {params['paper_sample_end']} is not {REPRO_SAMPLE_END}, "
            "the sample the reproduction targets come from"
        )
    root = J._repo_root(path.parent)
    if root != Path(J.CODE_ROOT).resolve():
        raise J.PreregMismatch(f"judge code runs from {Path(J.CODE_ROOT).resolve()} but {path.name} is in {root}")
    results_dir = Path(results_dir) if results_dir is not None else root / "results"
    cache_dir = Path(cache_dir) if cache_dir is not None else root / "data_cache" / "repro"
    out = results_dir / f"{cid}-repro.json"
    if out.exists():
        raise J.ResultExists(f"{out.name} already exists; results are never overwritten")
    blob = J._committed_blob(root, path)
    dirty = J._git(root, "status", "--porcelain", "--untracked-files=all", "--", "judge")
    if dirty:
        raise J.PreregMismatch(f"uncommitted changes under judge/: {', '.join(ln[3:] for ln in dirty.splitlines())}")
    tree_hash = J._git(root, "rev-parse", "HEAD:judge")
    git_head = J._git(root, "rev-parse", "HEAD")
    freeze = J._pip_freeze()
    run_utc = J._utc_now()

    with prices.record_api_prices() as rec:
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

    strat = strategy_canonical(params)
    per_coin_cap = strat["per_coin_cap_repro"]
    threshold = float(REPRO["rebalance_threshold"])
    checks = {}
    for name, asset, ls, gross, target, table in repro_plan(params["long_short"]):
        fields = _expected_fields(asset, ls, gross, target)
        r = repro_returns(
            closes[asset], long_short=ls, lookbacks=strat["lookbacks"], vol_target=strat["vol_target"],
            vol_window=strat["vol_window"], per_coin_cap=per_coin_cap, cost_per_side=fields["cost_per_side"],
            rebalance_threshold=threshold,
        )
        c = repro_check(target, sharpe(r))
        c.update(fields)
        c.update({
            "paper_table": table, "scored_days": int(len(r)), "total_return": total_return(r),
            "max_dd": max_drawdown(r),
        })
        checks[name] = c

    price_files = _write_price_data(results_dir, rec)
    result = {
        "candidate_id": cid,
        "kind": "repro",
        "passed": bool(all(c["passed"] for c in checks.values())),
        "checks": checks,
        "rel_error_limit": REPRO_REL_ERROR_LIMIT,
        "worst_rel_error": max(c["rel_error"] for c in checks.values()),
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
            "api_price_sha256": dict(sorted(rec.items())),
            "price_data": price_files,
            "python": sys.version,
            "pip_freeze": freeze,
        },
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(out, "x", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(result, indent=2) + "\n")
    except FileExistsError as exc:
        raise J.ResultExists(f"{out.name} already exists; results are never overwritten") from exc
    return result
