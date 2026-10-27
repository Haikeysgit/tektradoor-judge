"""Reproduction price loaders (Task 8): keyless exchange APIs.

Moved out of judge/data.py; ``judge.data`` still re-exports every name here.
This module decides the reproduction's input prices, so the Test-time gate
ties it to the reproduction (judge.REPRO_CODE_PATHS).
"""
from __future__ import annotations

import hashlib
import json
import logging
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

from judge import data
from judge.data import DataIntegrityError

log = logging.getLogger(__name__)

# Keyless exchange APIs for the paper reproduction (controller ruling
# 2026-10-27; the CoinGecko free tier only reaches back 365 days):
#   BTC/USD: Bitstamp daily OHLC, close of the candle opening 00:00 UTC.
#   ETH/USD: Poloniex ETH_BTC daily close x Bitstamp BTC/USD close, same UTC day.
# Raw responses are cached as JSON under the cache_dir passed in (the judge
# passes data_cache/repro). Each canonical (date,close) series is reported to
# record_api_prices, and the result manifest stores its sha256.

BITSTAMP_OHLC = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
POLONIEX_CANDLES = "https://api.poloniex.com/markets/ETH_BTC/candles"
BITSTAMP_LIMIT = 1000
POLONIEX_LIMIT = 500
DAY_S = 86_400
DAY_MS = DAY_S * 1000
# Poloniex DAY_1 row: low, high, open, close, amount, quantity, buyTakerAmount,
# buyTakerQuantity, tradeCount, ts, weightedAverage, interval, startTime, closeTime
# (checked against a live response on 2026-10-27; pinned in tests/test_repro.py).
POLONIEX_CLOSE = 3
POLONIEX_INTERVAL = 11
POLONIEX_START_MS = 12
MAX_FFILL_DAYS = 3

class PriceRecord(dict):
    """{series name: price_series_sha256}; ``.csv`` holds {series name: canonical CSV text}."""

    def __init__(self):
        super().__init__()
        self.csv: dict[str, str] = {}


_api_price_recorders: list[PriceRecord] = []


@contextmanager
def record_api_prices():
    """Collect {series name: price_series_sha256} (and, in ``.csv``, the
    canonical CSV text) for every API price series reported (note_api_price)
    inside the block."""
    rec = PriceRecord()
    _api_price_recorders.append(rec)
    try:
        yield rec
    finally:
        _api_price_recorders[:] = [r for r in _api_price_recorders if r is not rec]


def price_series_csv(series: pd.Series) -> str:
    """Canonical CSV of a daily price series: header ``date,close``, then one
    ``YYYY-MM-DD,repr(float(close))`` line per day in date order, LF line
    ends, trailing LF. repr round-trips the float exactly."""
    s = pd.Series(series, dtype=float).sort_index()
    lines = ["date,close"] + [f"{pd.Timestamp(d).date().isoformat()},{float(v)!r}" for d, v in s.items()]
    return "\n".join(lines) + "\n"


def price_series_sha256(series: pd.Series) -> str:
    """sha256 of price_series_csv(series), UTF-8."""
    return hashlib.sha256(price_series_csv(series).encode("utf-8")).hexdigest()


def note_api_price(name: str, series: pd.Series) -> None:
    """Report a price series to every active record_api_prices block.

    Names must be unique per series (exchange_daily puts the date range in
    them); the same name with a different hash raises DataIntegrityError
    rather than overwrite the first one.
    """
    if not _api_price_recorders:
        return
    csv = price_series_csv(series)
    sha = hashlib.sha256(csv.encode("utf-8")).hexdigest()
    for rec in _api_price_recorders:
        if rec.get(name, sha) != sha:
            raise DataIntegrityError(f"api price series {name} reported twice with different hashes")
        rec[name] = sha
        rec.csv[name] = csv


def _utc_s(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def _cached_json(url: str, params: dict, path: Path, cacheable: bool):
    """The JSON body of GET url?params; written to path when cacheable (the
    page is complete), and read from path when already cached."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    resp = data._http_get(url, params=params)
    try:
        body = resp.json()
    except ValueError as exc:
        raise data.ArchiveFetchError(url, resp.status_code, f"not JSON: {exc}") from exc
    if cacheable:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_text(json.dumps(body), encoding="utf-8")
        tmp.replace(path)
    return body


def _series(days: list[int], closes: list[float], unit: str, label: str) -> pd.Series:
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(days, dtype="int64"), unit=unit), name="date")
    off = idx != idx.normalize()
    if off.any():
        raise DataIntegrityError(f"{label}: candle at {idx[off][0]} does not open at 00:00 UTC")
    s = pd.Series(closes, index=idx, dtype=float)
    if s.index.has_duplicates:
        for d in s.index[s.index.duplicated()].unique():
            if s.loc[d].nunique() > 1:  # the same day on two pages must agree
                raise DataIntegrityError(f"{label}: conflicting closes for {d.date()}")
        s = s[~s.index.duplicated()]
    return s.sort_index()


def bitstamp_btcusd(start: date, end: date, cache_dir: Path) -> pd.Series:
    """Raw Bitstamp BTC/USD daily closes for [start, end] (gaps not filled).

    Pages of BITSTAMP_LIMIT candles from ``start`` (unix seconds); the next
    page starts the day after the page's last candle. A page is cached only
    when every day it can hold is before today UTC.
    """
    cache = Path(cache_dir) / "bitstamp"
    end_s, cursor = _utc_s(end), _utc_s(start)
    today_s = _utc_s(data._today_utc())
    days: list[int] = []
    closes: list[float] = []
    while cursor <= end_s:
        params = {"step": DAY_S, "limit": BITSTAMP_LIMIT, "start": cursor}
        path = cache / f"btcusd-step{DAY_S}-limit{BITSTAMP_LIMIT}-start{cursor}.json"
        body = _cached_json(BITSTAMP_OHLC, params, path, cursor + BITSTAMP_LIMIT * DAY_S <= today_s)
        try:
            rows = body["data"]["ohlc"]
        except (KeyError, TypeError) as exc:
            raise DataIntegrityError(f"bitstamp btcusd: unexpected response shape at start={cursor}: {exc}") from exc
        if not rows:
            break
        page_days = [int(r["timestamp"]) for r in rows]
        days += page_days
        closes += [float(r["close"]) for r in rows]
        if max(page_days) < cursor:
            raise DataIntegrityError(f"bitstamp btcusd: page at start={cursor} returned only earlier candles")
        if len(rows) < BITSTAMP_LIMIT:
            break
        cursor = max(page_days) + DAY_S
    if not days:
        return pd.Series([], index=pd.DatetimeIndex([], name="date"), dtype=float)
    s = _series(days, closes, "s", "bitstamp btcusd")
    return s.loc[pd.Timestamp(start): pd.Timestamp(end)]


def _parse_poloniex(rows) -> pd.Series:
    """Poloniex DAY_1 rows -> daily close series (field order: POLONIEX_*)."""
    for r in rows:
        if r[POLONIEX_INTERVAL] != "DAY_1":
            raise DataIntegrityError(f"poloniex ETH_BTC: row interval {r[POLONIEX_INTERVAL]!r}, expected DAY_1")
    return _series([int(r[POLONIEX_START_MS]) for r in rows], [float(r[POLONIEX_CLOSE]) for r in rows],
                   "ms", "poloniex ETH_BTC")


def poloniex_ethbtc(start: date, end: date, cache_dir: Path) -> pd.Series:
    """Raw Poloniex ETH_BTC daily closes for [start, end] (gaps not filled).

    Requested in windows of POLONIEX_LIMIT days (startTime..endTime, ms), so a
    window can never hold more rows than the limit. A window is cached only
    when it ends before today UTC.
    """
    cache = Path(cache_dir) / "poloniex"
    end_ms = _utc_s(end) * 1000 + DAY_MS - 1
    today_ms = _utc_s(data._today_utc()) * 1000
    cursor = _utc_s(start) * 1000
    rows: list = []
    while cursor <= end_ms:
        hi = cursor + POLONIEX_LIMIT * DAY_MS - 1
        params = {"interval": "DAY_1", "startTime": cursor, "endTime": hi, "limit": POLONIEX_LIMIT}
        path = cache / f"ETH_BTC-DAY_1-{cursor}-{hi}.json"
        body = _cached_json(POLONIEX_CANDLES, params, path, hi < today_ms)
        if not isinstance(body, list):
            raise DataIntegrityError(f"poloniex ETH_BTC: unexpected response at startTime={cursor}: {str(body)[:200]}")
        rows += body
        cursor = hi + 1
    if not rows:
        return pd.Series([], index=pd.DatetimeIndex([], name="date"), dtype=float)
    s = _parse_poloniex(rows)
    return s.loc[pd.Timestamp(start): pd.Timestamp(end)]


def fill_gaps(series: pd.Series, label: str, max_days: int = MAX_FFILL_DAYS) -> pd.Series:
    """Daily series from its first to its last day, missing days forward-filled.

    A run of more than max_days consecutive missing days raises
    DataIntegrityError naming its first and last day. Each fill is logged.
    """
    s = pd.Series(series, dtype=float).sort_index()
    if s.empty:
        return s
    full = pd.date_range(s.index[0], s.index[-1], freq="D", name="date")
    present = pd.Series(full.isin(s.index), index=full)
    run_id = present.cumsum()  # missing days share the id of the last present day
    for _, run in present[~present].groupby(run_id[~present]):
        a, b, n = run.index[0], run.index[-1], len(run)
        if n > max_days:
            raise DataIntegrityError(
                f"{label}: {n} consecutive missing days {a.date()}..{b.date()}; at most {max_days} may be forward-filled"
            )
        log.warning("gap in %s: %s..%s (%d days) forward-filled", label, a.date(), b.date(), n)
    return s.reindex(full).ffill()


def _covering(series: pd.Series, label: str, start: date, end: date) -> pd.Series:
    """fill_gaps, then require the series to cover start..end."""
    s = fill_gaps(series, label)
    if s.empty:
        raise DataIntegrityError(f"{label}: no data for {start}..{end}")
    first, last = s.index[0].date(), s.index[-1].date()
    if first > start:
        raise DataIntegrityError(f"{label}: data starts {first}, after the requested start {start}")
    if last < end:
        raise DataIntegrityError(f"{label}: data ends {last}, before the requested end {end}")
    return s.loc[pd.Timestamp(start): pd.Timestamp(end)]


def exchange_daily(asset: Literal["BTC", "ETH"], start: date, end: date, cache_dir: Path) -> pd.Series:
    """USD daily closes for [start, end] (DatetimeIndex of UTC days).

    BTC: Bitstamp BTC/USD. ETH: Poloniex ETH_BTC x Bitstamp BTC/USD, same day.
    Each source series may forward-fill at most MAX_FFILL_DAYS consecutive
    missing days (fill_gaps) and must cover start..end, else
    DataIntegrityError. The raw source series and the USD series are reported
    to record_api_prices as bitstamp:btcusd:<range>, poloniex:ETH_BTC:<range>
    and usd:<asset>:<range>, where <range> is the requested
    YYYY-MM-DD..YYYY-MM-DD, so loads over different ranges never share a key.
    """
    if asset not in ("BTC", "ETH"):
        raise ValueError(f"asset must be BTC or ETH, got {asset!r}")
    rng = f"{start.isoformat()}..{end.isoformat()}"
    raw_btc = bitstamp_btcusd(start, end, cache_dir)
    note_api_price(f"bitstamp:btcusd:{rng}", raw_btc)
    btc = _covering(raw_btc, "bitstamp btcusd", start, end)
    if asset == "BTC":
        out = btc
    else:
        raw_eth = poloniex_ethbtc(start, end, cache_dir)
        note_api_price(f"poloniex:ETH_BTC:{rng}", raw_eth)
        ethbtc = _covering(raw_eth, "poloniex ETH_BTC", start, end)
        out = ethbtc * btc.reindex(ethbtc.index)
    out = out.rename(f"{asset}USD")
    out.index.name = "date"
    note_api_price(f"usd:{asset}:{rng}", out)
    return out
