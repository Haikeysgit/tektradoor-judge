"""Binance public-archive loader: verified downloads and daily tables.

All dates are UTC calendar days. Tables are indexed by tz-naive midnight
timestamps that represent UTC days (index name "date").
"""
from __future__ import annotations

import hashlib
import io
import logging
import time
import xml.etree.ElementTree as ET
import zipfile
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

log = logging.getLogger(__name__)

ARCHIVE = "https://data.binance.vision"
S3_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FAPI_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
RETRY_WAITS = (2, 8, 30)
RETRY_STATUSES = (429, 451)
TIMEOUT = 60

Market = Literal["um", "spot"]


class MissingArchiveFile(Exception):
    """The archive has no file at this URL (HTTP 404)."""


class NoDataInWindow(MissingArchiveFile):
    """A symbol has no archive rows at all inside the requested window."""


class ChecksumMismatch(Exception):
    """Downloaded file does not match its .CHECKSUM, even after one re-download."""


class ArchiveFetchError(Exception):
    """Network failure or bad HTTP status after all retries."""

    def __init__(self, url: str, status: int | None, detail: str = ""):
        self.url = url
        self.status = status
        msg = f"fetch failed for {url}: status={status}"
        super().__init__(f"{msg} ({detail})" if detail else msg)


class DataIntegrityError(Exception):
    """The data itself is inconsistent (e.g. duplicate dates)."""


# ---------------------------------------------------------------- HTTP

def _http_get(url: str, params: dict | None = None, allow_404: bool = False):
    """GET with retries on network errors, 429 and 451 (waits 2s, 8s, 30s)."""
    attempt = 0
    while True:
        status: int | None = None
        detail = ""
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT)
            status = resp.status_code
            if status == 200:
                return resp
            if status == 404 and not allow_404:
                raise MissingArchiveFile(f"404 not found: {url}")
            if status not in RETRY_STATUSES:
                raise ArchiveFetchError(url, status)
        except requests.RequestException as exc:
            detail = f"{type(exc).__name__}: {exc}"
        if attempt >= len(RETRY_WAITS):
            raise ArchiveFetchError(url, status, detail)
        wait = RETRY_WAITS[attempt]
        attempt += 1
        log.warning("retry %d for %s (status=%s) in %ss", attempt, url, status, wait)
        time.sleep(wait)


def _local_path(url: str, cache_dir: Path) -> Path:
    rel = url.split("://", 1)[-1]
    return Path(cache_dir) / rel


def _expected_sha(text: str, url: str) -> str:
    parts = text.split()
    if not parts or len(parts[0]) != 64:
        raise ArchiveFetchError(url + ".CHECKSUM", 200, "unparseable checksum file")
    return parts[0].lower()


_fetch_recorders: list[list[Path]] = []


@contextmanager
def record_fetches():
    """Collect the cached path of every file fetch_file verifies in the block.

    The judge reads each path's .CHECKSUM sidecar (checksum_path) into the
    result manifest.
    """
    rec: list[Path] = []
    _fetch_recorders.append(rec)
    try:
        yield rec
    finally:
        _fetch_recorders[:] = [r for r in _fetch_recorders if r is not rec]


def note_fetch(path: Path) -> None:
    """Report a verified cached file to every active record_fetches block."""
    for rec in _fetch_recorders:
        rec.append(Path(path))


def checksum_path(path: Path) -> Path:
    """The cached copy of a file's archive .CHECKSUM, next to the file."""
    path = Path(path)
    return path.with_name(path.name + ".CHECKSUM")


def _verified(path: Path, checksum_text: str) -> Path:
    side = checksum_path(path)
    blob = checksum_text.encode("utf-8")
    if not side.exists() or side.read_bytes() != blob:
        side.write_bytes(blob)
    note_fetch(path)
    return path


def fetch_file(url: str, cache_dir: Path) -> Path:
    """Download url (+ .CHECKSUM), verify sha256, return the cached path.

    On mismatch re-downloads once, then raises ChecksumMismatch. A cached file
    is trusted only if it still matches its freshly fetched checksum. The
    verified .CHECKSUM text is kept next to the file (checksum_path) and the
    path is reported to record_fetches.
    """
    path = _local_path(url, Path(cache_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    checksum_text = _http_get(url + ".CHECKSUM").text
    expected = _expected_sha(checksum_text, url)

    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        return _verified(path, checksum_text)

    actual = ""
    for attempt in (1, 2):
        blob = _http_get(url).content
        actual = hashlib.sha256(blob).hexdigest()
        if actual == expected:
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(blob)
            tmp.replace(path)
            return _verified(path, checksum_text)
        log.warning("checksum mismatch for %s (download %d)", url, attempt)
        if attempt == 1:
            checksum_text = _http_get(url + ".CHECKSUM").text
            expected = _expected_sha(checksum_text, url)
    path.unlink(missing_ok=True)
    raise ChecksumMismatch(f"{url}: expected {expected}, got {actual}")


# ---------------------------------------------------------------- listing

def list_symbols(market: Market) -> list[str]:
    root = {"um": "data/futures/um", "spot": "data/spot"}[market]
    prefix = f"{root}/monthly/klines/"
    base = f"{S3_LIST}?prefix={prefix}&delimiter=/"
    out: list[str] = []
    marker = ""
    while True:
        url = base + (f"&marker={marker}" if marker else "")
        tree = ET.fromstring(_http_get(url).content)
        ns = {"s": tree.tag.split("}")[0].lstrip("{")} if tree.tag.startswith("{") else None

        def find_all(tag):
            return tree.findall(f"s:{tag}", ns) if ns else tree.findall(tag)

        def find_one(tag):
            return tree.find(f"s:{tag}", ns) if ns else tree.find(tag)

        last = ""
        for cp in find_all("CommonPrefixes"):
            p = (cp.find("s:Prefix", ns) if ns else cp.find("Prefix")).text
            last = p
            out.append(p[len(prefix):].strip("/"))
        trunc = find_one("IsTruncated")
        if trunc is None or trunc.text.strip().lower() != "true":
            break
        nxt = find_one("NextMarker")
        marker = nxt.text if nxt is not None and nxt.text else last
    return sorted(set(out))


# ---------------------------------------------------------------- helpers

def _month_starts(start: date, end: date) -> list[date]:
    months, cur = [], date(start.year, start.month, 1)
    while cur <= end:
        months.append(cur)
        cur = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
    return months


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _read_csv_from_zip(path: Path, names: list[str], first_col_numeric: str) -> pd.DataFrame:
    """Read the single CSV in a zip, with or without a header row."""
    with zipfile.ZipFile(path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise DataIntegrityError(f"{path}: expected exactly one csv, found {members}")
        raw = zf.read(members[0]).decode("utf-8")
    first = raw.split("\n", 1)[0].split(",")[0].strip()
    has_header = not first.lstrip("-").isdigit()
    df = pd.read_csv(io.StringIO(raw), header=0 if has_header else None)
    if has_header:
        df.columns = [str(c).strip() for c in df.columns]
        if first_col_numeric not in df.columns:
            df.columns = names + list(df.columns[len(names):])
    else:
        df.columns = names + [f"extra{i}" for i in range(df.shape[1] - len(names))]
    return df


def _to_utc_day(values: pd.Series) -> pd.DatetimeIndex:
    """Epoch ms (or us, 16 digits) -> tz-naive UTC midnight."""
    v = values.astype("int64")
    unit = "us" if v.abs().max() >= 10**14 else "ms"
    return pd.DatetimeIndex(pd.to_datetime(v, unit=unit, utc=True).dt.tz_localize(None).dt.normalize())


KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
]


def _parse_kline_zip(path: Path) -> pd.DataFrame:
    df = _read_csv_from_zip(path, KLINE_COLS, "open_time")
    out = pd.DataFrame(
        {"close": df["close"].astype(float).to_numpy(),
         "quote_volume": df["quote_volume"].astype(float).to_numpy()},
        index=_to_utc_day(df["open_time"]),
    )
    out.index.name = "date"
    if out.index.has_duplicates:
        dups = sorted({d.date().isoformat() for d in out.index[out.index.duplicated()]})
        raise DataIntegrityError(f"{path}: duplicate dates {dups}")
    return out.sort_index()


def _kline_url(market: Market, symbol: str, kind: str, stamp: str) -> str:
    root = {"um": "futures/um", "spot": "spot"}[market]
    return f"{ARCHIVE}/data/{root}/{kind}/klines/{symbol}/1d/{symbol}-1d-{stamp}.zip"


# ---------------------------------------------------------------- klines

def daily_klines(symbol: str, market: Market, start: date, end: date, cache_dir: Path) -> pd.DataFrame:
    """Daily close and quote_volume for [start, end], indexed by UTC date.

    Past months come from monthly zips; the current month (and a just-ended
    month whose monthly zip is not published yet) from daily zips.

    Missing files: leading ones (before listing) are skipped; trailing ones
    mean end of data (delisting, or not yet published when dated within 2 days
    of today, which is logged as a warning); a missing file BETWEEN two
    existing files raises MissingArchiveFile. No data at all also raises.
    Gaps inside a file span are forward-filled (volume 0) and logged. Data is
    never extended past the last observed day: later rows do not exist.

    The last observed date is exposed as ``df.attrs["last_date"]`` (Timestamp).
    """
    cache_dir = Path(cache_dir)
    today = _today_utc()
    entries: list[tuple[date, str, pd.DataFrame | None]] = []

    def load(url: str, when: date, label: str) -> None:
        try:
            entries.append((when, label, _parse_kline_zip(fetch_file(url, cache_dir))))
        except MissingArchiveFile:
            entries.append((when, label, None))

    for m in _month_starts(start, end):
        month_end = date(m.year + (m.month == 12), m.month % 12 + 1, 1) - timedelta(days=1)
        use_daily = (m.year, m.month) == (today.year, today.month)
        if not use_daily:
            url = _kline_url(market, symbol, "monthly", f"{m.year:04d}-{m.month:02d}")
            load(url, month_end, url)
            if entries[-1][2] is not None or month_end < today - timedelta(days=2):
                continue
            entries.pop()  # monthly zip not published yet: use daily zips
            use_daily = True
        last = min(end, today - timedelta(days=1), month_end)
        day = max(start, m)
        while day <= last:
            load(_kline_url(market, symbol, "daily", day.isoformat()), day, day.isoformat())
            day += timedelta(days=1)

    loaded = [i for i, e in enumerate(entries) if e[2] is not None]
    if not loaded:
        raise NoDataInWindow(f"no kline archive files for {symbol} {market} {start}..{end}")
    for when, label, frame in entries[loaded[0]: loaded[-1] + 1]:
        if frame is None:
            raise MissingArchiveFile(f"missing archive file between existing files: {label}")
    for when, label, frame in entries[loaded[-1] + 1:]:
        if when >= today - timedelta(days=2):
            log.warning("%s %s: %s not published yet; treating as end of data", symbol, market, label)
        else:
            log.info("%s %s: %s missing after last data; treating as end of listing", symbol, market, label)
    frames = [e[2] for e in entries if e[2] is not None]
    df = pd.concat(frames)
    if df.index.has_duplicates:
        dups = sorted({d.date().isoformat() for d in df.index[df.index.duplicated()]})
        raise DataIntegrityError(f"{symbol} {market}: duplicate dates across files {dups}")
    df = df.sort_index()
    df = df.loc[pd.Timestamp(start): pd.Timestamp(end)]
    if df.empty:
        raise NoDataInWindow(f"{symbol} {market}: no data within {start}..{end}")
    full = pd.date_range(df.index[0], df.index[-1], freq="D", name="date")
    missing = full.difference(df.index)
    for d in missing:
        log.warning("gap in %s %s: %s missing, forward-filled", symbol, market, d.date().isoformat())
    df = df.reindex(full)
    df["close"] = df["close"].ffill()
    df["quote_volume"] = df["quote_volume"].fillna(0.0)
    df.attrs["last_date"] = df.index[-1]
    return df


# ---------------------------------------------------------------- funding

FUNDING_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]


_api_funding_recorders: list[dict[str, list[tuple[int, float]]]] = []


@contextmanager
def record_api_funding():
    """Collect {symbol: [(funding_time_ms, rate), ...]} for every row fetched
    from the live fapi (months with no funding archive) inside the block.
    These rows have no archive checksum; the judge hashes them instead
    (funding_rows_sha256)."""
    rec: dict[str, list[tuple[int, float]]] = {}
    _api_funding_recorders.append(rec)
    try:
        yield rec
    finally:
        _api_funding_recorders[:] = [r for r in _api_funding_recorders if r is not rec]


def note_api_funding(symbol: str, rows) -> None:
    """Report API funding rows (time_ms, rate) to every active record_api_funding block."""
    rows = [(int(t), float(r)) for t, r in rows]
    for rec in _api_funding_recorders:
        rec.setdefault(symbol, []).extend(rows)


def funding_rows_sha256(symbol: str, rows) -> str:
    """sha256 of a canonical CSV of API funding rows.

    Format: header ``symbol,time,rate``, then one ``symbol,time_ms,repr(float(rate))``
    line per distinct row, sorted by (time, rate), LF line ends, trailing LF,
    UTF-8. Row order and duplicates do not change the hash.
    """
    distinct = sorted({(int(t), float(r)) for t, r in rows})
    lines = ["symbol,time,rate"] + [f"{symbol},{t},{r!r}" for t, r in distinct]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def _funding_from_api(symbol: str, m: date) -> pd.Series:
    start_ms = int(datetime(m.year, m.month, 1, tzinfo=timezone.utc).timestamp() * 1000)
    nm = date(m.year + (m.month == 12), m.month % 12 + 1, 1)
    end_ms = int(datetime(nm.year, nm.month, 1, tzinfo=timezone.utc).timestamp() * 1000) - 1
    times: list[int] = []
    rates: list[float] = []
    cursor = start_ms
    while cursor <= end_ms:
        rows = _http_get(
            FAPI_FUNDING,
            params={"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000},
            allow_404=True,
        ).json()
        if not rows:
            break
        for r in rows:
            times.append(int(r["fundingTime"]))
            rates.append(float(r["fundingRate"]))
        if len(rows) < 1000:
            break
        cursor = int(rows[-1]["fundingTime"]) + 1
    note_api_funding(symbol, zip(times, rates))
    return pd.Series(rates, index=pd.Index(times, dtype="int64"), dtype="float64")


def _funding_from_zip(path: Path) -> pd.Series:
    df = _read_csv_from_zip(path, FUNDING_COLS, "calc_time")
    return pd.Series(
        df["last_funding_rate"].astype(float).to_numpy(),
        index=pd.Index(df["calc_time"].astype("int64").to_numpy(), dtype="int64"),
    )


def daily_funding(symbol: str, start: date, end: date, cache_dir: Path) -> pd.Series:
    """Sum of each UTC day's funding rates, over [start, end]; days with none are 0."""
    cache_dir = Path(cache_dir)
    parts: list[pd.Series] = []
    for m in _month_starts(start, end):
        url = f"{ARCHIVE}/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{m.year:04d}-{m.month:02d}.zip"
        try:
            parts.append(_funding_from_zip(fetch_file(url, cache_dir)))
        except MissingArchiveFile:
            log.info("%s: no funding archive for %s, using fapi", symbol, m)
            parts.append(_funding_from_api(symbol, m))
    days = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D", name="date")
    if not parts:
        return pd.Series(0.0, index=days, name=symbol)
    s = pd.concat(parts)
    if s.index.has_duplicates:
        s = s[~s.index.duplicated(keep="last")]
    if len(s):
        day = pd.to_datetime(s.index, unit="ms", utc=True).tz_localize(None).normalize()
        s = s.groupby(day).sum()
    s = s.reindex(days).fillna(0.0)
    s.name = symbol
    return s


# ---------------------------------------------------------------- panel

def panel(symbols: list[str], market: Market, start: date, end: date, cache_dir: Path):
    """(close, quote_volume, funding): wide frames, dates x symbols.

    Dates before a symbol's listing or after its last observed day (delisting)
    are NaN in all three frames; positions must be closed at the last price.
    Spot funding is all 0. Per-symbol last observed dates are attached as
    ``close.attrs["last_date"]``: dict symbol -> Timestamp.

    A symbol with no rows at all in the window is skipped with a warning: it is
    left out of all three frames and listed in ``close.attrs["skipped"]``. A gap
    between existing files is still a hard failure.
    """
    days = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D", name="date")
    closes, vols, funds, last_dates, skipped = {}, {}, {}, {}, []
    for sym in symbols:
        try:
            k = daily_klines(sym, market, start, end, cache_dir)
        except NoDataInWindow as exc:
            log.warning("skipping %s: %s", sym, exc)
            skipped.append(sym)
            continue
        closes[sym] = k["close"]
        vols[sym] = k["quote_volume"]
        last = k.attrs["last_date"]
        last_dates[sym] = last
        if market == "um":
            f = daily_funding(sym, start, end, cache_dir)
            funds[sym] = f.loc[:last]
    kept = [s for s in symbols if s not in skipped]
    close = pd.DataFrame(closes, columns=kept).reindex(days)
    vol = pd.DataFrame(vols, columns=kept).reindex(days)
    if market == "um":
        fund = pd.DataFrame(funds, columns=kept).reindex(days)
    else:
        fund = pd.DataFrame(0.0, index=days, columns=kept)
    close.attrs["last_date"] = last_dates
    close.attrs["skipped"] = skipped
    return close, vol, fund


# ---------------------------------------------------------------- universe

EXCLUDED_BASES = frozenset({
    "USDC", "BUSD", "TUSD", "USDP", "FDUSD", "DAI", "USDE", "PYUSD", "EUR", "EURI",
    "AEUR", "GBP", "WBTC", "WBETH", "BFUSD", "USD1", "XUSD",
})
QUOTE_SUFFIXES = ("USDT", "USDC", "FDUSD")
VOLUME_WINDOW = 30
EXIT_MIN_MEDIAN_VOLUME = 1_000_000
EXIT_MIN_MEDIAN_ABS_CHANGE = 0.005


def base_asset(symbol: str) -> str:
    """BTCUSDT -> BTC. A multiplier prefix such as 1000 stays part of the name."""
    for q in QUOTE_SUFFIXES:
        if symbol.endswith(q) and len(symbol) > len(q):
            return symbol[: -len(q)]
    return symbol


def _day_ts(day: date) -> pd.Timestamp:
    return pd.Timestamp(day)


def top_universe(
    close: pd.DataFrame,
    quote_volume: pd.DataFrame,
    month_end: date,
    size: int = 20,
    min_age_days: int = 365,
    min_median_volume: float = 2_000_000,
) -> list[str]:
    """Paper 7.1: the month's tradable coins, ranked by 30-day median quote volume.

    Kept: a close on month_end (not delisted), at least min_age_days non-NaN
    closes up to month_end, 30-day median quote volume >= min_median_volume,
    and a base asset outside the fixed stable/wrapped exclusion list. Ties
    rank by symbol name.
    """
    ts = _day_ts(month_end)
    if ts not in close.index:
        raise ValueError(f"{month_end} is not in the close index")
    hist = close.loc[:ts]
    age = hist.notna().sum()
    median_vol = quote_volume.loc[:ts].tail(VOLUME_WINDOW).median()
    alive = hist.loc[ts].notna()
    ok = alive & (age >= min_age_days) & (median_vol >= min_median_volume)
    ok &= pd.Series({s: base_asset(s) not in EXCLUDED_BASES for s in close.columns})
    ranked = sorted(ok.index[ok], key=lambda s: (-median_vol[s], s))
    return ranked[:size]


def exit_flags(close: pd.DataFrame, quote_volume: pd.DataFrame, day: date) -> list[str]:
    """Coins to drop from tomorrow: stale or illiquid over the 30 days ending day.

    Flagged when the 30-day median quote volume is below $1M, or the 30-day
    median absolute daily price change is below 0.5%. Checked daily. Coins
    with no data on day (delisted or not yet listed) are not flagged here.
    """
    ts = _day_ts(day)
    if ts not in close.index:
        raise ValueError(f"{day} is not in the close index")
    median_vol = quote_volume.loc[:ts].tail(VOLUME_WINDOW).median()
    median_chg = close.loc[:ts].pct_change(fill_method=None).abs().tail(VOLUME_WINDOW).median()
    alive = close.loc[ts].notna()
    flagged = alive & ((median_vol < EXIT_MIN_MEDIAN_VOLUME) | (median_chg < EXIT_MIN_MEDIAN_ABS_CHANGE))
    return [s for s in close.columns if flagged[s]]
