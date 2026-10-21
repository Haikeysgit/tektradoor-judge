import hashlib
import io
import logging
import zipfile
from datetime import date
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from judge import data
from judge.data import (
    ArchiveFetchError,
    ChecksumMismatch,
    DataIntegrityError,
    MissingArchiveFile,
)

URL = "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2026-01.zip"
FUND_URL = "https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2026-01.zip"
HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore"
)


def make_zip(name: str, csv_text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, csv_text)
    return buf.getvalue()


def checksum_body(blob: bytes, bad: bool = False) -> bytes:
    digest = "0" * 64 if bad else hashlib.sha256(blob).hexdigest()
    return f"{digest}  x.zip\n".encode()


class Resp:
    def __init__(self, status=200, content=b"", json_data=None):
        self.status_code = status
        self.content = content
        self.text = content.decode("utf-8", "ignore")
        self._json = json_data

    def json(self):
        return self._json


def router(mapping):
    calls = []

    def fake_get(url, params=None, timeout=None, **kw):
        calls.append(url)
        return mapping[url]

    fake_get.calls = calls
    return fake_get


def ms(d: str) -> int:
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def kline_row(day: str, close: float, qv: float) -> str:
    t = ms(day)
    return f"{t},1,2,0.5,{close},10,{t + 86399999},{qv},5,1,1,0"


def serve_klines(csv_text, url=URL):
    blob = make_zip("f.csv", csv_text)
    return router(
        {url: Resp(200, blob), url + ".CHECKSUM": Resp(200, checksum_body(blob))}
    )


def test_checksum_ok_returns_path(tmp_path):
    blob = make_zip("f.csv", "a,b\n")
    fake = router({URL: Resp(200, blob), URL + ".CHECKSUM": Resp(200, checksum_body(blob))})
    with mock.patch("judge.data.requests.get", fake):
        p = data.fetch_file(URL, tmp_path)
    assert isinstance(p, Path) and p.exists() and p.read_bytes() == blob


def test_checksum_mismatch_redownloads_once_then_raises(tmp_path):
    blob = make_zip("f.csv", "a,b\n")
    fake = router(
        {URL: Resp(200, blob), URL + ".CHECKSUM": Resp(200, checksum_body(blob, bad=True))}
    )
    with mock.patch("judge.data.requests.get", fake):
        with pytest.raises(ChecksumMismatch):
            data.fetch_file(URL, tmp_path)
    assert fake.calls.count(URL) == 2
    assert not list(tmp_path.rglob("*.zip"))  # corrupt file not kept


def test_404_raises_missing_archive_file(tmp_path):
    fake = router({URL: Resp(404), URL + ".CHECKSUM": Resp(404)})
    with mock.patch("judge.data.requests.get", fake):
        with pytest.raises(MissingArchiveFile) as ei:
            data.fetch_file(URL, tmp_path)
    assert URL in str(ei.value)


def test_429_retries_three_times_then_raises(tmp_path):
    fake = router({URL: Resp(429), URL + ".CHECKSUM": Resp(429)})
    sleeps = []
    with mock.patch("judge.data.requests.get", fake), mock.patch(
        "judge.data.time.sleep", sleeps.append
    ):
        with pytest.raises(ArchiveFetchError) as ei:
            data.fetch_file(URL, tmp_path)
    assert sleeps == [2, 8, 30]
    assert ei.value.status == 429
    assert URL in str(ei.value)


def test_daily_klines_parses_close_and_quote_volume(tmp_path):
    csv = "\n".join(
        [HEADER, kline_row("2026-01-01", 100.0, 1000.0), kline_row("2026-01-02", 101.5, 2000.0)]
    )
    with mock.patch("judge.data.requests.get", serve_klines(csv)):
        df = data.daily_klines("BTCUSDT", "um", date(2026, 1, 1), date(2026, 1, 2), tmp_path)
    assert list(df["close"]) == [100.0, 101.5]
    assert list(df["quote_volume"]) == [1000.0, 2000.0]
    assert [d.date().isoformat() for d in df.index] == ["2026-01-01", "2026-01-02"]


def test_daily_klines_headerless_and_microseconds(tmp_path):
    rows = []
    for day, c in (("2026-01-01", 100.0), ("2026-01-02", 101.5)):
        t = ms(day) * 1000
        rows.append(f"{t},1,2,0.5,{c},10,{t + 86399999999},7,5,1,1,0")
    with mock.patch("judge.data.requests.get", serve_klines("\n".join(rows))):
        df = data.daily_klines("BTCUSDT", "um", date(2026, 1, 1), date(2026, 1, 2), tmp_path)
    assert list(df["close"]) == [100.0, 101.5]
    assert df.index[0] == pd.Timestamp("2026-01-01")


def test_duplicate_date_raises_data_integrity_error(tmp_path):
    csv = "\n".join(
        [HEADER, kline_row("2026-01-01", 100.0, 1.0), kline_row("2026-01-01", 101.0, 1.0)]
    )
    with mock.patch("judge.data.requests.get", serve_klines(csv)):
        with pytest.raises(DataIntegrityError):
            data.daily_klines("BTCUSDT", "um", date(2026, 1, 1), date(2026, 1, 2), tmp_path)


def test_missing_day_inside_month_forward_filled_and_logged(tmp_path, caplog):
    csv = "\n".join(
        [HEADER, kline_row("2026-01-01", 100.0, 1.0), kline_row("2026-01-03", 103.0, 3.0)]
    )
    with caplog.at_level(logging.WARNING), mock.patch(
        "judge.data.requests.get", serve_klines(csv)
    ):
        df = data.daily_klines("BTCUSDT", "um", date(2026, 1, 1), date(2026, 1, 3), tmp_path)
    assert list(df["close"]) == [100.0, 100.0, 103.0]
    assert "gap" in caplog.text and "2026-01-02" in caplog.text


def test_daily_funding_sums_three_rates_per_day(tmp_path):
    h = 3600 * 1000
    d0 = ms("2026-01-01")
    csv = "calc_time,funding_interval_hours,last_funding_rate\n" + "\n".join(
        f"{d0 + i * 8 * h},8,{r}" for i, r in enumerate(["0.0001", "0.0002", "-0.0001"])
    )
    blob = make_zip("f.csv", csv)
    fake = router(
        {FUND_URL: Resp(200, blob), FUND_URL + ".CHECKSUM": Resp(200, checksum_body(blob))}
    )
    with mock.patch("judge.data.requests.get", fake):
        s = data.daily_funding("BTCUSDT", date(2026, 1, 1), date(2026, 1, 1), tmp_path)
    assert s.iloc[0] == pytest.approx(0.0002)


def test_daily_funding_api_fallback_when_archive_missing(tmp_path):
    d0 = ms("2026-01-01")
    api = [
        {"symbol": "BTCUSDT", "fundingTime": d0, "fundingRate": "0.0003"},
        {"symbol": "BTCUSDT", "fundingTime": d0 + 8 * 3600 * 1000, "fundingRate": "0.0001"},
    ]

    def fake_get(u, params=None, timeout=None, **kw):
        if u in (FUND_URL, FUND_URL + ".CHECKSUM"):
            return Resp(404)
        assert u == "https://fapi.binance.com/fapi/v1/fundingRate"
        return Resp(200, json_data=api if params["startTime"] <= d0 else [])

    with mock.patch("judge.data.requests.get", fake_get):
        s = data.daily_funding("BTCUSDT", date(2026, 1, 1), date(2026, 1, 1), tmp_path)
    assert s.iloc[0] == pytest.approx(0.0004)


def test_list_symbols_follows_pagination():
    def xml(prefixes, trunc, nxt=""):
        body = "".join(
            f"<CommonPrefixes><Prefix>data/futures/um/monthly/klines/{p}/</Prefix></CommonPrefixes>"
            for p in prefixes
        )
        n = f"<NextMarker>{nxt}</NextMarker>" if nxt else ""
        return (
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            f"<IsTruncated>{'true' if trunc else 'false'}</IsTruncated>{n}{body}</ListBucketResult>"
        ).encode()

    pages = [Resp(200, xml(["AAAUSDT", "BTCUSDT"], True, "m1")), Resp(200, xml(["ETHUSDT"], False))]
    seen = []

    def fake_get(u, params=None, timeout=None, **kw):
        seen.append(u)
        return pages.pop(0)

    with mock.patch("judge.data.requests.get", fake_get):
        out = data.list_symbols("um")
    assert out == ["AAAUSDT", "BTCUSDT", "ETHUSDT"]
    assert "marker=m1" in seen[1]


# ------------------------------------------------------------ fix round 1

TODAY = date(2026, 10, 21)


@pytest.fixture(autouse=True)
def fixed_clock():
    with mock.patch("judge.data._today_utc", lambda: TODAY):
        yield


def murl(sym, ym, kind="monthly", market="futures/um"):
    base = f"https://data.binance.vision/data/{market}/{kind}/klines/{sym}"
    return f"{base}/1d/{sym}-1d-{ym}.zip"


def archive(files, extra=None):
    """fake requests.get: files maps zip url -> csv text; anything else is 404."""
    calls = []

    def fake_get(url, params=None, timeout=None, **kw):
        calls.append(url)
        if extra and url in extra:
            return extra[url](url, params)
        base = url[: -len(".CHECKSUM")] if url.endswith(".CHECKSUM") else url
        if base not in files:
            return Resp(404)
        blob = make_zip("f.csv", files[base])
        return Resp(200, checksum_body(blob) if url != base else blob)

    fake_get.calls = calls
    return fake_get


def month_csv(year, month, days, close0=100.0):
    rows = [kline_row(f"{year}-{month:02d}-{d:02d}", close0 + d, 10.0 * d) for d in days]
    return "\n".join([HEADER] + rows)


def test_trailing_404_is_end_of_listing_with_last_date(tmp_path):
    fake = archive({murl("OLDUSDT", "2026-01"): month_csv(2026, 1, range(1, 32))})
    with mock.patch("judge.data.requests.get", fake):
        df = data.daily_klines("OLDUSDT", "um", date(2026, 1, 1), date(2026, 4, 30), tmp_path)
    assert len(df) == 31 and df.index[-1] == pd.Timestamp("2026-01-31")
    assert df.attrs["last_date"] == pd.Timestamp("2026-01-31")


def test_404_between_existing_files_raises(tmp_path):
    fake = archive({
        murl("XUSDT", "2026-01"): month_csv(2026, 1, range(1, 32)),
        murl("XUSDT", "2026-03"): month_csv(2026, 3, range(1, 32)),
    })
    with mock.patch("judge.data.requests.get", fake):
        with pytest.raises(MissingArchiveFile):
            data.daily_klines("XUSDT", "um", date(2026, 1, 1), date(2026, 3, 31), tmp_path)


def test_leading_404_skipped(tmp_path):
    fake = archive({murl("NEWUSDT", "2026-02"): month_csv(2026, 2, range(1, 29))})
    with mock.patch("judge.data.requests.get", fake):
        df = data.daily_klines("NEWUSDT", "um", date(2026, 1, 1), date(2026, 2, 28), tmp_path)
    assert df.index[0] == pd.Timestamp("2026-02-01") and len(df) == 28


def test_all_missing_raises(tmp_path):
    with mock.patch("judge.data.requests.get", archive({})):
        with pytest.raises(MissingArchiveFile):
            data.daily_klines("NOPEUSDT", "um", date(2026, 1, 1), date(2026, 2, 28), tmp_path)


def test_current_month_uses_daily_zips_and_unpublished_yesterday_is_end(tmp_path, caplog):
    files = {murl("BTCUSDT", "2026-09"): month_csv(2026, 9, range(1, 31))}
    for d in range(1, 20):  # 10-01..10-19; 10-20 (yesterday) not published yet
        files[murl("BTCUSDT", f"2026-10-{d:02d}", kind="daily")] = month_csv(2026, 10, [d])
    fake = archive(files)
    with caplog.at_level(logging.WARNING), mock.patch("judge.data.requests.get", fake):
        df = data.daily_klines("BTCUSDT", "um", date(2026, 9, 1), date(2026, 10, 20), tmp_path)
    assert len(df) == 30 + 19 and df.attrs["last_date"] == pd.Timestamp("2026-10-19")
    assert murl("BTCUSDT", "2026-10-05", kind="daily") in fake.calls
    assert not any("2026-10.zip" in c for c in fake.calls)
    assert "not published yet" in caplog.text


def test_previous_month_unpublished_early_in_month_falls_back_to_daily(tmp_path):
    files = {murl("BTCUSDT", "2026-09"): month_csv(2026, 9, range(1, 31))}
    for d in range(1, 32):
        files[murl("BTCUSDT", f"2026-10-{d:02d}", kind="daily")] = month_csv(2026, 10, [d])
    with mock.patch("judge.data._today_utc", lambda: date(2026, 11, 2)), mock.patch(
        "judge.data.requests.get", archive(files)
    ):
        df = data.daily_klines("BTCUSDT", "um", date(2026, 9, 1), date(2026, 10, 31), tmp_path)
    assert len(df) == 61 and df.attrs["last_date"] == pd.Timestamp("2026-10-31")


def test_old_missing_trailing_month_not_treated_as_unpublished(tmp_path, caplog):
    fake = archive({murl("OLDUSDT", "2026-01"): month_csv(2026, 1, range(1, 32))})
    with caplog.at_level(logging.WARNING), mock.patch("judge.data.requests.get", fake):
        data.daily_klines("OLDUSDT", "um", date(2026, 1, 1), date(2026, 2, 28), tmp_path)
    assert "not published yet" not in caplog.text


@pytest.mark.parametrize("status", [451, 503])
def test_451_retries_503_does_not(tmp_path, status):
    fake = router({URL: Resp(status), URL + ".CHECKSUM": Resp(status)})
    sleeps = []
    with mock.patch("judge.data.requests.get", fake), mock.patch("judge.data.time.sleep", sleeps.append):
        with pytest.raises(ArchiveFetchError) as ei:
            data.fetch_file(URL, tmp_path)
    assert ei.value.status == status
    assert sleeps == ([2, 8, 30] if status == 451 else [])


def test_request_exception_retries_then_succeeds(tmp_path):
    import requests as rq

    blob = make_zip("f.csv", "a,b\n")
    state = {"n": 0}

    def fake_get(url, params=None, timeout=None, **kw):
        if url.endswith(".CHECKSUM"):
            state["n"] += 1
            if state["n"] <= 2:
                raise rq.ConnectionError("boom")
            return Resp(200, checksum_body(blob))
        return Resp(200, blob)

    sleeps = []
    with mock.patch("judge.data.requests.get", fake_get), mock.patch("judge.data.time.sleep", sleeps.append):
        p = data.fetch_file(URL, tmp_path)
    assert p.exists() and sleeps == [2, 8]


def test_request_exception_exhausted_raises(tmp_path):
    import requests as rq

    def fake_get(url, params=None, timeout=None, **kw):
        raise rq.Timeout("slow")

    sleeps = []
    with mock.patch("judge.data.requests.get", fake_get), mock.patch("judge.data.time.sleep", sleeps.append):
        with pytest.raises(ArchiveFetchError) as ei:
            data.fetch_file(URL, tmp_path)
    assert sleeps == [2, 8, 30] and ei.value.status is None


def test_cache_reuse_does_not_redownload_zip(tmp_path):
    blob = make_zip("f.csv", "a,b\n")
    fake = router({URL: Resp(200, blob), URL + ".CHECKSUM": Resp(200, checksum_body(blob))})
    with mock.patch("judge.data.requests.get", fake):
        data.fetch_file(URL, tmp_path)
        data.fetch_file(URL, tmp_path)
    assert fake.calls.count(URL) == 1


def test_panel_wide_frames_and_delisted_symbol_tolerated(tmp_path):
    files = {
        murl("AUSDT", "2026-01"): month_csv(2026, 1, range(1, 32)),
        murl("AUSDT", "2026-02"): month_csv(2026, 2, range(1, 29)),
        murl("DEADUSDT", "2026-01"): month_csv(2026, 1, range(1, 32), close0=5.0),
    }
    start, end = date(2026, 1, 1), date(2026, 2, 28)
    fake = archive(files, extra={"https://fapi.binance.com/fapi/v1/fundingRate": lambda u, p: Resp(200, json_data=[])})
    with mock.patch("judge.data.requests.get", fake):
        close, vol, fund = data.panel(["AUSDT", "DEADUSDT"], "um", start, end, tmp_path)
    assert list(close.columns) == ["AUSDT", "DEADUSDT"] and len(close) == 59
    assert close.shape == vol.shape == fund.shape
    assert close["DEADUSDT"].loc["2026-01-31"] == 36.0
    assert close["DEADUSDT"].loc["2026-02-01":].isna().all()
    assert close["AUSDT"].notna().all()
    assert close.attrs["last_date"]["DEADUSDT"] == pd.Timestamp("2026-01-31")
    assert close.attrs["last_date"]["AUSDT"] == pd.Timestamp("2026-02-28")
    assert fund["DEADUSDT"].loc["2026-02-01":].isna().all()


def test_panel_spot_funding_is_zero(tmp_path):
    spot = murl("AUSDT", "2026-01", market="spot")
    fake = archive({spot: month_csv(2026, 1, range(1, 32))})
    with mock.patch("judge.data.requests.get", fake):
        close, vol, fund = data.panel(["AUSDT"], "spot", date(2026, 1, 1), date(2026, 1, 31), tmp_path)
    assert (fund == 0.0).all().all() and fund.shape == close.shape
