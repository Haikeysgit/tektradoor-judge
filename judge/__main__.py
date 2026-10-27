"""CLI: python -m judge run prereg/NN-name.json --kind test|final|final_alone

Exit codes: 0 the run completed and passed, 1 it completed and did not pass
(the result file is written either way), 2 the judge refused to run
(integrity, window, Final gate, existing result), 3 a data error.

stamp_confirmed (needed by final and final_alone): the committed companion
file prereg/NN-name.stamp_confirmed (one ISO date line) wins. --stamp-confirmed
is used only when that file does not exist, and a flag that disagrees with the
file is refused.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from judge import data
from judge.judge import KINDS, JudgeRefusal, run


def _iso_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not an ISO date (YYYY-MM-DD): {text!r}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m judge", description="tektradoor-judge")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="judge one pre-registered candidate on one window")
    r.add_argument("prereg", type=Path, help="prereg/NN-name.json")
    r.add_argument("--kind", required=True, choices=KINDS)
    r.add_argument(
        "--stamp-confirmed", type=_iso_date, default=None, metavar="YYYY-MM-DD",
        help="Bitcoin-confirmed stamp date, used only if prereg/NN-name.stamp_confirmed does not exist",
    )
    r.add_argument("--cache-dir", type=Path, default=None, help="archive cache (default: <repo>/data_cache)")
    args = parser.parse_args(argv)

    try:
        out = run(args.prereg, args.kind, stamp_confirmed=args.stamp_confirmed, cache_dir=args.cache_dir)
    except JudgeRefusal as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except (data.MissingArchiveFile, data.ChecksumMismatch, data.ArchiveFetchError, data.DataIntegrityError) as exc:
        print(f"data error: {exc}", file=sys.stderr)
        return 3
    res = json.loads(out.read_text(encoding="utf-8"))
    v = res["verdict"]
    print(
        f"{'PASSED' if res['passed'] else 'FAILED'} {res['candidate_id']} {res['kind']} "
        f"{res['window']['start']}..{res['window']['end']}: sharpe {v['sharpe']:.3f}, "
        f"max_dd {v['max_dd']:.3f}, bench_sharpe {v['bench_sharpe']:.3f}, luck_odds {res['luck_odds']:.3f}"
    )
    print(f"wrote {out}")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
