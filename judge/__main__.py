"""CLI.

    python -m judge run prereg/NN-name.json --kind test
    python -m judge open-final prereg/NN-name.json      # Final step 1
    python -m judge run prereg/NN-name.json --kind final|final_alone   # step 2
    python -m judge reproduce prereg/drafts/NN-name.json  # paper reproduction

Final is two steps. ``open-final`` runs every Final gate check, writes
results/NN-name.final_opened (prereg sha256, judge tree hash, window) and
stops without loading data. Commit that marker. ``run --kind final`` (or
final_alone) then requires the committed, matching, single-commit marker
before it loads any Final data. A failed Final run can simply be rerun.

Exit codes for run: 0 the run completed and passed, 1 it completed and did not
pass (the result file is written either way), 2 the judge refused (integrity,
window, Final gate, existing result), 3 a data error. open-final: 0 marker
written, 2 refused. reproduce: 0 every check passed, 1 completed and failed
(the result file is written either way), 2 refused, 3 a data error.

Test (and Final-alone) need a committed, passing results/NN-name-repro.json
whose strategy_sha256 matches the prereg's rules.

stamp_confirmed (needed for Final) comes only from the committed companion
file prereg/NN-name.stamp_confirmed (one ISO date line, one commit).
--stamp-confirmed may only confirm it: a flag without the file, or one that
disagrees with it, is refused.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from judge import data
from judge import judge as J
from judge.judge import KINDS, JudgeRefusal, prepare_final, reproduce, run

DATA_ERRORS = (data.MissingArchiveFile, data.ChecksumMismatch, data.ArchiveFetchError, data.DataIntegrityError)


def _iso_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not an ISO date (YYYY-MM-DD): {text!r}") from exc


def _stamp_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--stamp-confirmed", type=_iso_date, default=None, metavar="YYYY-MM-DD",
        help="must equal the committed prereg/NN-name.stamp_confirmed (confirmation only)",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m judge", description="tektradoor-judge")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="judge one pre-registered candidate on one window")
    r.add_argument("prereg", type=Path, help="prereg/NN-name.json")
    r.add_argument("--kind", required=True, choices=KINDS)
    _stamp_arg(r)
    r.add_argument("--cache-dir", type=Path, default=None, help="archive cache (default: <repo>/data_cache)")
    o = sub.add_parser("open-final", help="Final step 1: gate checks, write the marker to commit; loads no data")
    o.add_argument("prereg", type=Path, help="prereg/NN-name.json")
    _stamp_arg(o)
    rp = sub.add_parser("reproduce", help="reproduce the paper's single-asset Sharpe ratios for a prereg draft")
    rp.add_argument("prereg", type=Path, help="prereg/drafts/NN-name.json")
    rp.add_argument("--cache-dir", type=Path, default=None, help="API response cache (default: <repo>/data_cache/repro)")
    args = parser.parse_args(argv)

    if args.command == "reproduce":
        try:
            res = reproduce(args.prereg, loader=J.exchange_daily, cache_dir=args.cache_dir)
        except JudgeRefusal as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
        except DATA_ERRORS as exc:
            print(f"data error: {exc}", file=sys.stderr)
            return 3
        print(f"{'PASSED' if res['passed'] else 'FAILED'} {res['candidate_id']} repro (max rel_error {res['max_rel_error']})")
        print(f"{'check':<22} {'target':>7} {'ours':>8} {'rel_error':>10}  passed")
        for name, c in res["checks"].items():
            print(f"{name:<22} {c['target']:>7.2f} {c['ours']:>8.4f} {c['rel_error']:>10.4f}  {c['passed']}")
        print(f"wrote {Path(J.CODE_ROOT) / 'results' / (res['candidate_id'] + '-repro.json')}")
        return 0 if res["passed"] else 1

    if args.command == "open-final":
        try:
            marker = prepare_final(args.prereg, stamp_confirmed=args.stamp_confirmed)
        except JudgeRefusal as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
        rec = json.loads(marker.read_text(encoding="utf-8"))
        print(f"Final opened: {rec['window']['start']}..{rec['window']['end']}; wrote {marker}")
        print(f"next: git add {marker.name} under results/, commit it, then run --kind {'final_alone' if rec.get('final_alone') else 'final'}")
        return 0

    try:
        out = run(args.prereg, args.kind, stamp_confirmed=args.stamp_confirmed, cache_dir=args.cache_dir)
    except JudgeRefusal as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except DATA_ERRORS as exc:
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
