# tektradoor-judge

An open-source, pre-registered test rig for crypto trading strategies. Each candidate strategy is written down (rules, seed, code version) and timestamped before it is run on out-of-sample data, so the result cannot be tuned after the fact, by anyone including the owner. Costs, funding and a same-universe buy-and-hold benchmark are applied the same way to every candidate.

The code, the registered rules and every result are public in this repository. The sha256 of each prereg file is also posted publicly (and stamped with OpenTimestamps) before that candidate's Test run, so anyone can check that the rules in this repo are the ones that were registered; the judge itself refuses to run if that file or the judge code changed after registration.

## Running a candidate

```
python -m judge reproduce prereg/drafts/01-name.json   # paper reproduction, before registering
git add results/ && git commit            # commit results/01-name-repro.json

python -m judge run prereg/01-name.json --kind test
git add results/ && git commit            # commit the Test result

python -m judge open-final prereg/01-name.json    # Final step 1: checks, writes the marker, loads no data
git add results/ && git commit            # commit results/01-name.final_opened
python -m judge run prereg/01-name.json --kind final   # Final step 2 (final_alone if the Test window is under 12 months)
git add results/ && git commit            # commit the Final result
```

`reproduce` first checks that the judge reproduces the paper's own single-asset Sharpe ratios with the paper's own costs (10 bps per side, 20% rebalance threshold, 200% per-coin cap, spot prices from Bitstamp and Poloniex). Every check must land within 20% of the paper's number. It writes `results/{candidate_id}-repro.json`, pass or fail, never overwritten, and the canonical `date,close` CSV of every price series it used to `results/repro-data/`. Their sha256s are in the result's manifest, so the reproduction can be rerun and audited without the exchange APIs. Commit both.

`run --kind test` (and `final_alone`, and `open-final` for a Final-alone candidate) refuses unless the repro result passes all of the following checks. The file's own `passed` flag is not trusted alone.

- It is committed in exactly one commit.
- It recomputes as a pass: the expected checks, with the judge's targets, costs, threshold and periods, and each `|ours - target| / |target|` within 20%.
- Its `strategy_sha256` matches the prereg's `long_short`, `lookbacks`, `vol_target`, `vol_window`, `per_coin_cap_repro` and `paper_sample_end`.
- Its price CSVs are committed with matching hashes.
- The code that decides reproduction numbers is byte-identical between the judge tree that reproduced and `HEAD:judge`. That code is `judge/strategies/`, `costs.py`, `metrics.py`, `repro.py` and `prices.py`. Other judge code, such as `judge.py`, may change without a rerun.

A failed reproduction is not tuned: the candidate stops there.

Each run writes `results/{candidate_id}-{kind}.json`, pass or fail, with a manifest (archive checksums, prereg sha256, judge tree hash, seed, `pip freeze`, run time). Results are never overwritten. `results/` is committed: it is the record, so it is deliberately not gitignored.

Final is two steps. `open-final` runs every Final gate check and writes `results/{id}.final_opened` (prereg sha256, judge tree hash, window); commit it. `run --kind final` loads data only when that marker is committed in exactly one commit and matches. If a Final run fails (network error, Ctrl-C), run it again: the committed marker allows it as long as no Final result exists. Once a Final result has been committed, Final never runs again. See `prereg/README.md` for the files each candidate needs.

## Limits

The judge's history checks read local git: commit dates, "exactly one commit" and "never in history" are only as trustworthy as the repository. Whoever holds the repo can rewrite history and commit dates. These checks catch mistakes and make casual edits visible; they do not prove anything to an outsider. The real anchors are outside the repo:

- the public post of each prereg's sha256, made before its Test run;
- GitHub's push timestamps for the commits;
- the OpenTimestamps proof (`.ots`) of each prereg, attested in a Bitcoin block.

The judge does not yet verify the `.ots` proof itself (TODO-4).

The reproduction gate checks a result file's internal consistency and its committed price data. It does not recompute the Sharpe ratios from that data at Test time. The code tie ensures the same reproduction code would produce them.

## Development

```
pip install --require-hashes -r requirements.txt
pytest -q
python scripts/check_no_secrets.py
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
