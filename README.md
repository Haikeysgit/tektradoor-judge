# tektradoor-judge

A private, pre-registered test rig for crypto trading strategies. Each candidate strategy is written down (rules, seed, code version) and timestamped before it is run on out-of-sample data, so the result cannot be tuned after the fact, by anyone including the owner. Costs, funding and a same-universe buy-and-hold benchmark are applied the same way to every candidate.

The repository is private. The public proof is the sha256 of each prereg file, posted publicly (and stamped with OpenTimestamps) before that candidate's Test run. When a prereg file is shown later, it can be checked against the posted hash and its stamp; the judge itself refuses to run if that file or the judge code changed after registration.

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

`reproduce` first checks that the judge reproduces the paper's own single-asset Sharpe ratios with the paper's own costs (10 bps per side, 20% rebalance threshold, 200% per-coin cap, spot prices from Bitstamp and Poloniex). Every check must land within 20% of the paper's number. It writes `results/{candidate_id}-repro.json`, pass or fail, never overwritten. `run --kind test` (and `final_alone`) refuses unless that file is committed, says `passed: true`, and its `strategy_sha256` matches the prereg's rules (`long_short`, `lookbacks`, `vol_target`, `vol_window`, `per_coin_cap_repro`). A failed reproduction is not tuned: the candidate stops there.

Each run writes `results/{candidate_id}-{kind}.json`, pass or fail, with a manifest (archive checksums, prereg sha256, judge tree hash, seed, `pip freeze`, run time). Results are never overwritten. `results/` is committed: it is the record, so it is deliberately not gitignored.

Final is two steps. `open-final` runs every Final gate check and writes `results/{id}.final_opened` (prereg sha256, judge tree hash, window); commit it. `run --kind final` loads data only when that marker is committed in exactly one commit and matches. If a Final run fails (network error, Ctrl-C), run it again: the committed marker allows it as long as no Final result exists. Once a Final result has been committed, Final never runs again. See `prereg/README.md` for the files each candidate needs.

## Limits

The judge's history checks read local git: commit dates, "exactly one commit" and "never in history" are only as trustworthy as the repository. Whoever holds the repo can rewrite history and commit dates. These checks catch mistakes and make casual edits visible; they do not prove anything to an outsider. The real anchors are outside the repo:

- the public post of each prereg's sha256, made before its Test run;
- GitHub's push timestamps for the commits;
- the OpenTimestamps proof (`.ots`) of each prereg, attested in a Bitcoin block.

The judge does not yet verify the `.ots` proof itself (TODO-4).

## Development

```
pip install --require-hashes -r requirements.txt
pytest -q
python scripts/check_no_secrets.py
```
