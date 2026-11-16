# tektradoor-judge

An open-source, pre-registered test rig for crypto trading strategies. Each candidate strategy is written down (rules, seed, code version) and timestamped before it is run on out-of-sample data, so the result cannot be tuned after the fact, by anyone including the owner. Costs, funding and a same-universe buy-and-hold benchmark are applied the same way to every candidate.

The repository is public. Before each candidate's Test run, its prereg file is committed here and its sha256 is posted publicly and stamped with OpenTimestamps, so anyone can check the file against the posted hash, the stamp and GitHub's push time. The judge itself refuses to run if that file or the judge code changed after registration.

## Running a candidate

```
python -m judge reproduce prereg/drafts/01-name.json   # paper reproduction, before registering
git add results/ && git commit            # commit results/01-name-repro.json
```

Register, in this order (details in `prereg/README.md`):

1. write `prereg/01-name.json` from its draft, with the real `judge_tree_hash` (`git rev-parse HEAD:judge`);
2. `sha256sum 01-name.json > 01-name.sha256`;
3. `ots stamp 01-name.json`;
4. commit the three files and merge;
5. post the sha256 publicly on X;
6. add the entry `{"01-name": {"sha256": "...", "url": "https://x.com/<user>/status/<id>"}}` to `prereg/announced.json` and commit it;
7. run Test:

```
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

Every run (Test, Final, Final-alone, and `open-final`) also refuses unless `prereg/announced.json` holds the candidate's entry, committed, with the prereg's sha256 and an `x.com`/`twitter.com` status url, and that entry has the same value in every commit that ever touched the file. The post's time is decoded offline from the status id (a snowflake: `(id >> 22) + 1288834974657` ms, UTC). It must be on or after 2026-11-01, on or after the prereg's first commit date minus one day, and before the run starts. The `.ots` must already be in the commit that first added the entry. Every run needs the committed `prereg/01-name.json.ots` to stamp the committed prereg's sha256; a pending calendar attestation is enough for Test. Final, Final-alone and `open-final` need a Bitcoin attestation, verified against the block's raw header from blockstream.info (the header must hash to the block hash; the merkle root and time are read from the header bytes), and `01-name.stamp_confirmed` must equal that block's UTC date (the earliest attestation if there are several). That block must also fall within 7 days before to 48 hours after the announcement post, so a proof stamped again later cannot move the Final window, and the post must come strictly before the Final window starts (stamp date + 1 day, 00:00 UTC). The header must also meet its own proof-of-work target. Final attestations must be at block height 840000 or later, where the header's difficulty must be at least 1e13. A daily workflow (`.github/workflows/ots.yml`) only checks: it tries `ots upgrade` on a scratch copy, posts a notice when an upgrade is available, prints `python -m judge ots-status`, and fails if a proof is still pending 48 hours after its first commit. It never commits or pushes; run `ots upgrade` locally and commit the upgraded proof through a normal PR. GitHub disables scheduled workflows after 60 days of no repo activity; trigger ots.yml manually if so.

Each run writes `results/{candidate_id}-{kind}.json`, pass or fail, with a manifest (archive checksums, prereg sha256, judge tree hash, seed, `pip freeze`, run time, the `.ots` state, and the announcement url and post time). Results are never overwritten. `results/` is committed: it is the record, so it is deliberately not gitignored.

Every result file also has `diagnostics` (`judge/diagnostics.py`). They are report only and never change pass or fail.
- **Trades.** A trade is one coin's position episode in the executed weights, from entry to the return to zero; in long-short a sign flip starts a new trade. The section gives the count, win rate, average win and loss, and holding days. P&L is the trade's summed contribution to portfolio return, net of that coin's costs and funding, in percent of the portfolio.
- **Exit reasons.** Each exit is a trailing stop, leaving the universe, an exit flag or the window end. The section counts them and gives the stop-out share and cost.
- **Worst cases.** The worst trade, the 5 worst trades, and the worst day for the strategy and the benchmark.
- **Leverage.** Max gross and net exposure.

Leverage reports the highest target and executed gross exposure and the number of days executed gross is above the cap. Target gross above the cap, or executed gross above cap / (1 - rebalance threshold) (1.25x), stops the run as a judge bug (exit code 4) and writes no result. `python -m judge run` prints a short summary of the diagnostics after the verdict.

Final is two steps. `open-final` runs every Final gate check and writes `results/{id}.final_opened` (prereg sha256, judge tree hash, window); commit it. `run --kind final` loads data only when that marker is committed in exactly one commit and matches. If a Final run fails (network error, Ctrl-C), run it again: the committed marker allows it as long as no Final result exists. Once a Final result has been committed, Final never runs again. See `prereg/README.md` for the files each candidate needs.

## Limits

The judge's history checks read local git: commit dates, "exactly one commit" and "never in history" are only as trustworthy as the repository. Whoever holds the repo can rewrite history and commit dates. These checks catch mistakes and make casual edits visible; they do not prove anything to an outsider. The real anchors are outside the repo:

- the public post of each prereg's sha256, made before its Test run;
- GitHub's push timestamps for the commits;
- the OpenTimestamps proof (`.ots`) of each prereg, attested in a Bitcoin block.

The judge verifies the `.ots` proof's Bitcoin attestation against one public explorer (blockstream.info), not against a local node. It checks that the header the explorer serves hashes to the block hash it names, and reads the merkle root and time from that header, so the only trust left is that the hash it gives for each height is the main chain's block; anyone can repeat the check with `ots verify` and their own node. The announcement check confirms that `announced.json` names a well-formed post, decodes the post's time from its id and checks it against the registration and the run, and checks that the entry never changed in git. It does not fetch the post, so the post's text (the sha256) must be compared by hand.

"Never changed in git" holds only while the old commits are reachable from some ref. A rebase plus deleting the branch removes the evidence (`--all` does not read reflogs), and a stray stash can block a run. GitHub's public history is the reference for `announced.json`, as it is for dates.

Executed exposure can drift above the 1.0x gross cap, up to 1.25x: the 20% rebalance rule leaves a position alone until its target moves by more than 20%, while new positions enter at their capped targets. This is the published method's rebalance rule, which the reproduction also uses. Targets never exceed the cap, and each result's diagnostics report the drift.

The reproduction gate checks a result file's internal consistency and its committed price data. It does not recompute the Sharpe ratios from that data at Test time. The code tie ensures the same reproduction code would produce them.

## Development

```
pip install --require-hashes -r requirements.txt
pytest -q
python scripts/check_no_secrets.py
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
