# Pre-registration files

One candidate = one `NN-name.json` here, registered before its Test run. The judge (`python -m judge run`) refuses to run unless everything below holds.

## Files per candidate

| File | What it is | When |
| --- | --- | --- |
| `NN-name.json` | Every rule value: `candidate_id`, `paper`, `long_short`, `lookbacks`, `vol_target`, `vol_window`, `per_coin_cap_repro`, `gross_cap`, `cost_per_side`, `rebalance_threshold`, `universe_size`, `paper_sample_end`, `seed`, `n_sims`, `judge_tree_hash` | Registration |
| `NN-name.sha256` | sha256 of the `.json` bytes (`sha256sum` format is fine) | Registration |
| `NN-name.json.ots` | OpenTimestamps stamp of the `.json` | Registration |
| `NN-name.stamp_confirmed` | One line: the ISO date (UTC) the stamp was confirmed in a Bitcoin block | Once confirmed; needed for Final |

`judge_tree_hash` is `git rev-parse HEAD:judge` at registration. `candidate_id` must equal the file name without `.json` (`NN-name`).

These values are fixed by the judge and must appear exactly: `cost_per_side` 0.0011, `rebalance_threshold` 0.2, `gross_cap` 1.0, `n_sims` 1000, `universe_size` 20. Types are strict: `long_short` is a JSON `true`/`false`, integers are JSON integers.

## Registering

1. Commit the judge code. Record `git rev-parse HEAD:judge` as `judge_tree_hash`.
2. Write `NN-name.json` (LF line endings; the repo's `.gitattributes` forces them, so the sha256 is the same on every OS), then `sha256sum NN-name.json > NN-name.sha256`, then `ots stamp NN-name.json`.
3. Commit all three and post the sha256 publicly before the Test run.
4. When the stamp is confirmed, commit `NN-name.stamp_confirmed`.

## What the judge checks before running

- The prereg is `prereg/NN-name.json`, and it, its `.sha256` and any `.stamp_confirmed` are committed with working bytes identical to HEAD (a CRLF working copy is refused).
- sha256 of the committed `NN-name.json` blob equals `NN-name.sha256`.
- `judge_tree_hash` equals the current `git rev-parse HEAD:judge`.
- No uncommitted or untracked changes under `judge/` or `prereg/`.
- The judge code running is the code in this repo.

Any failure names the file and both hashes. Editing a prereg after registration means a new candidate with a new number, never an edit.

## Windows

- Test: max(2023-01-01, `paper_sample_end` + 1 day) .. 2026-10-19.
- Final: the 90 days starting the day after `stamp_confirmed`. It waits until the data covers all 90 days, and it needs:
  - a committed, unmodified `results/{id}-test.json` that says `passed: true` and whose `prereg_sha256` and `judge_tree_hash` equal the current ones (not needed for a Final-alone candidate, i.e. Test window under 12 months, run with `--kind final_alone`);
  - no uncommitted changes under `results/`, and no Final result or opened marker ever in git history.
- Final opens once. `results/{id}.final_opened` is written before any Final data loads and records the prereg sha256, judge tree hash and window. If a run fails after that (network, Ctrl-C), rerunning resumes only when all three match and no Final result exists; any other marker is refused. Commit the marker together with the Final result.

## stamp_confirmed

Final reads the date only from the committed `NN-name.stamp_confirmed`. `--stamp-confirmed YYYY-MM-DD` can only confirm it: a flag without the file, or one that disagrees with it, is refused. The manifest records `stamp_source` (`file` or `file+flag`). The judge refuses a stamp date unless (a) the Final start (stamp + 1 day) is after the Test window end (2026-10-19), and (b) the stamp date is on or after the prereg's first commit date (`git log --diff-filter=A --format=%cs`, oldest).
