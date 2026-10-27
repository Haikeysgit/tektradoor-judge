# Pre-registration files

One candidate = one `NN-name.json` here, registered before its Test run. The judge (`python -m judge run`) refuses to run unless everything below holds.

## Files per candidate

| File | What it is | When |
| --- | --- | --- |
| `NN-name.json` | Every rule value: `candidate_id`, `paper`, `long_short`, `lookbacks`, `vol_target`, `vol_window`, `per_coin_cap_repro`, `gross_cap`, `cost_per_side`, `rebalance_threshold`, `universe_size`, `paper_sample_end`, `seed`, `n_sims`, `judge_tree_hash` | Registration |
| `NN-name.sha256` | sha256 of the `.json` bytes (`sha256sum` format is fine) | Registration |
| `NN-name.json.ots` | OpenTimestamps stamp of the `.json` | Registration |
| `NN-name.stamp_confirmed` | One line: the ISO date (UTC) the stamp was confirmed in a Bitcoin block | Once confirmed; needed for Final |

`judge_tree_hash` is `git rev-parse HEAD:judge` at registration. `candidate_id` names the result files and may use only letters, digits, `.`, `_` and `-`.

## Registering

1. Commit the judge code. Record `git rev-parse HEAD:judge` as `judge_tree_hash`.
2. Write `NN-name.json` (LF line endings; the repo's `.gitattributes` forces them, so the sha256 is the same on every OS), then `sha256sum NN-name.json > NN-name.sha256`, then `ots stamp NN-name.json`.
3. Commit all three and post the sha256 publicly before the Test run.
4. When the stamp is confirmed, commit `NN-name.stamp_confirmed`.

## What the judge checks before running

- sha256 of `NN-name.json` equals `NN-name.sha256`.
- `judge_tree_hash` equals the current `git rev-parse HEAD:judge`.
- No uncommitted or untracked changes under `judge/` or `prereg/`.
- The judge code running is the code in this repo.

Any failure names the file and both hashes. Editing a prereg after registration means a new candidate with a new number, never an edit.

## Windows

- Test: max(2023-01-01, `paper_sample_end` + 1 day) .. 2026-10-19.
- Final: the 90 days starting the day after `stamp_confirmed`. It needs a Test pass (unless the candidate is Final-alone, i.e. its Test window is under 12 months, run with `--kind final_alone`), opens once (`results/{id}.final_opened` is written before any Final data loads) and waits until the data covers all 90 days.

## stamp_confirmed: file or flag

The committed `NN-name.stamp_confirmed` file wins. `--stamp-confirmed YYYY-MM-DD` is used only when no file exists, and a flag that disagrees with the file is refused. Prefer the file: it is committed, so it is part of the audited record.
