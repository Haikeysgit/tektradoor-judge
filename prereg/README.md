# Pre-registration files

One candidate = one `NN-name.json` here, registered before its Test run. The judge (`python -m judge run` and `open-final`) refuses to run unless everything below holds. The `.json`, `.sha256` and `.stamp_confirmed` files are each committed exactly once and never edited. Two files are exempt from that rule: the `.json.ots` proof, which `ots upgrade` rewrites, and `announced.json`, which gains one entry per candidate.

## Files per candidate

| File | What it is | When |
| --- | --- | --- |
| `NN-name.json` | Every rule value: `candidate_id`, `paper`, `long_short`, `lookbacks`, `vol_target`, `vol_window`, `per_coin_cap_repro`, `gross_cap`, `cost_per_side`, `rebalance_threshold`, `universe_size`, `paper_sample_end`, `seed`, `n_sims`, `judge_tree_hash` | Registration |
| `NN-name.sha256` | sha256 of the `.json` bytes (`sha256sum` format is fine) | Registration |
| `NN-name.json.ots` | OpenTimestamps proof of the `.json` (`ots stamp`). Recommitted when upgraded (`ots upgrade` locally, through a PR) | Registration; upgraded later |
| `NN-name.stamp_confirmed` | One line: the UTC date of the Bitcoin block that confirms the `.ots` (the earliest Bitcoin attestation's block header time) | Once confirmed, in one commit dated on or before the Final window start; needed for Final |
| `announced.json` (one file for all candidates) | `{"NN-name": {"sha256": "...", "url": "https://x.com/<user>/status/<id>"}}`: where the sha256 was posted | After the public post, before Test; an entry is never changed |

`judge_tree_hash` is `git rev-parse HEAD:judge` at registration. `candidate_id` must equal the file name without `.json` (`NN-name`).

These values are fixed by the judge and must appear exactly: `cost_per_side` 0.0011, `rebalance_threshold` 0.2, `gross_cap` 1.0, `n_sims` 1000, `universe_size` 20. Types are strict: `long_short` is a JSON `true`/`false`, integers are JSON integers.

## Registering

In this order:

1. Write `NN-name.json` from its draft in `drafts/`, with the real `judge_tree_hash`: commit the judge code first, then record `git rev-parse HEAD:judge`. Use LF line endings; the repo's `.gitattributes` forces them, so the sha256 is the same on every OS.
2. `sha256sum NN-name.json > NN-name.sha256`.
3. `ots stamp NN-name.json` (writes `NN-name.json.ots`, pending at the calendars).
4. Commit the three files and merge them to main.
5. Post the sha256 publicly on X.
6. Add the candidate's entry to `announced.json` (`sha256` and the post's `https://x.com/<user>/status/<id>` url) and commit it.
7. Run Test: `python -m judge run prereg/NN-name.json --kind test`.

Later, when the daily `ots` workflow posts a notice that an upgrade is available, run `ots upgrade prereg/NN-name.json.ots` locally (`python -m judge ots-status` then shows `bitcoin`) and commit the upgraded proof through a normal PR, together with `NN-name.stamp_confirmed` holding that block's UTC date. Do this before Final: Final needs both, and the workflow never commits for you.

## What the judge checks before running

- The prereg is `prereg/NN-name.json`, and it, its `.sha256` and any `.stamp_confirmed` are committed with working bytes identical to HEAD (a CRLF working copy is refused).
- sha256 of the committed `NN-name.json` blob equals `NN-name.sha256`.
- `NN-name.json` and `NN-name.sha256` each have exactly one commit touching them (`git log --all`): rewriting both together is refused.
- `judge_tree_hash` equals the current `git rev-parse HEAD:judge`.
- No uncommitted or untracked changes under `judge/` or `prereg/`.
- The judge code running is the code in this repo.
- `NN-name.json.ots` is committed (working bytes equal to HEAD; it may have several commits) and stamps the sha256 of the committed `NN-name.json`. For Test a pending calendar attestation is enough. For Final, Final-alone and `open-final` it needs a Bitcoin block header attestation, and every Bitcoin attestation in it must match its block: the judge fetches the block's raw 80-byte header from blockstream.info (`/api/block/{hash}/header`, hash from `/api/block-height/{h}`), requires its double sha256 (reversed) to equal the block hash, and requires the merkle root in the header to equal the one the proof commits to at that height. The block time is the header's timestamp. No local node, no `ots verify`. The manifest records the proof's sha256, its attestations and, for Final, the verified block.
- Drop stashes before running the judge: the history checks read every ref (`git log --all`), stashes included.
- Test and Final-alone (`run` and `open-final`): `announced.json` is committed and holds an entry for the candidate with exactly `sha256` (equal to the committed prereg's) and `url` (matching `^https://(x|twitter)\.com/[A-Za-z0-9_]+/status/\d+$`). Every version of `announced.json` in git history (all refs) either lacks that entry or holds the same value: an entry that was ever changed is refused, even if it was changed back. Duplicate keys are refused. The manifest records the url.

Any failure names the file and both hashes. Editing a prereg after registration means a new candidate with a new number, never an edit.

## Windows

- Test: max(2023-01-01, `paper_sample_end` + 1 day) .. 2026-10-19.
- Final: the 90 days starting the day after `stamp_confirmed`. It runs in two steps.
  1. `python -m judge open-final prereg/NN-name.json` checks the gate and writes `results/{id}.final_opened`, which records the prereg sha256, judge tree hash and window. It loads no data. It refuses when:
     - the committed, unmodified `results/{id}-test.json` is missing, does not say `passed: true`, or has a `prereg_sha256` or `judge_tree_hash` other than the current ones (this check is skipped for a Final-alone candidate, i.e. Test window under 12 months);
     - `results/` has uncommitted changes;
     - a Final result or the marker was ever in git history.

     Commit the marker.
  2. `python -m judge run prereg/NN-name.json --kind final` (or `final_alone`) loads data only when the marker is committed (tracked and unmodified), touched by exactly one commit, and matches. It also waits until the data covers all 90 days. If the run fails (network, Ctrl-C), run it again; the committed marker allows that while no Final result exists. Commit the Final result; after that Final never runs again.

## stamp_confirmed

Final reads the date only from the committed `NN-name.stamp_confirmed`. `--stamp-confirmed YYYY-MM-DD` can only confirm it: a flag without the file, or one that disagrees with it, is refused. The manifest records `stamp_source` (`file` or `file+flag`). The judge refuses a stamp date unless all of these hold:
- (a) the Final start (stamp + 1 day) is after the Test window end (2026-10-19);
- (b) the stamp date is on or after the prereg's first commit date (`git log --all --diff-filter=A --format=%cs`, oldest);
- (c) the `.stamp_confirmed` file has exactly one commit, dated on or before the Final start, so the window is fixed before its first day and never moved;
- (d) the stamp date equals the UTC date of the block header time of the earliest (lowest height) Bitcoin attestation in the committed `.ots`, verified as above.

Rules (b) and (c) are local git checks; see "Limits" in the top-level README.

## OpenTimestamps upgrades

`.github/workflows/ots.yml` runs daily (and on demand) with read-only permissions, and never commits or pushes. It copies each `prereg/*.ots` to a scratch directory, runs `ots upgrade` on the copy (tolerating proofs that are still pending), and posts a notice naming every proof whose copy changed, meaning an upgrade is available. It then runs `python -m judge ots-status --fail-pending-hours 48`, which prints each proof's state and fails the job if a committed proof is missing, invalid or for other bytes, or still has no Bitcoin attestation 48 hours after the `.ots` was first committed. To upgrade, run `ots upgrade prereg/NN-name.json.ots` locally and commit the result through a normal PR (the `.bak` file it leaves is gitignored).
