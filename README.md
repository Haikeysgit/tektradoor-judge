# tektradoor-judge

A public, pre-registered test rig for crypto trading strategies. Each candidate strategy is written down (rules, seed, code version) and timestamped before it is run on out-of-sample data, so the result cannot be tuned after the fact. Costs, funding and a same-universe buy-and-hold benchmark are applied the same way to every candidate.

## How to verify a result

1. Open the candidate's file in `prereg/NN-name.json`. It records the rules, the seed and `judge_tree_hash` (`git rev-parse HEAD:judge`).
2. Check that the `.ots` stamp matches the file (`ots verify prereg/NN-name.json.ots`) and that the file's sha256 was posted publicly before the Test run.
3. Check out this repo, confirm `git rev-parse HEAD:judge` equals the recorded `judge_tree_hash`, install pinned dependencies (`pip install --require-hashes -r requirements.txt`) and re-run. The judge refuses to run if a stamped file was edited.
4. Run `pytest -q` and `python scripts/check_no_secrets.py`.

## Spec

See the Phase 1 spec and plan in the project documents (link to be added when the repo is published).

## License

Apache-2.0. Copyright 2026 Tektradoor.
