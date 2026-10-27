# tektradoor-judge

A private, pre-registered test rig for crypto trading strategies. Each candidate strategy is written down (rules, seed, code version) and timestamped before it is run on out-of-sample data, so the result cannot be tuned after the fact, by anyone including the owner. Costs, funding and a same-universe buy-and-hold benchmark are applied the same way to every candidate.

The repository is private. The public proof is the sha256 of each prereg file, posted publicly (and stamped with OpenTimestamps) before that candidate's Test run. When a prereg file is shown later, it can be checked against the posted hash and its stamp; the judge itself refuses to run if that file or the judge code changed after registration.

## Running a candidate

```
python -m judge run prereg/01-name.json --kind test
python -m judge run prereg/01-name.json --kind final        # after a Test pass
python -m judge run prereg/01-name.json --kind final_alone  # Test window under 12 months
```

Each run writes `results/{candidate_id}-{kind}.json`, pass or fail, with a manifest (archive checksums, prereg sha256, judge tree hash, seed, `pip freeze`, run time). Results are never overwritten, and Final opens once. See `prereg/README.md` for the files each candidate needs.

## Development

```
pip install --require-hashes -r requirements.txt
pytest -q
python scripts/check_no_secrets.py
```
