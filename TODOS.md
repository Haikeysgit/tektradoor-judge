# TODOS

## TODO-1: Phase 2 - live record
Build the live, unfakeable public record (requirements R2-1..R2-15 in the spec). Starts only after a candidate passes the Test window. The repo is public (Apache-2.0); the record rests on the committed rules and results, the posted hashes and the timestamps.

## TODO-2: Darwinex Zero check after a Test pass
After any candidate passes Test, check whether a Darwinex Zero track record is a viable independent verification route before going live.

## TODO-3: Final marker resume
Final is two steps: `open-final` writes results/{id}.final_opened and the operator commits it; `run --kind final` needs that committed, single-commit, matching marker. A Final run that fails (network error, short data, Ctrl-C) is simply rerun: the committed marker allows it while no Final result exists. Once a Final result is in git history, Final never runs again.

## TODO-4: Verify the OpenTimestamps proof (Task 9)
The judge does not check `prereg/NN-name.json.ots`. Task 9: require it committed and verify it (`ots verify`), and check that the attested block time agrees with `NN-name.stamp_confirmed`. Until then the stamp date and all git dates are local claims (README "Limits").

## TODO-5: repro results 01/02 Bitstamp hash range (first run only)
The first reproduction run's results 01/02 recorded the bitstamp:btcusd hash for 2015-08-08.. only (key collision, fixed in the commit "fix(judge): unique api price keys, tie repro gate to strategy and cost code"; find it with the oldest commit in `git log -S"TODO-5" -- TODOS.md`). The full BTC series hash was under usd:BTC. That run's result commit was dropped from the unpushed branch before merge (review round 1), and its numbers are kept in task-8-report.md. The rerun's results use range keys, so this entry is history only.

## TODO-6: Bitstamp zero-volume candles 2015-01-06..08
Bitstamp serves flat, zero-volume candles (O=H=L=C=276.80) for 2015-01-06..08, during its January 2015 halt. They are rows, not missing days, so the 3-day forward-fill rule (`prices.fill_gaps`) does not see them. Here they fall inside the warmup (weight 0 until day 91) and change no number. The fill rule is left unchanged by ruling. A later candidate or data source should treat zero-volume rows as missing for the gap rule, or at least log them, so that a longer exchange halt cannot pass as real prices.
