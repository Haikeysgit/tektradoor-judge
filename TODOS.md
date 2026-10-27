# TODOS

## TODO-1: Phase 2 - live record
Build the live, unfakeable public record (requirements R2-1..R2-15 in the spec). Starts only after a candidate passes the Test window. The repo stays private; the public record rests on posted hashes and timestamps, not on publishing the code.

## TODO-2: Darwinex Zero check after a Test pass
After any candidate passes Test, check whether a Darwinex Zero track record is a viable independent verification route before going live.

## TODO-3: Final marker resume
Final is two steps: `open-final` writes results/{id}.final_opened and the operator commits it; `run --kind final` needs that committed, single-commit, matching marker. A Final run that fails (network error, short data, Ctrl-C) is simply rerun: the committed marker allows it while no Final result exists. Once a Final result is in git history, Final never runs again.

## TODO-4: Verify the OpenTimestamps proof (Task 9)
The judge does not check `prereg/NN-name.json.ots`. Task 9: require it committed and verify it (`ots verify`), and check that the attested block time agrees with `NN-name.stamp_confirmed`. Until then the stamp date and all git dates are local claims (README "Limits").
