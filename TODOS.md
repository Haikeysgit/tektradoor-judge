# TODOS

## TODO-1: Phase 2 - live record
Build the live, unfakeable public record (requirements R2-1..R2-15 in the spec). Starts only after a candidate passes the Test window. The repo stays private; the public record rests on posted hashes and timestamps, not on publishing the code.

## TODO-2: Darwinex Zero check after a Test pass
After any candidate passes Test, check whether a Darwinex Zero track record is a viable independent verification route before going live.

## TODO-3: Final marker resume
A Final run that fails after writing results/{id}.final_opened (network error, short data, Ctrl-C) can be rerun only when the marker's prereg sha256, judge tree hash and window all match and no Final result exists; any other marker is refused. The marker stays local until it is committed with the Final result.
