# Local verification — 25 September 2026

- Environment: Windows, Python 3.13.14, pinned dependencies installed in the project virtual environment.
- 19 automated tests passed, including interrupted index recovery, prediction checkpoint reuse, corrupted checkpoint rejection, metric edge cases, strict output checks and submission archive structure.
- End-to-end trial: 1,000 actual training reference businesses, 33,507 target records, 57,141 final candidate pairs.
- Fixed model trained on 33,575 retrieved pairs. Targets owned by held-out businesses were excluded from supervised fit pairs.
- Threshold tuning macro F0.5: 0.977846; selected threshold: 0.93.
- Untouched trial evaluation macro F0.5: 0.973084.
- Both generated trial TSVs passed the local strict validator and the organizer's validator with `--check-ids` enabled.
- Trial artifacts: `runs/smoke/` relative to the workspace root.

**These are reduced-catalog trial results.** The target pool is much smaller than the official dataset; synthetic test inputs reuse training records to exercise the output pipeline. This score is not a full-catalog validation estimate or a leaderboard score. Full training-catalog indexing, full test inference and final submission packaging have not been run.

Run `run.ps1 -Mode full` as described in README.md to process the complete dataset. Indexing and prediction checkpoints allow restarting after interruption.
