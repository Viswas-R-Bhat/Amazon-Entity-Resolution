# Cached model improvement — 26 September 2026

Measured in the existing private Kaggle notebook against the full training target catalog.
Original model and retrieval are unchanged; the alternative is in
`/kaggle/working/entity_resolution/full/improved_cached/`.

| Evaluation metric | Baseline | Selected alternative |
|---|---:|---:|
| Entity macro F0.5 | 0.9073949333162753 | 0.912783269175377 |
| Micro precision | 0.9751886001676446 | 0.9788997739261492 |
| Micro recall | 0.822830468915765 | 0.8268618714194781 |
| Micro F1 | 0.8925543749280755 | 0.8964803312629399 |
| Exact-match rate | 0.5384615384615384 | 0.5502204801567859 |

Selected on 4,004 tuning businesses: 500 trees, 47 leaves, learning rate 0.04,
minimum child samples 60, L2 regularization 5, threshold 0.66. Three prespecified
trials took 35.48, 40.05 and 41.58 seconds. No retrieval rerun was needed.
Evaluation contains the same 4,082 previously inspected businesses, not a new blind test.
This is not a hidden-test or leaderboard score. Retrieval cap remains 60.

Audit confirmed:
- Original model checksum and original validation metrics reproduced.
- All cached pair labels agree with saved group truth; fit pair count reproduced.
- All 20,000 group labels checked against the original organizer ground-truth TSV.
- No shared truth target IDs between fit/tune/evaluation groups in this sample.
- Independent arithmetic reproduced the selected model's macro F0.5.
- 21 local tests passed, including an independent randomized metric check.

There are still 252 false-positive links on evaluation. This classifier selects existing
candidate IDs rather than generating businesses; that does not make its matches infallible.
No claim is made that all possible leakage, data defects, or generalization failures are ruled out.

Country diagnostics for selected alternative: India F0.5 0.8832147744 and candidate
recall 0.8416264389; US F0.5 0.9315328671 and candidate recall 0.9247115275.
France is not evaluated here.

Corrections to earlier advice: the common-name cutoff skips only the exact-core pass;
other passes may still retrieve those businesses. Its impact was not measured.
The 95.69% candidate oracle is a theoretical ceiling for current candidates, not a
forecast. The roughly ten-minute training duration excludes initial index construction.
The cached tuning improvement does not implement the proposed address or retrieval changes.

Kaggle contains additional cells for tuning, original-label auditing, and a compact
`improved_model_audited.zip` backup. The local notebook generator includes an optional
`RUN_STAGE="improve"` stage. Future prediction must explicitly select the alternative;
the original prediction cell still uses the baseline work directory.
