# Kaggle notebook

Open `business_entity_resolution_kaggle.ipynb` in Kaggle. It contains the pipeline source and tests, so no separate code upload is needed.

## Input

Upload `amazon_entity_resolution_dataset.zip` as a **private** Kaggle dataset and attach it to the notebook. This archive contains only the seven original TSV files from:

`C:\Users\VISWAS\OneDrive\Desktop\Amazon\6ab10eb3b23ba_student_resource\student_resource\dataset`

Both `train` and `test` are required. The notebook discovers their paths automatically.

## Settings and first run

- Accelerator: **None (CPU)**. SQLite retrieval and this LightGBM implementation use the CPU.
- Internet: On for the initial package installation.
- Keep the notebook and dataset private.
- The initial smoke trial passed. The current notebook defaults to `RUN_MODE = "full"` and `RUN_STAGE = "train"`. Run All builds/reuses the full training index, trains, and evaluates without starting test prediction.
- For a fresh smoke trial, select `RUN_MODE = "smoke"` and `RUN_STAGE = "all"`.

The cleaning section reports missing values and shows original/normalized examples. Every record is normalized during indexing or inference, with Unicode and ASCII views, punctuation/spacing normalization, address abbreviations, and explicit missing-value features. Original files and entity IDs are preserved.

The smoke run uses a small training-derived fixture to verify the workflow. Its scores are optimistic because the candidate catalog is reduced, and its output files must not be submitted.

## Full run

After the smoke run passes, change `RUN_MODE` to `"full"`. Training samples 20,000 reference businesses by default, searches the full target catalog, and then predicts every actual test reference. Indexing and prediction save checkpoints in the working directory. Full-catalog runtime and quality must be measured; they are not established by the smoke run.

Kaggle sessions have runtime and saved-output limits. Preserve the working files before ending a session; a saved code draft does not preserve runtime files. Reuse preserved work with the appropriate `RUN_STAGE` rather than training again.

Final output paths:

- `/kaggle/working/entity_resolution/full/output/matching_results.tsv`
- `/kaggle/working/entity_resolution/full/output/candidate_pairs.tsv`
- `/kaggle/working/entity_resolution/full/work/validation_report.json`
- `/kaggle/working/entity_resolution/cleaning_report.json`

The leaderboard uses the matching TSV. The final archive also needs candidate pairs, source code, and the filled methodology document.

## Retrieval optimization verified September 26

The country filter now runs as an ordinary record filter after text retrieval rather than as a zero-weight BM25 query term. Country restrictions, candidate cap, ranking weights, and tie breaking are retained; the existing full index is reusable.

Training retrieves with up to four worker processes, returning results in original order. Kaggle checks found identical complete candidate results on 30 businesses for the query change and identical serial/parallel results on 80 businesses. The parallel benchmark processed 7.56 businesses/second; the restarted training run reached 7.5 businesses/second at its first 250-business checkpoint, versus approximately 0.7 previously. These are early measurements, not a completion-time guarantee. No training sample reduction was used.
