# Local business entity resolution baseline

Runs on Windows/Python 3.13, CPU and local disk. It uses only the supplied records and training labels. No AWS account, paid service, GPU, business lookup, geocoder, or downloaded model is used. Internet access is needed only to install the pinned software dependencies once.

## Run on this PC

Open PowerShell in `C:\Users\VISWAS\OneDrive\Desktop\Amazon`.

```powershell
# Small end-to-end check (safe default)
& .\code\business_entity_resolution\run.ps1 -Mode smoke

# Full dataset: build indexes, train, predict, validate
& .\code\business_entity_resolution\run.ps1 -Mode full
```

The launcher creates its own `.venv`. The smoke run writes only under `runs/smoke/`. It samples 1,000 real training references, retains their true targets plus 15,000 random distractors per target source, and reuses those records as synthetic test inputs. **Smoke scores are not full-corpus or leaderboard estimates. Do not submit smoke outputs.**

The full run writes persistent indexes and the trained model under `runs/local/work/`. It writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` only after finishing every test reference. Logs show progress. Indexing and prediction are restartable; press Ctrl+C to stop, then repeat the command. Training restarts its sampled feature extraction. Keep the machine awake and connected to power for the full run.

If PowerShell blocks local scripts, invoke this one script without changing your permanent policy:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\business_entity_resolution\run.ps1 -Mode full
```

Use `-Dataset 'D:\path\to\dataset'` for a relocated dataset. This path must contain `train/` and `test/`. `-SampleSize 20000` and `-Threads 4` are the defaults. Increasing the supervised sample increases training memory and runtime; it does not change how many test references are processed.

## Hardware and runtime

Designed for the available i7 CPU and 16 GB RAM. SQLite keeps the complete target catalog on disk with a 96 MiB page cache; inference scores at most 500 references × 60 candidates per batch. Training uses disk-backed feature arrays and a 20,000-reference reservoir sample. LightGBM uses four CPU threads and a 128 MiB histogram cache. These limits are component budgets, not an operating-system hard memory limit. Close heavy applications for a full run.

Expect a long full run: there are roughly 20 million target records to index across train and test and 1.73 million references to predict. Small-catalog smoke timing does not predict full-catalog search time. Reserve at least 30 GB of free disk initially; actual size depends on indexes, candidates and checkpoints. Prediction checkpoints temporarily duplicate output content. This directory is under OneDrive on this PC, which may add sync overhead; an unsynced location can be used through the explicit CLI work/output paths below.

## Method

1. Preserve an original-script Unicode view and derive deterministic ASCII transliterations with AnyAscii. Normalize punctuation, common legal suffixes for one blocking view, and a short generic address abbreviation list. No external business information is introduced.
2. Index all S2/S3 targets in SQLite FTS5. Retrieve within the same country using exact sorted name keys, rare words, consonant keys, word intersections, name/address intersections and address pairs. Common exact-name blocks do not expand without bounds.
3. Union retrieved targets and rank by fuzzy name/consonant/address similarity. Keep at most 60 candidates. This cap can lose true matches: the report measures recall and the achievable score ceiling.
4. Train an MIT-licensed LightGBM classifier using 26 name, address, number-conflict, missingness, source, frequency and retrieval features. No country one-hot vocabulary or country-specific model excludes France.
5. Split entire S1 businesses by a stable hash: 60% fit, 20% threshold tuning, 20% untouched evaluation. Each S1's positive targets remain in that group. Target variants owned by tuning/evaluation businesses are also excluded from supervised fit negatives. All targets remain available as an unlabeled retrieval catalog, as they would at inference. No missed truths are inserted into candidates and no label-derived aliases are used.
6. Choose one threshold on tuning entities using the official entity-macro F0.5. Keep the fixed model for untouched evaluation and test prediction. Emit any number of matching targets, including none. There is no one-match-per-S1 restriction.

For each entity, the score is `1.25 TP / (1.25 TP + FP + 0.25 FN)`; both truth and prediction empty score 1. All reference entities, including those with no retrieved candidates, count in the average. The report includes per-country scores, candidate recall, oracle score, feature importance and concrete FP/FN examples.

The final `candidate_pairs.tsv` contains **exactly the candidates passed to LightGBM**, after retrieval pruning and before thresholding. Matches must be a subset. Output order follows source1 input order, with one row per reference and comma-separated target IDs in the second TSV field.

## Individual stages and portable CLI

The Windows launcher also supports `index-train`, `train`, `index-test`, `predict`, and `validate`. Example:

```powershell
& .\code\business_entity_resolution\run.ps1 -Mode train -SampleSize 30000
```

For explicit paths, activate a Python environment with the pinned requirements and set `PYTHONPATH` to the `src` directory. Commands below are run from `code/business_entity_resolution/`; replace the dataset path as appropriate.

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$data = '..\..\6ab10eb3b23ba_student_resource\student_resource\dataset'
$work = '..\..\runs\local\work'
$out = '..\..\output'
$py = '.\.venv\Scripts\python.exe'
& $py -m er index --dataset $data --work $work --split train
& $py -m er train --dataset $data --work $work --sample-size 20000 --cap 60 --threads 4
& $py -m er index --dataset $data --work $work --split test
& $py -m er predict --dataset $data --work $work --output $out --threads 4
& $py -m er validate --dataset $data --work $work --output $out
```

On Linux/macOS, use `python -m venv .venv`, install `requirements.txt`, set `PYTHONPATH=src` and run the same `python -m er` subcommands. SQLite must include FTS5. Model/source feature and retrieval versions must agree; input changes require a fresh work directory. Changed model/configuration requires a fresh output directory. Full file-content hashes protect the model and prediction checkpoints; input compatibility checks use absolute path, file size and modification time.

To reuse the shipped model from an archive, copy its `model/` files into an empty work directory, index the supplied test dataset there, then run `predict`. Training data is needed only to retrain.

## Validation and submission

The local validator streams outputs and uses disk-backed uniqueness checks. It checks all test references, every target's existence, duplicate rows/IDs, headers and the match/candidate subset. Failures stop the run. It requires output rows in input order, which is stricter than the official format but guaranteed by this pipeline.

The organizer's independent validator can also be run from the workspace root:

```powershell
python .\6ab10eb3b23ba_student_resource\student_resource\utils\validate_submission.py --matching .\output\matching_results.tsv --candidate .\output\candidate_pairs.tsv --test-dir .\6ab10eb3b23ba_student_resource\student_resource\dataset\test --check-ids
```

The organizer validator loads large sets into RAM; close other applications before a full run. The local validator performs these checks with bounded memory and treats subset violations as errors, rather than warnings.

After a successful full run, create the final archive from the code directory:

```powershell
& $py -m er package --dataset $data --work $work --output $out --team 'YOUR TEAM' --members 'YOUR NAMES' --archive '..\..\YOUR_TEAM_submission.zip'
```

Packaging revalidates outputs, refuses marked smoke datasets, checks model provenance and fills the organizer's methodology sections with actual results. It includes both mandatory TSVs, code, pinned requirements, README, trained model and validation report. It does not include the provided datasets, virtual environment or indexes. Team details must be supplied by you.

## Tests and limitations

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover metric semantics, empty-candidate references, threshold selection, missing-value guards, transliteration, France, multiple matches, country separation, index reuse/input changes, and malformed outputs. End-to-end real-data smoke testing additionally covers model fitting, prediction, checkpoint reuse, and the organizer validator.

This is a baseline, not a claim of an optimal score. Generic transliteration is imperfect for Indic-to-English spelling; word retrieval can miss heavily corrupted names/addresses; candidate truncation can drop positives. Validation labels do not cover France. Inspect `validation_report.json` from a full-catalog training run before interpreting model quality.

Code and final tree model use MIT licensing. LightGBM is MIT; AnyAscii is an ISC-licensed deterministic text utility, not a matching model. RapidFuzz is MIT; the remaining scientific Python dependencies carry their respective permissive licenses. No pretrained neural model is used. References: [LightGBM parameters](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html), [AnyAscii](https://pypi.org/project/anyascii/).
