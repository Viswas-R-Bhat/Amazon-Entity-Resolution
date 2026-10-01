"""Generate a self-contained Kaggle notebook and an optional data upload archive."""
import argparse
import base64
import hashlib
import io
import json
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
OUT = WORKSPACE / 'kaggle'


def cell(kind, text):
    value = {'cell_type': kind, 'metadata': {}, 'source': textwrap.dedent(text).strip() + '\n'}
    if kind == 'code':
        value.update(execution_count=None, outputs=[])
    return value


def build_notebook():
    OUT.mkdir(exist_ok=True)
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
        for directory in ('src', 'tests'):
            for path in sorted((ROOT / directory).rglob('*.py')):
                archive.write(path, path.relative_to(ROOT).as_posix())
        for name in ('README.md', 'requirements.txt', 'LICENSE', 'run.ps1'):
            archive.write(ROOT / name, name)
        archive.write(ROOT / 'tools/improve_cached.py', 'tools/improve_cached.py')
    data = bundle.getvalue()
    payload = base64.b64encode(data).decode('ascii')
    bundle_hash = hashlib.sha256(data).hexdigest()
    cells = [cell('markdown', '''
        # Business Entity Resolution — local model on Kaggle

        **Accelerator: None (CPU).** This pipeline uses SQLite search and CPU LightGBM.
        Keep the notebook and challenge dataset **private**. No business lookups, geocoding,
        paid APIs, or external training records are used.

        Upload `amazon_entity_resolution_dataset.zip` using **Input → Upload**, choose a
        private dataset, and attach it. It contains the original seven TSV files in
        `dataset/train/` and `dataset/test/`. The code is already embedded in this notebook.

        Start with `RUN_MODE = "smoke"` to verify the environment. Its reduced candidate pool
        makes scores optimistic and its outputs must never be submitted. After that works,
        set `RUN_MODE = "full"` and run all cells for the complete corpus.
        Indexing and prediction checkpoint to disk. Kaggle sessions are time-limited:
        preserve/download work files if you need to continue in a new session. A saved
        notebook draft alone does not preserve runtime files.
    '''), cell('code', '''
        from pathlib import Path
        import os, sys, json, shutil, subprocess, importlib.util, importlib.metadata

        RUN_MODE = "full"                # Smoke trial has passed; use actual challenge data.
        RUN_STAGE = "train"              # all | index_train | train | improve | index_test | predict
        TRAIN_SAMPLE = 20_000             # Entire test dataset is still predicted in full mode.
        CANDIDATE_CAP = 60
        THREADS = min(4, os.cpu_count() or 2)
        DATASET_OVERRIDE = ""             # Leave blank to discover the attached dataset.
        ROOT = Path("/kaggle/working/entity_resolution")
        INPUT_ROOT = Path("/kaggle/input")
        ROOT.mkdir(parents=True, exist_ok=True)
        assert RUN_MODE in {"smoke", "full"}
        assert RUN_STAGE in {"all", "index_train", "train", "improve", "index_test", "predict"}
        print("Python:", sys.version.split()[0], "| CPU threads:", THREADS)
        print("Working disk free:", round(shutil.disk_usage(ROOT).free / 2**30, 1), "GiB")
    '''), cell('markdown', '''
        ## Install the small matching dependencies
        Internet is needed for package installation only. Keep Kaggle's preinstalled NumPy,
        SciPy and scikit-learn to avoid breaking its environment. Exact installed versions
        are recorded alongside the outputs for reproducibility.
    '''), cell('code', '''
        matching_packages = {"anyascii": "0.3.3", "lightgbm": "4.6.0", "rapidfuzz": "3.14.5"}
        needed = []
        for package, version in matching_packages.items():
            try:
                installed = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                installed = None
            if installed != version:
                needed.append(f"{package}=={version}")
        if importlib.util.find_spec("sklearn") is None:
            needed.append("scikit-learn==1.6.1")
        if needed:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *needed])
        import sqlite3
        with sqlite3.connect(":memory:") as connection:
            connection.execute("CREATE VIRTUAL TABLE probe USING fts5(text)")
        versions = {name: importlib.metadata.version(name) for name in
                    ["anyascii", "lightgbm", "rapidfuzz", "numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl"]}
        (ROOT / "runtime_versions.json").write_text(json.dumps(versions, indent=2))
        (ROOT / "requirements-kaggle-resolved.txt").write_text("\\n".join(f"{k}=={v}" for k,v in versions.items()) + "\\n")
        print(versions)
    '''), cell('markdown', '''
        ## Restore the embedded, MIT-licensed source code
        This cell contains the complete pipeline and its tests. It downloads no repository
        or model. The checksum verifies the embedded archive before extraction.
    '''), cell('code', f'''
        import base64, hashlib, io, zipfile
        CODE_B64 = {payload!r}
        code_bytes = base64.b64decode(CODE_B64)
        assert hashlib.sha256(code_bytes).hexdigest() == {bundle_hash!r}
        CODE = ROOT / "code" / "business_entity_resolution"
        CODE.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(code_bytes)) as bundle:
            for item in bundle.infolist():
                destination = (CODE / item.filename).resolve()
                if not destination.is_relative_to(CODE.resolve()):
                    raise ValueError("Invalid embedded archive path")
            bundle.extractall(CODE)
        sys.path.insert(0, str(CODE / "src"))
        ENV = dict(os.environ, PYTHONPATH=str(CODE / "src"), OMP_NUM_THREADS=str(THREADS))
        print("Pipeline restored:", CODE)
    '''), cell('markdown', '''
        ## Locate inputs and inspect data cleaning

        The preview checks headers, ID prefixes, missing fields and scripts on a bounded
        sample. **Every record** later receives the same deterministic cleaning during
        indexing or inference:

        - Normalize Unicode, case, punctuation, spacing and generic address abbreviations.
        - Preserve the original-script view, including Indic combining marks; add a separate ASCII transliteration.
        - Use a separate legal-suffix/name-order blocking view, retaining the fuller name for scoring.
        - Treat missing addresses explicitly; two missing values never count as a perfect match.
        - Preserve IDs, house numbers and distinct S2/S3 records. Do not silently discard duplicates.
        - Match only within the same country; France and other country strings remain supported.

        No input file is modified. Diagnostics are saved to `cleaning_report.json`.
    '''), cell('code', '''
        from er.kaggle_support import find_dataset, cleaning_preview
        from er.io import dump, source_paths
        DATASET = find_dataset(INPUT_ROOT, ROOT / "input_view", DATASET_OVERRIDE)
        report = cleaning_preview(DATASET, sample_limit=10_000)
        dump(ROOT / "cleaning_report.json", report)
        print("Dataset:", DATASET)
        print(report["note"])
        for file in report["files"]:
            print(file["file"], "| sampled:", file["sampled_rows"],
                  "| missing addresses:", file["missing_addresses"],
                  "| non-ASCII names:", file["non_ascii_names"])
        for example in report["examples"][:4]:
            print("\\nName:", example["original_name"], "→", example["normalized_name"])
            print("Address:", example["original_address"], "→", example["normalized_address"])
    '''), cell('markdown', '''
        ## Verify code and choose the dataset for this run
        The smoke fixture retains 1,000 real training businesses and their true targets
        plus random distractors. It reuses training records as synthetic test inputs solely
        to check the pipeline. Full mode uses the actual separate test files.
    '''), cell('code', '''
        subprocess.check_call([sys.executable, "-m", "unittest", "discover", "-s", str(CODE / "tests"), "-q"], env=ENV)
        def run_er(*arguments):
            subprocess.run([sys.executable, "-u", "-m", "er", *map(str, arguments)], env=ENV, check=True)

        RUN_ROOT = ROOT / RUN_MODE
        WORK = RUN_ROOT / "work"
        OUTPUT = RUN_ROOT / "output"
        if RUN_MODE == "smoke":
            RUN_DATA = RUN_ROOT / "dataset"
            if not (RUN_DATA / "SMOKE_ONLY.json").exists():
                run_er("make-smoke", "--dataset", DATASET, "--destination", RUN_DATA,
                       "--entities", 1000, "--distractors", 15000)
            sample_size = 1000
        else:
            RUN_DATA = DATASET
            sample_size = TRAIN_SAMPLE
        print("Mode:", RUN_MODE, "| Stage:", RUN_STAGE, "| Working files:", WORK)
        print("This run needs disk for indexes and checkpoints; stop if the available disk becomes low.")
    '''), cell('markdown', '''
        ## Build candidates, train, and evaluate
        The supervised sample splits whole businesses into fit/tune/evaluation groups.
        Held-out target variants are excluded from supervised negatives too. Threshold
        selection uses entity-macro F0.5, including no-match businesses. The fixed model
        is evaluated on untouched groups; candidate recall and the oracle ceiling are reported.

        These stages can take a long time on the full target catalog. Reuse a completed
        model by setting `RUN_STAGE` to `index_test` or `predict` instead of retraining.
    '''), cell('code', '''
        if RUN_STAGE in {"all", "index_train", "train"}:
            run_er("index", "--dataset", RUN_DATA, "--work", WORK, "--split", "train")
        if RUN_STAGE in {"all", "train"}:
            run_er("train", "--dataset", RUN_DATA, "--work", WORK, "--sample-size", sample_size,
                   "--cap", CANDIDATE_CAP, "--threads", THREADS)
        if (WORK / "validation_report.json").exists():
            evaluation = json.loads((WORK / "validation_report.json").read_text())
            for group in ("tune", "eval"):
                values = evaluation["folds"][group]
                print(group, {key: values[key] for key in ["entities", "macro_f05", "precision_micro", "recall_micro", "f1_micro", "exact_match_rate", "candidate_recall_micro", "candidate_oracle_macro_f05"]})
            if RUN_MODE == "smoke":
                print("SMOKE ONLY: the reduced candidate universe makes this score optimistic.")
    '''), cell('markdown', '''
        ## Predict all references and validate both TSV files
        Candidate lists are exactly the pairs scored by the model after pruning. Each
        reference gets one row; a reference can match zero, one or several targets.
        Both files are assembled only after all references finish. Completed checkpoints
        are reused within a preserved work directory; changing the model requires a new output directory.
    '''), cell('code', '''
        if RUN_STAGE in {"all", "index_test"}:
            run_er("index", "--dataset", RUN_DATA, "--work", WORK, "--split", "test")
        if RUN_STAGE in {"all", "predict"}:
            run_er("predict", "--dataset", RUN_DATA, "--work", WORK, "--output", OUTPUT, "--threads", THREADS)
        if (OUTPUT / "validation.json").exists():
            print(json.loads((OUTPUT / "validation.json").read_text()))
            print("Completed outputs:", OUTPUT)
            if RUN_MODE == "smoke":
                print("DO NOT SUBMIT these trial files. Set RUN_MODE='full' for the competition files.")
    '''), cell('markdown', '''
        ## Download results and preserve progress
        For a completed full run, download `matching_results.tsv` and `candidate_pairs.tsv`
        from `entity_resolution/full/output/`, plus the model and report in `full/work/`.
        The leaderboard receives the matching TSV. The final submission archive needs
        both TSVs, source code and the filled methodology document.

        Kaggle's editor currently describes a 20 GB saved-output limit. Check the sizes
        below before saving a version. Runtime indexes are regenerable but are needed for
        resuming and revalidating. Save/download the desired artifacts before ending the session.
        A full-corpus score is not claimed until a full-catalog validation run has finished.
    '''), cell('code', '''
        from IPython.display import display, FileLink
        for relative in ["cleaning_report.json", "runtime_versions.json", "requirements-kaggle-resolved.txt",
                         f"{RUN_MODE}/work/validation_report.json", f"{RUN_MODE}/work/model.txt",
                         f"{RUN_MODE}/output/matching_results.tsv", f"{RUN_MODE}/output/candidate_pairs.tsv"]:
            path = ROOT / relative
            if path.exists():
                display(FileLink(str(path)))
        total = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file() and not p.is_symlink())
        print("Working files:", round(total / 2**30, 2), "GiB; free disk:", round(shutil.disk_usage(ROOT).free / 2**30, 1), "GiB")
    ''')]
    cells.extend([cell('markdown', '''
        ## Optional cached model improvement
        Set RUN_STAGE="improve" only after training completes. This audits cached labels
        and reproduces baseline metrics, then selects among three settings on tuning
        businesses. It evaluates the selected model once on the existing evaluation split.
        The original model is preserved; alternatives and comparison.json are written
        to full/improved_cached. This does not automatically change prediction's model.
    '''), cell('code', '''
        if RUN_STAGE == "improve":
            subprocess.run([sys.executable, "-u", str(CODE / "tools/improve_cached.py"),
                            str(WORK)], env=ENV, check=True)
    ''')])
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name':'Python 3', 'language':'python', 'name':'python3'},
                                          'language_info': {'name':'python'}}, 'nbformat':4, 'nbformat_minor':5}
    for index, item in enumerate(cells):
        item['id'] = f'er-cell-{index:02d}'
        if item['cell_type'] == 'code':
            compile(item['source'], f'cell_{index}', 'exec')
    target = OUT / 'business_entity_resolution_kaggle.ipynb'
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'Notebook: {target} ({target.stat().st_size:,} bytes)', flush=True)


def build_dataset():
    dataset = WORKSPACE / '6ab10eb3b23ba_student_resource/student_resource/dataset'
    files = sorted(dataset.rglob('*.tsv'))
    if len(files) != 7:
        raise ValueError('Expected exactly seven challenge TSV files')
    target = OUT / 'amazon_entity_resolution_dataset.zip'
    if target.exists():
        raise ValueError('Dataset archive exists; use the existing archive or choose a new location')
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as archive:
        for path in files:
            archive.write(path, 'dataset/' + path.relative_to(dataset).as_posix())
            print(f'Packed {path.name}', flush=True)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC verification failed')
    print(f'Dataset archive: {target} ({target.stat().st_size / 2**20:.1f} MiB)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-archive', action='store_true')
    arguments = parser.parse_args()
    build_notebook()
    if arguments.dataset_archive:
        build_dataset()
