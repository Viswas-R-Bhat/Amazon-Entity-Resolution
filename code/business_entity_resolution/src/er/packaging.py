import datetime
import json
import zipfile
from pathlib import Path
from .inference import validate
from .io import digest, log


def package(dataset, work, output, team, members, archive):
    dataset, work, output, archive = map(Path, (dataset, work, output, archive))
    if (dataset / 'SMOKE_ONLY.json').exists():
        raise ValueError('Cannot package smoke outputs for submission.')
    validation = validate(dataset, work, output)
    config = json.loads((work / 'model_config.json').read_text(encoding='utf-8'))
    run = json.loads((output / 'run_summary.json').read_text(encoding='utf-8'))
    report = json.loads((work / 'validation_report.json').read_text(encoding='utf-8'))
    if digest(work / 'model.txt') != config['model_sha256'] or run['model_sha256'] != config['model_sha256']:
        raise ValueError('Output provenance does not match the supplied model.')
    evaluation = report['folds']['eval']
    documentation = f'''# ML Challenge 2026: Business Entity Resolution

**Team Name:** {team}
**Team Members:** {members}
**Submission Date:** {datetime.date.today().isoformat()}

## 1. Executive Summary
An offline CPU pipeline retrieves candidates from an on-disk SQLite FTS5 catalog and classifies pairs with LightGBM. Memory stays independent of the full target corpus; final matches may contain zero, one or multiple targets.

## 2. Methodology
### 2.1 Problem Analysis
Names and addresses contain spelling, ordering, suffix and script variation. Common names and missing target addresses cause ambiguity. France has no labeled training examples, so country is only an equality gate, never a closed-set model feature.
### 2.2 Solution Strategy
**Approach Type:** Multi-pass retrieval plus supervised pair classifier.
**Core contribution:** Resumable disk search and bounded batches fit a 16 GB personal computer. Deterministic Unicode/ASCII views use no external business records or APIs.

## 3. Candidate Generation (Blocking)
Country-gated exact sorted name keys, rare name words, consonant keys, two-word intersections, name/address intersections and address pairs are unioned. Word searches use BM25; a fuzzy name/address pre-ranker retains at most {config['cap']} candidates per S1. Large exact-name blocks are handled by textual search instead of unbounded expansion.

Final test candidate pairs: **{validation['candidate_pairs']:,}**. This exact set is scored and exported in candidate_pairs.tsv. Retrieval is lossy: untouched validation micro candidate recall is {evaluation['candidate_recall_micro']:.6f}; macro recall on nonempty truths is {evaluation['candidate_recall_macro_nonempty']:.6f}. Candidate-oracle macro F0.5 is {evaluation['candidate_oracle_macro_f05']:.6f}. No missed truths are injected into validation candidates.

## 4. Matching Model
Name features: edit/token similarities, exact normalized/core equality, lengths, original-script and consonant similarity. Address features: edit/token similarity, missingness, numeric overlap/conflict. Other features: source, name block frequency, retrieval score and exact-block flag.

Model: MIT-licensed LightGBM, {config['trees']} small decision trees, far below 8B parameters; no foundation model. A deterministic hash assigns entire S1 label groups to fit/tune/eval (60/20/20). {report['sample_entities']:,} reference entities are sampled reproducibly; all provided train targets form the unsupervised retrieval catalog. Variants owned by tune/eval entities are excluded from supervised fit negatives as well as positives. No supervised aliases are learned. The threshold {config['threshold']:.6f} maximizes macro F0.5 on tune entities; the model is not refit on tune/eval afterward.

## 5. Results & Error Analysis
Untouched evaluation macro F0.5: **{evaluation['macro_f05']:.6f}** across {evaluation['entities']:,} entities. This is a local training-domain estimate, not a hidden-test score. Wrong links: {evaluation['false_positive_links']:,}; missed links: {evaluation['false_negative_links']:,}. Per-country metrics and concrete FP/FN/retrieval-miss examples are in validation_report.json. Unseen French data and severe cross-script spelling differences remain unmeasured generalization risks.

## 6. Conclusion
The baseline is reproducible and runs entirely on local CPU and disk. Candidate recall limits the final score; broader retrieval and better transliteration should be evaluated against the supplied error examples before increasing model complexity.

## Appendix
Code: code/business_entity_resolution/src/er/. Windows entry: run.ps1; cross-platform entry: python -m er. Reproduction instructions and pinned dependencies are in README.md and requirements.txt. Trained artifacts are in model/. Both mandatory TSVs contain all {validation['entities']:,} test references; strict local validation checks coverage, unique IDs, target existence and match-subset relationships.
'''
    if archive.exists():
        raise ValueError('Archive already exists; choose a new archive name.')
    archive.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as bundle:
        for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
            bundle.write(output / name, f'output/{name}')
        for name in ('README.md', 'requirements.txt', 'LICENSE', 'run.ps1'):
            bundle.write(root / name, f'code/business_entity_resolution/{name}')
        for directory in ('src', 'tests'):
            for path in sorted((root / directory).rglob('*.py')):
                bundle.write(path, f'code/business_entity_resolution/{path.relative_to(root).as_posix()}')
        for name in ('model.txt', 'model_config.json', 'validation_report.json'):
            bundle.write(work / name, f'code/business_entity_resolution/model/{name}')
        bundle.writestr('Documentation_template.md', documentation)
    log(f'Submission archive ready: {archive}')
