# Leakage and overfitting audit — 26 September 2026

## Conclusion

The checks below found no evidence that the roughly 91% entity-macro F0.5 is caused
by direct label leakage or exact business duplicates across the checked splits.
This does not certify absence of every possible leakage mechanism. Generalization
to a new country is a material unresolved risk, and validation is no longer a blind
test after repeated inspection.

## Measured checks

| Check | Result |
|---|---|
| Improved model, fit / tune / evaluation F0.5 | 92.85% / 91.33% / 91.28% |
| Fresh 2,000 references, frozen model and threshold | 90.59% F0.5, 97.78% precision, 82.04% recall |
| Fresh subset whose true targets were never supervised, even as negatives | 1,765 references; 90.16% F0.5 |
| Fresh references sharing true target IDs with fit positives | 0 |
| Shared true target IDs between original folds | 0 |
| Evaluation reference exact normalized core-name/address duplicates in fit | 0 |
| Evaluation true targets with exact normalized content aliases in fit positives | 0 |
| Evaluation F0.5 on names absent from fit references | 91.60% on 3,744 businesses |
| Evaluation F0.5 excluding singletons | 91.21% on 3,859 businesses |
| Always predict no matches | 5.46% F0.5 |
| No exact-name-and-address true candidate | 90.69% F0.5 on 3,524 positive businesses |
| Independently rebuilt cached feature rows | 100 / 100 matched |
| Bootstrap interval on original evaluation | Approximately 90.66–91.80%; reference-level resampling, 1,000 draws |
| Refit without held-out-label-based candidate exclusion | 91.30% evaluation F0.5 |
| Shuffled training labels, real tune/evaluation labels | 8.38% evaluation F0.5; diagnostic control, not a formal permutation significance test |
| Full train reference text duplicates | 0 of 2,206,821 |
| Full test references whose exact text occurs in train references | 0 of 1,732,544 |

The full reference overlap scan used country/name/address with case folding and
collapsed whitespace. It does not rule out approximate aliases, branch relationships,
shared source provenance or synthetic corruption-pattern overlap.

## Findings and limitations

1. The code splits by reference ID, rather than by connected components of approximate
   business identities. The organizer describes source 1 as deduplicated. The checks
   above support that description for exact matches, but cannot prove fuzzy duplicates
   are absent. 338 evaluation references share a normalized core name with fit references;
   those score lower (87.72%), rather than explaining the high aggregate score.
2. The training pipeline uses held-out target IDs to remove 1,910 fit candidate pairs.
   This is an indirect dependency on held-out labels. The removal is intended to prevent
   target reuse in supervision; it does not expose held-out labels as model features.
   The diagnostic refit with all these pairs retained scores 91.30%, versus 91.28%, so
   this mechanism does not explain the reported high score in this experiment.
3. Retrieval uses only supplied target records, not matching labels, and does not force
   true matches into candidates. All 10,320,219 training targets are indexed. Token
   frequencies use the available target catalog, analogous to the available test catalog.
4. The 26 features contain similarities, missingness, numeric agreement, source prefix,
   target-name frequency and retrieval metadata. Raw IDs, ground-truth membership and
   fold labels are not model features. Normalization is deterministic, not fitted to labels.
5. Threshold and the three alternative model settings were selected on tuning references.
   Evaluation results have since been inspected repeatedly; they should not be described
   as a fresh blind test. The fresh 2,000-reference check used seed 20260926 and threshold
   0.66 without retraining or threshold adjustment. It is still the same training domain.
6. The formula matches the organizer's entity-macro F0.5 including singleton semantics.
   This is not pair accuracy or micro F0.5. High precision makes a high F0.5 possible even
   with missed matches: recovering 3 of 4 links with no false positives scores 93.75%.
7. The shuffled-label control is only a sanity check; its failure does not by itself
   prove absence of leakage. The bootstrap interval likewise excludes domain-shift and
   repeated-model-selection uncertainty and assumes independent reference units.

## Country transfer risk

Models trained and threshold-tuned exclusively in one country were evaluated in the
other country's evaluation references:

| Experiment | F0.5 |
|---|---:|
| Train/tune US, evaluate India | 78.18% |
| Train/tune India, evaluate US | 87.72% |
| Normal two-country model on original India evaluation | 88.32% |
| Normal two-country model on original US evaluation | 93.15% |

These experiments confound domain shift with reduced training size but show a substantial
transfer risk. Test includes 259,452 France references (about 15% of test), and there are
no France labels to measure its score. No France or leaderboard score is claimed.

Audit cells and JSON reports are in the Kaggle notebook under
`/kaggle/working/entity_resolution/full/leakage_audit/`.
Original labels and independent score arithmetic were also verified in the previous
audit; see IMPROVEMENT_AUDIT.md. The full local overlap counts are in reference_overlap_audit.json.
