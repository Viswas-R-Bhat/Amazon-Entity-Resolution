# Submission and experiment ledger

Updated 2026-09-27. Scores below are user-reported leaderboard scores; unsubmitted variants have no claimed leaderboard score.

| Matching TSV folder | Reported leaderboard score | Notes |
|---|---:|---|
| `final` | 0.884324 | Original full baseline |
| `rarity_france_safe` | 0.902 | Rarity model for US/India, baseline for France; current confirmed best |
| `rarity_imputed_france_safe` | 0.902 | Same approach with address imputation; no visible leaderboard gain |
| `rarity_imputed` | 0.893 | Rarity model and imputation in every country |
| `rarity_model` | Not separately confirmed | Rarity model in every country, without imputation |
| `france_safe_unique_owner` | Pending | Starts from usual France-safe file; resolves duplicate target ownership |
| `rarity_unique_owner` | Pending | Starts from non-imputed full rarity file; resolves duplicate target ownership |
| `france_adapted_unique_owner` | Pending | Uses test-catalog word frequencies for France, then resolves ownership; US/India from France-safe ownership variant |

Every listed finished variant has all 1,732,544 source-1 rows. The three ownership variants passed the official matching format/coverage validator. Official validation was run without the optional memory-intensive ID-existence flag. Matched IDs originate in the supplied test catalog and existing candidate lists.

## Ownership shortcut

The challenge describes Source 1 as deduplicated. All 7,638,365 training ground-truth links assign each target to exactly one source-1 entity. The usual France-safe output contained 229,767 assignments above one per target; the full rarity output contained 728,489.

Disputed links were rescored using the same models and original features. Every original positive score was reproduced. For each disputed target, the highest raw model score wins only when its lead over the runner-up exceeds 0.1; otherwise all competing assignments are omitted. Other links remain unchanged. The ordinary validation query samples contained no ownership conflicts. A focused same-name rival benchmark was used as a diagnostic, not as a representative macro-F0.5 estimate.

`france_safe_unique_owner`: 236,157 links removed, 75,806 rows changed, 5,051,888 retained links. No target has multiple owners.

`rarity_unique_owner`: 738,702 links removed, 103,208 rows changed, 5,065,306 retained links. No target has multiple owners.

## France frequency adaptation

Training contains no France labels. `sarl` occurs in 3 training names versus 298,036 test names; `rue` occurs in 608 training addresses versus 557,995 test addresses. Training-only IDF overweights these ordinary French terms.

The adaptation rescored only the 1,503,732 existing positive French pairs using frequencies from the supplied, unlabeled test catalog. It retains a winner at probability >=0.5 with a raw-score ownership lead >0.1. This changed 40,357 rows relative to the France-safe ownership file and retains 760,703 French links. French accuracy remains unvalidated until leaderboard feedback.

## Candidate recovery in progress

The checkpointed rare-address search examines US/India queries with at most one original prediction. It retrieves postings for up to two address tokens with document frequency <=1,000, ranks by address similarity, and scores at most 12 new candidates. A compact target lookup skips targets already assigned in the ownership reference. Final export additionally protects every target claimed by the original France-safe output, resolves conflicts among new links, and limits newly added US/India links to the observed per-source maxima of 5 S2 and 6 S3 records.

On the exploratory 500-query-per-fold pilot, the at-most-one-match policy improved tune macro F0.5 by 0.930 points and evaluation by 0.572 points. These are local results on repeatedly inspected samples, not a leaderboard estimate. The first 1,000 test queries took about 12 seconds after setup. The full current pass covers 290,243 queries and resumes from committed checkpoints.

Recovery checkpoints and status are under `%LOCALAPPDATA%/BusinessEntityResolution/full/sparse_recovery`. The original files are preserved. Recovery export writes both a complete matching TSV and an expanded candidate TSV; a partial recovery pass still retains every source-1 row from the complete reference.

## Candidate file compatibility

The ownership and France-adaptation outputs only remove/reassign existing candidate links and can use the original full `candidate_pairs.tsv` under `%LOCALAPPDATA%/BusinessEntityResolution/full/output`. Recovery outputs must use their newly exported candidate file, which includes every additional scored pair.

Relevant tools are in `code/business_entity_resolution/tools`: `resolve_output_conflicts.py`, `adapt_france_idf.py`, `recover_sparse_matches.py`, and `export_sparse_recovery.py`. Experiment reports are under `%LOCALAPPDATA%/BusinessEntityResolution/full/shortcut_experiment`.
