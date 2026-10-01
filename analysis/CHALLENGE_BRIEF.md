# Amazon ML Challenge 2026: data audit and possible approaches

Prepared September 25, 2026. This is an investigation and planning artifact. No matching model has been trained, predictions submitted, or cloud resources provisioned. The user wants to choose a path before implementation and does not want work divided among teammates.

**Recommendation.** Start with a memory-conscious retrieval pipeline and a supervised classifier using name/address comparisons. Design retrieval to combine several ways of finding a match. Add a small multilingual model only if measured retrieval or classification errors justify its runtime. The main bottleneck is finding the right few candidates among millions of records.

**Evidence and scope.** All seven TSVs were parsed locally. Full-data aggregates and integrity checks are in `reports/dataset_profile.json` and `reports/ambiguity_profile.json`. Similarity statistics explicitly labeled as sampled use 75,781 known-positive pairs belonging to 20,676 reference entities. No business records were searched on the web or augmented externally. Internet research covered competition logistics, AWS access and software licenses only. Original challenge files were left unchanged.

**The actual dataset**

| File | Records | Countries |
|---|---:|---|
| train_source1.tsv | 2,206,821 | India, US |
| train_source2.tsv | 5,034,616 | India, US |
| train_source3.tsv | 5,285,603 | India, US |
| train_ground_truth.tsv | 2,206,821 | One label row per training S1 |
| test_source1.tsv | 1,732,544 | France, India, US |
| test_source2.tsv | 4,887,273 | France, India, US |
| test_source3.tsv | 5,082,316 | France, India, US |

There are 24,229,173 business records, excluding the ground-truth rows, in approximately 2.52 GB of TSVs including labels. Train has 12,527,040 business records; test has 11,702,133. Source records contain only `entity_id`, `business_name`, `business_address`, and `country`. There are no phone numbers, websites, coordinates or separate address-component fields.

| Test country | S1 | S2 | S3 | Share of scored S1 entities |
|---|---:|---:|---:|---:|
| France | 259,452 | 703,378 | 731,615 | 14.98% |
| India | 809,986 | 2,312,565 | 2,405,000 | 46.75% |
| US | 663,106 | 1,871,330 | 1,945,701 | 38.27% |

Training S1 is approximately 40% India and 60% US. Both the unseen country and the change in country proportions matter. Test singleton rates and France matching difficulty are unknown because test labels are absent.

**Label structure and integrity**

- 7,638,365 labeled positive links; 3.461 matches per training S1 on average; observed range 0–11.
- 123,247 training S1 entities have no matches, or 5.585%. Predicting empty lists everywhere scores only 0.055848 on the full training set.
- 1,776,047 S1 entities have matches in both target sources. 1,129,968 have multiple S2 matches and 1,224,128 have multiple S3 matches. Choosing only one target per reference loses real matches.
- 2,681,854 training S2/S3 records (25.986%) are absent from all positive lists. Under the supplied ground truth they are distractors relative to S1, not records that should be forced into a match.
- Every training S1 has exactly one ground-truth row. All labeled target IDs exist. There are no duplicate targets inside label lists, and no target is assigned to two different reference IDs.
- IDs are unique within each source file, source prefixes are correct, and country fields are populated. No same-source IDs overlap between train and test.
- No exact duplicate `(country, business_name, business_address)` tuples were found within either S1 split, and no exact S1 content tuples overlap between train and test. This does not establish absence of near-duplicates or shared naming patterns.
- All 7,638,365 labeled positive links stay within the same country. Country partitioning is supported by training evidence, while country labels must remain open to France and any future values.

**What makes this difficult**

1. **Name equality is ambiguous.** After case, accent and punctuation normalization, 43.76% of Indian, 29.13% of US and 34.41% of French test S1 records share their name key with another S1 record in the same country. For example, 205 French references normalize to `bordeaux club sarl`. The deduplicated reference source still contains distinct businesses with the same name. Address evidence and the gap between competing candidates matter.

2. **Simple equality misses most true links.** In the sampled positives, 25.60% have equal normalized names, 8.28% have equal normalized addresses and 32.26% have either. For India, the last figure is 24.90%. These are pair-recall diagnostics for this particular simple normalization, not end-to-end validation scores or limits on more advanced methods.

3. **Cross-script names are substantial.** The training Indian S1 names are all ASCII, but S2/S3 include Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada and Malayalam. Examples include `One Construction` paired with `वन कंस्ट्रक्शन`. In the positive sample, 24.05% of Indian pairs share no normalized name token. Address-based retrieval and locally learned cross-script aliases are useful avenues. Preserve original Unicode text. Removing all combining marks is an exploratory simplification here and would damage meaningful marks in Indic text if used as the sole production normalization.

4. **Addresses can be absent or heavily edited.** Names are nonempty throughout all six source files; S1 addresses are also populated. There are 344,883 missing target addresses in training and 265,506 in test. Full-label joins show 337,018 true positive links with a missing target address, about 4.41% of positives. Missingness is not the same as disagreement. S3 address character similarity is lower than S2 in the sample, suggesting that source-specific error analysis may be useful.

5. **Postal-code blocking cannot be the foundation.** Only 0.27% of Indian training S1 addresses and 0.40% of French test S1 addresses contain even a standalone five- or six-digit token. Such a token is not necessarily a postal code. US five-digit strings also include street numbers. Use multiple address cues and avoid requiring a ZIP/PIN code.

6. **The search space is enormous.** Test S1 × (S2 + S3) is 17,272,751,604,416 pairs. Restricting to same-country pairs still leaves 6,724,569,566,212. All-pairs fuzzy matching or a dense similarity matrix is infeasible. With an illustrative average of 50 candidates per S1, there would still be 86.6 million pair scores; this is a sizing example, not a chosen cap.

7. **France requires generalization.** French records comprise nearly 15% of the macro-average. Generic character/token features, robust number comparisons and Unicode-safe normalization should transfer more naturally than memorizing US/India categories. No validation experiment can directly measure French performance with these labels alone.

**Metric and validation design**

For each reference entity, compare sets of predicted and true targets. With at least one true or predicted target:

`F0.5 = 1.25 * TP / (1.25 * TP + FP + 0.25 * FN)`

If both lists are empty, score 1. If the true list is empty but the prediction is nonempty, score 0. Average these entity scores over every S1, including references with no candidates. This is not the global pairwise F-score or binary classification accuracy. The formula gives an FP coefficient four times the FN coefficient; the statement's informal “2×” wording should not substitute for the actual formula.

Split by reference entity, keeping its linked variants out of supervised training when that entity is held out. A random split of positive/negative pairs leaks identities. Learn aliases, transformations and any label-informed features only from the training portion. Keep realistic distractors and competing names in evaluation; otherwise retrieval and precision appear artificially easy.

Use a fixed main validation split stratified by country and match-count/singleton status. Add US-to-India and India-to-US country holdouts as transfer stress tests, clearly acknowledging that neither estimates French performance directly. Tune thresholds on per-S1 macro F0.5, including abstention. Measure blocking pair recall, entity-level recall/complete coverage, candidate counts, peak memory, runtime, precision and singleton errors separately. Measure the blocking ceiling by retaining only known-positive retrieved candidates. Never add missed ground-truth pairs into the validation candidate list and then call that end-to-end performance.

**Possible paths**

| Path | Concrete approach | Benefit | Main risk | Priority |
|---|---|---|---|---|
| A: retrieval plus tabular classifier | Country partitions; union of rare name-token, character-pattern and address-token/number retrieval; name/address similarity and missingness features; boosted-tree classifier; calibrated thresholds | Can learn from millions of labels, uses CPU well, easy to inspect and reproduce | Cross-script retrieval and common-name ambiguity | First choice |
| B: selective multilingual extension | Keep A and add locally run, license-checked multilingual retrieval or pair scoring for measured failure categories | May recover script and spelling variations missed by lexical methods | Encoding/inference throughput, checkpoint license and model size | Add only after evidence |
| C: consistency refinement | Resolve competition for the same target using confidence margins; use high-confidence matched variants to find further candidates, then rescore | May recover variants and prevent conflicting assignments | False links can spread; naive graph transitivity can damage precision | Later, validate as an ablation |

For A, a tree model could be LightGBM or CatBoost. Their project licenses are respectively [MIT](https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE) and [Apache 2.0](https://github.com/catboost/catboost/blob/master/LICENSE). This verifies library licenses, not blanket approval for arbitrary added pretrained checkpoints. The submitted model artifacts and their dependencies must satisfy the challenge rules.

An alternative retrieval direction deserves benchmarking: index the smaller, cleaner S1 catalog and query it with noisy S2/S3 records, then regroup scored pairs by S1 for export. This reduces the test reference index to 1.73 million records rather than 9.97 million targets. It increases the number of queries, so it is an option to measure, not a proven speedup. The labels support at most one S1 owner per target, but the final decision rule still needs validation. Never impose one-target-per-S1.

For training negatives, emphasize plausible incorrect candidates returned by the actual retriever: repeated names, nearby-looking addresses and wrong street numbers. Uniform random pairs alone would make an unrealistically easy classification problem. Retain raw and normalized fields, numeric conflicts, rarity, missingness and runner-up confidence as distinct evidence. Numeric differences cannot be an unconditional veto because valid pairs can have changed or missing components.

**Local and AWS resources**

The inspected machine has an Intel i7-13620H (10 physical cores, 16 logical processors), approximately 15.64 GiB RAM and an NVIDIA RTX 3050 Laptop GPU reported as 6 GB. Around 4.2 GiB RAM was available at the initial check. Existing Python libraries include pandas, NumPy, PyArrow, scikit-learn and RapidFuzz. DuckDB 1.5.5 was installed only into workspace `.analysis-deps`; the core audit used a 1.6 GB database memory limit and two threads. Its first full pass took approximately 75 seconds. This is an audit runtime, not a forecast of training or prediction time. The reusable database is about 1.52 GiB.

Local CPU development is a reasonable starting point. Run partitions and feature extraction in batches, cache reusable representations and train initially on representative entity groups with hard negatives. Full inference throughput must be benchmarked before committing to a pipeline. Six GB VRAM can support suitably small batched models, but is a constraint for large models and large batches. A ceiling of eight billion parameters is permission, not a target model size.

The [official Unstop event page](https://unstop.com/hackathons/crp-amazon-ml-challenge-2026-amazon-1743604) confirms the deadline as **September 27, 2026, 11:59 PM IST**. It requires a valid Builder Center profile alias and advertises $200 in credits for participants, plus $100 for the top 500 teams at the 48-hour mark. These are advertised benefits; no account balance or credit allocation was verified.

The linked [AWS employee preparation guide](https://builder.aws.com/content/3HiM6zDmFrF98fRzOUETnGFDoqz/amazon-ml-challenge-2026-your-complete-prep-guide-with-live-demo) describes Builder Center setup and SageMaker notebook use. The [current AWS Free Tier page](https://aws.amazon.com/free/) describes $100 at new-account creation and up to another $100 earned through activities, with different Free/Paid-plan access and promotional-credit rules. Do not assume event credits and Free Tier credits are additive, that existing accounts qualify, or that the required instance quota is available. The guide's churn example, CSV format and accuracy numbers do not describe this entity-resolution dataset.

An [AWS Builder ID](https://docs.aws.amazon.com/signin/latest/userguide/differences-builder-id.html) is an identity; creating a profile alone does not verify usable compute, credit balance or GPU quota. AWS documents a limited rollout of a newer Builder-ID account experience, so the account's actual console state matters. For this challenge, cloud compute would be useful for more RAM/CPU or optional faster neural inference. A production prediction endpoint is unnecessary because the deliverable is batch TSV files. No AWS account was created, inspected or charged during this work.

Do not make new SageMaker Studio Lab access a dependency: [AWS currently says Studio Lab is closed to new customers](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab.html). Existing customers can continue, subject to service limits.

**Submission requirements and discrepancies**

- Live leaderboard: `matching_results.tsv` with exactly `source1_entity_id` and `matched_entity_ids`.
- Final archive: both `output/matching_results.tsv` and `output/candidate_pairs.tsv`, reproducible code under `code/business_entity_resolution/src/`, a README, pinned environment and completed `Documentation_template.md`.
- Both output TSVs require all 1,732,544 test S1 IDs once. Lists must be unique, comma-separated S2/S3 IDs from the supplied test files. Preserve empty lists as empty fields.
- `candidate_pairs.tsv` records the exact candidate pairs fed to the final matcher. Every final match must appear in it. A raw early blocking set or a list rebuilt from final predictions is not an accurate candidate audit.
- The provided validator defaults to skipping target-ID existence checks. Use `--check-ids` when validating the final output, subject to memory requirements. Missing candidate files and matches outside candidates produce warnings rather than failures. Add strict pipeline assertions for these requirements.
- README says nonexistent target IDs cause rejection, while validator comments say they only reduce score. Satisfy the stricter requirement: emit only real IDs.
- The event overview requests a 1–2-page methodology document, while the supplied task-specific README explicitly says there is no page limit. A short executive summary plus the detailed filled template can preserve both intents; exact upload constraints remain to be checked in the authenticated portal.
- External business lookup, geocoding and externally gathered business data are prohibited. Use provided records and labels. Model limit: at most eight billion parameters and MIT/Apache 2.0 licensing.
- Private leaderboard determines final rankings. Neither its country composition nor its split ratio is disclosed in the supplied files.

**What remains unknown**

The exact round URL redirects to sign-in in the available browser session, so daily submission limits, current round announcements, upload size limits and submission selection rules were not verified. AWS credit redemption, available balance, account plan and compute quotas also remain unknown. No actual retrieval recall, model validation score or full inference runtime has been measured. Those are the next decision-making experiments once implementation is authorized.

**Efficient continuation**

Start future work with `CONTEXT.md`, this brief and the compact JSON profiles rather than loading raw TSVs into the conversation again. The audit scripts and local database support targeted follow-up queries. Input SHA-256 hashes are recorded in `reports/input_manifest.json`; runtime details are in `reports/audit_environment.json`. Keep evaluation splits, candidate sets and experiment results cached and versioned. Keep web research about tools/logistics separate from the business data. No extra plugin is required for the current work.
