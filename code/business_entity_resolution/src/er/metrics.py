"""Exact entity-macro F0.5, including empty predictions and empty truth."""
import numpy as np


def entity_score(truth, prediction):
    truth, prediction = set(truth), set(prediction)
    if not truth and not prediction:
        return 1.0
    tp = len(truth & prediction)
    return 1.25 * tp / (1.25 * tp + len(prediction - truth) + .25 * len(truth - prediction))


def evaluate(groups, probabilities, threshold):
    results = []
    examples = []
    for group in groups:
        p = probabilities[group['start']:group['end']]
        predicted = {eid for eid, score in zip(group['candidates'], p) if score >= threshold}
        truth = set(group['truth'])
        recalled = truth & set(group['candidates'])
        results.append((group['country'], entity_score(truth, predicted), len(truth), len(recalled),
                        entity_score(truth, recalled), len(predicted - truth), len(truth - predicted)))
        if len(examples) < 25 and predicted != truth:
            examples.append({'source1_entity_id': group['eid'], 'country': group['country'],
                             'false_positives': sorted(predicted - truth), 'false_negatives': sorted(truth - predicted),
                             'missed_by_retrieval': sorted(truth - set(group['candidates']))})

    def summarize(values):
        if not values:
            return {'entities': 0, 'macro_f05': None}
        positives = [v for v in values if v[2]]
        fn = sum(v[6] for v in values)
        fp = sum(v[5] for v in values)
        tp = sum(v[2] for v in values) - fn
        return {'entities': len(values), 'macro_f05': float(np.mean([v[1] for v in values])),
                'precision_micro': tp / (tp + fp) if tp + fp else None,
                'recall_micro': tp / (tp + fn) if tp + fn else None,
                'f1_micro': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                'exact_match_rate': sum(v[5] == 0 and v[6] == 0 for v in values) / len(values),
                'true_positive_links': tp,
                'candidate_recall_micro': sum(v[3] for v in values) / max(1, sum(v[2] for v in values)),
                'candidate_recall_macro_nonempty': float(np.mean([v[3] / v[2] for v in positives])) if positives else None,
                'candidate_oracle_macro_f05': float(np.mean([v[4] for v in values])),
                'false_positive_links': sum(v[5] for v in values), 'false_negative_links': sum(v[6] for v in values)}
    result = summarize(results)
    result['by_country'] = {c: summarize([v for v in results if v[0] == c]) for c in sorted({v[0] for v in results})}
    result['error_examples'] = examples
    return result


def select_threshold(groups, probabilities):
    if not groups:
        raise ValueError('No threshold-tuning entities; increase the sample size.')
    # Aggregation by entity avoids Python pair loops for each trial threshold.
    ids, scores, labels = [], [], []
    truth_count = np.array([len(g['truth']) for g in groups], dtype=np.float64)
    for i, group in enumerate(groups):
        truth = set(group['truth'])
        ids.extend([i] * len(group['candidates']))
        scores.extend(probabilities[group['start']:group['end']])
        labels.extend(eid in truth for eid in group['candidates'])
    ids, scores, labels = np.asarray(ids, dtype=np.int32), np.asarray(scores), np.asarray(labels, dtype=bool)
    trials = sorted(set(np.linspace(.01, .99, 99).tolist() + [.995, .999, 1.000001]))
    best = (-1.0, 1.000001)
    for threshold in trials:
        mask = scores >= threshold
        predicted = np.bincount(ids[mask], minlength=len(groups))
        tp = np.bincount(ids[mask & labels], minlength=len(groups))
        denominator = predicted + .25 * truth_count
        values = np.ones(len(groups))
        np.divide(1.25 * tp, denominator, out=values, where=denominator != 0)
        score = float(values.mean())
        if score > best[0] + 1e-12 or (abs(score - best[0]) <= 1e-12 and threshold > best[1]):
            best = (score, threshold)
    return {'macro_f05': best[0], 'threshold': best[1], 'trials': len(trials)}
