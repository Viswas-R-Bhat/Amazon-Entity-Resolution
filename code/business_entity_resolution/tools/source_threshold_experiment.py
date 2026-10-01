"""Check whether S2 and S3 need different precision thresholds."""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np


def score_groups(groups, probabilities, threshold2, threshold3):
    values = []
    for g in groups:
        truth = set(g['truth'])
        pred = [eid for i, eid in enumerate(g['candidates'])
                if probabilities[g['start'] + i] >= (threshold2 if eid.startswith('S2-') else threshold3)]
        if not truth:
            values.append(1. if not pred else 0.)
        elif not pred:
            values.append(0.)
        else:
            tp = len(truth.intersection(pred))
            if tp == 0:
                values.append(0.)
            else:
                precision = tp / len(pred)
                recall = tp / len(truth)
                values.append(1.25 * precision * recall / (.25 * precision + recall))
    return float(np.mean(values))


def tables(groups, probabilities, thresholds):
    result = []
    for g in groups:
        ids = g['candidates']
        start = g['start']
        truth = set(g['truth'])
        stats = {}
        for prefix in ('S2-', 'S3-'):
            scores = np.asarray([probabilities[start + i] for i, eid in enumerate(ids) if eid.startswith(prefix)])
            correct = np.asarray([eid in truth for eid in ids if eid.startswith(prefix)], dtype=bool)
            counts, tps = [], []
            for t in thresholds:
                selected = scores >= t
                counts.append(int(selected.sum()))
                tps.append(int(correct[selected].sum()))
            stats[prefix] = (np.asarray(counts), np.asarray(tps))
        result.append((len(truth), stats['S2-'][0], stats['S2-'][1], stats['S3-'][0], stats['S3-'][1]))
    return result


def macro_from_tables(data, i2, i3):
    vals = []
    for ntrue, p2, tp2, p3, tp3 in data:
        npred = int(p2[i2] + p3[i3])
        tp = int(tp2[i2] + tp3[i3])
        if ntrue == 0:
            vals.append(1. if npred == 0 else 0.)
        elif npred == 0:
            vals.append(0.)
        elif tp == 0:
            vals.append(0.)
        else:
            precision, recall = tp / npred, tp / ntrue
            vals.append(1.25 * precision * recall / (.25 * precision + recall))
    return float(np.mean(vals))


def run(work, rarity_dir):
    start_time = time.monotonic()
    work, rarity_dir = Path(work), Path(rarity_dir)
    groups = json.loads((work / 'training_groups.json').read_text())
    n = groups[-1]['end']
    base = np.memmap(work / 'features.float32', dtype='float32', mode='r').reshape(-1, 26)[:n]
    extra = np.memmap(rarity_dir / 'extra.float32', dtype='float32', mode='r', shape=(n, 10))
    model = lgb.Booster(model_file=str(rarity_dir / 'model.txt'))
    scores = np.empty(n, dtype=np.float32)
    for lo in range(0, n, 50000):
        hi = min(n, lo + 50000)
        scores[lo:hi] = model.predict(np.concatenate((base[lo:hi], extra[lo:hi]), axis=1), num_threads=2)
    tune = [g for g in groups if g['fold'] == 'tune']
    evaluation = [g for g in groups if g['fold'] == 'eval']
    thresholds = np.arange(.50, .991, .01)
    tune_tables = tables(tune, scores, thresholds)
    eval_tables = tables(evaluation, scores, thresholds)
    grid = np.empty((len(thresholds), len(thresholds)), dtype=np.float64)
    for i in range(len(thresholds)):
        for j in range(len(thresholds)):
            grid[i, j] = macro_from_tables(tune_tables, i, j)
    i, j = np.unravel_index(np.argmax(grid), grid.shape)
    single_i = int(np.argmin(np.abs(thresholds - .79)))
    result = {'source_thresholds': {'S2': float(thresholds[i]), 'S3': float(thresholds[j])},
              'tune_f05': float(grid[i, j]),
              'tune_global_079': macro_from_tables(tune_tables, single_i, single_i),
              'eval_f05': macro_from_tables(eval_tables, i, j),
              'eval_global_079': macro_from_tables(eval_tables, single_i, single_i),
              'tune_entities': len(tune), 'eval_entities': len(evaluation),
              'seconds': round(time.monotonic() - start_time, 1),
              'note': 'Thresholds selected only on tune; evaluation is previously inspected and not a leaderboard estimate.'}
    out = rarity_dir / 'source_threshold_report.json'
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    import sys
    run(*sys.argv[1:])
