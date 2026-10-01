"""Read-only full reference overlap check; never uses hidden test labels."""
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path


def audit(dataset):
    begin = time.monotonic()
    train = set()
    result = {'key': 'Country/name/address, casefolded with whitespace collapsed; 160-bit digest. '
                     'Exact text check only, not fuzzy entity deduplication.'}
    for split in ('train', 'test'):
        countries = Counter()
        count = overlap = 0
        with (dataset / split / f'{split}_source1.tsv').open(encoding='utf-8-sig', newline='') as handle:
            for r in csv.DictReader(handle, delimiter='\t'):
                countries[r['country']] += 1
                count += 1
                key = json.dumps([' '.join(r[c].casefold().split()) for c in
                                  ('country', 'business_name', 'business_address')], ensure_ascii=False)
                signature = hashlib.blake2b(key.encode(), digest_size=20).digest()
                if split == 'train':
                    train.add(signature)
                else:
                    overlap += signature in train
        result[split] = {'reference_rows': count, 'country_counts': dict(countries)}
        if split == 'train':
            result['train']['duplicate_text_rows'] = count - len(train)
        else:
            result['test']['rows_with_exact_text_in_train_reference'] = overlap
        print(json.dumps({split: result[split]}), flush=True)
    result['elapsed_seconds'] = round(time.monotonic()-begin, 2)
    return result


if __name__ == '__main__':
    result = audit(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2), encoding='utf-8')
