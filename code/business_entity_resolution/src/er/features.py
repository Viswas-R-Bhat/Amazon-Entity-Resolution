import math
import re
from rapidfuzz.fuzz import ratio, token_sort_ratio, token_set_ratio, partial_ratio
from .text import phonetic

FEATURES = ['name_ratio', 'name_sort', 'name_set', 'name_partial', 'name_jaccard',
            'core_equal', 'name_equal', 'name_length_ratio', 'phonetic_ratio',
            'unicode_name_ratio', 'address_ratio', 'address_sort', 'address_set',
            'address_jaccard', 'address_equal', 'address_missing', 'address_length_ratio',
            'number_overlap', 'number_conflict', 'number_missing', 'name_number_conflict',
            'unicode_address_ratio', 'source3', 'log_name_frequency', 'retrieval_score', 'exact_block']


def sim(function, a, b):
    return function(a, b) / 100 if a and b else 0.0


def jaccard(a, b):
    a, b = set(a.split()), set(b.split())
    return len(a & b) / len(a | b) if a and b else 0.0


def length_ratio(a, b):
    return min(len(a), len(b)) / max(len(a), len(b)) if a and b else 0.0


def numeric(a):
    return set(re.findall(r'\d+', a))


def features(a, b, retrieval):
    n, m, x, y = a['name'], b['name'], a['address'], b['address']
    ax, bx, an, bn = numeric(x), numeric(y), numeric(n), numeric(m)
    return [sim(ratio, n, m), sim(token_sort_ratio, n, m), sim(token_set_ratio, n, m),
            sim(partial_ratio, n, m), jaccard(n, m), float(bool(a['core']) and a['core'] == b['core']),
            float(bool(n) and n == m), length_ratio(n, m), sim(ratio, phonetic(n), phonetic(m)),
            sim(ratio, a['uname'], b['uname']), sim(ratio, x, y), sim(token_sort_ratio, x, y),
            sim(token_set_ratio, x, y), jaccard(x, y), float(bool(x) and x == y),
            float(not x or not y), length_ratio(x, y),
            len(ax & bx) / len(ax | bx) if ax and bx else 0.0,
            float(bool(ax and bx) and not ax & bx), float(not ax or not bx),
            float(bool(an and bn) and not an & bn), sim(ratio, a['uaddress'], b['uaddress']),
            float(b['eid'].startswith('S3-')), math.log1p(retrieval['name_frequency']),
            retrieval['retrieval_score'], retrieval['exact_block']]
