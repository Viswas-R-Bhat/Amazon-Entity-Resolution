"""Reproduce pair scores for a small set of disputed output links."""
import math
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
from rapidfuzz.fuzz import ratio, token_sort_ratio

from er.features import features
from er.index import connect
from er.text import country_key, phonetic


class PairScorer:
    def __init__(self, work, split="test", impute=True, baseline_france=True, frequency_split="train"):
        work = Path(work)
        self.db = connect(work / f"{split}.sqlite")
        self.db.execute("PRAGMA query_only=ON")
        self.rarity_model = lgb.Booster(model_file=str(work.parent / "rarity_feature_experiment" / "model.txt"))
        self.baseline_model = lgb.Booster(model_file=str(work / "model.txt"))
        self.impute = impute
        self.baseline_france = baseline_france
        self.france = country_key("France")
        self.freq = {"name": {}, "address": {}}
        train = connect(work / f"{frequency_split}.sqlite")
        for term, col, df in train.execute("SELECT term,col,doc FROM vocab WHERE col IN ('name','address')"):
            self.freq[col][term] = df
        train.close()
        self.scale = max(df for values in self.freq.values() for df in values.values())

    @lru_cache(maxsize=200000)
    def weighted(self, column, value):
        return {t: math.log1p((self.scale + 1) / (self.freq[column].get(t, 0) + 1))
                for t in set(value.split()) if len(t) >= 2}

    def rarity(self, a, b, column):
        left, right = self.weighted(column, a), self.weighted(column, b)
        overlap = left.keys() & right.keys()
        shared = sum(left[t] for t in overlap)
        ls, rs = sum(left.values()), sum(right.values())
        union = ls + rs - shared
        return [shared / union if union else 0., shared / ls if ls else 0., shared / rs if rs else 0.,
                max((left[t] for t in overlap), default=0.) / math.log1p(self.scale + 1),
                sum(t.isdigit() for t in overlap) / max(1, sum(t.isdigit() for t in left.keys() | right.keys()))]

    @lru_cache(maxsize=200000)
    def core_count(self, country, core):
        return self.db.execute("SELECT count(*) FROM records WHERE country=? AND core=?", (country, core)).fetchone()[0] if core else 0

    @lru_cache(maxsize=30000)
    def donor(self, country, core, name):
        addresses = {}
        for row in self.db.execute("SELECT address,uaddress FROM records WHERE country=? AND core=? AND name=? AND address<>''",
                                   (country, core, name)):
            addresses[row["address"]] = row["uaddress"]
            if len(addresses) > 1:
                return None
        return next(iter(addresses.items())) if addresses else None

    def score(self, pairs):
        vectors = []
        france = []
        for q, original in pairs:
            t = original
            is_france = self.baseline_france and q["country"] == self.france
            if self.impute and not is_france and not t["address"]:
                donor = self.donor(t["country"], t["core"], t["name"])
                if donor:
                    t = dict(t, address=donor[0], uaddress=donor[1])
            name_sim = token_sort_ratio(q["name"], t["name"]) / 100
            address_sim = token_sort_ratio(q["address"], t["address"]) / 100 if q["address"] and t["address"] else 0.
            qp, tp = phonetic(q["name"]), phonetic(t["name"])
            sound = ratio(qp, tp) / 100 if qp and tp else 0.
            count = self.core_count(q["country"], q["core"])
            meta = {"name_frequency": count, "retrieval_score": max(name_sim, .92 * sound) * .65 + address_sim * .35,
                    "exact_block": int(bool(count) and count <= 240 and q["core"] == t["core"])}
            vectors.append(features(q, t, meta) + self.rarity(q["name"], t["name"], "name") + self.rarity(q["address"], t["address"], "address"))
            france.append(is_france)
        if not vectors:
            return np.empty(0), np.empty((0, 36))
        x = np.asarray(vectors, dtype=np.float32)
        mask = np.asarray(france, dtype=bool)
        scores = np.empty(len(pairs))
        if mask.any():
            scores[mask] = self.baseline_model.predict(x[mask, :26], raw_score=True, num_threads=3)
        if (~mask).any():
            scores[~mask] = self.rarity_model.predict(x[~mask], raw_score=True, num_threads=3)
        return scores, x
