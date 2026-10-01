"""Experimental candidate expansion through predicted, never ground-truth, aliases."""
import numpy as np
from rapidfuzz.fuzz import ratio, token_sort_ratio
from er.index import Retriever
from er.features import features
from er.text import phonetic


class AliasRetriever(Retriever):
    def __init__(self, path, model, **kwargs):
        super().__init__(path, **kwargs)
        self.model = model

    def candidates(self, query):
        original = super().candidates(query)
        if not original:
            return original
        scores = self.model.predict(np.asarray([features(query,target,meta) for target,meta in original],
                                               dtype=np.float32),num_threads=1)
        seeds = []
        forms = {(query['name'],query['address'])}
        for i in np.argsort(-scores,kind='stable'):
            if scores[i] < .98 or len(seeds) == 2:
                break
            target = original[i][0]
            form = target['name'],target['address']
            if form not in forms:
                forms.add(form)
                seeds.append(target)
        result = {target['eid']:(target,meta) for target,meta in original}
        frequency = original[0][1]['name_frequency']
        qp = phonetic(query['name'])
        for seed in seeds:
            for target,_ in super().candidates(seed):
                if target['eid'] in result:
                    continue
                n = token_sort_ratio(query['name'],target['name'])/100
                a = token_sort_ratio(query['address'],target['address'])/100 if query['address'] and target['address'] else 0
                tp = phonetic(target['name'])
                p = ratio(qp,tp)/100 if qp and tp else 0
                result[target['eid']] = target,{'retrieval_score':max(n,.92*p)*.65+a*.35,
                    'exact_block':int(bool(frequency) and frequency<=240 and target['core']==query['core']),
                    'name_frequency':frequency}
        return list(result.values())
