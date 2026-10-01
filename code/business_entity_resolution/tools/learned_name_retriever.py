"""Learn token spelling alternatives exclusively from supervised fit identities."""
import collections
import json
import random
from pathlib import Path
from rapidfuzz.fuzz import ratio, token_sort_ratio
from er.index import Retriever
from er.io import rows, dump
from er.text import record, phonetic, SUFFIXES


def learn(work, dataset):
    work = Path(work)
    groups = json.loads((work/'training_groups.json').read_text())
    population = [g for g in groups if g['fold']=='fit']
    fit = {g['eid']:g for g in random.Random(42).sample(population,min(4000,len(population)))}
    excluded = {eid for g in groups if g['fold']!='fit' for eid in g['truth']}
    retriever = Retriever(work/'train.sqlite')
    retriever.db.execute('PRAGMA query_only=ON')
    counts = collections.defaultdict(collections.Counter)
    pairs = 0
    try:
        for raw in rows(Path(dataset)/'train'/'train_source1.tsv'):
            if raw['entity_id'] not in fit:
                continue
            query = record(raw)
            original = set(query['name'].split())-SUFFIXES-{'and','the'}
            for eid in fit[query['eid']]['truth']:
                if eid in excluded:
                    continue
                target = retriever.db.execute('SELECT name FROM records WHERE eid=?',(eid,)).fetchone()
                if target is None:
                    raise ValueError('Missing supervised target.')
                alternate = set(target['name'].split())-SUFFIXES-{'and','the'}
                left,right = original-alternate,alternate-original
                if len(left)>5 or len(right)>5:
                    continue
                for token in left:
                    if len(token)<3:
                        continue
                    ranked = sorted(((max(ratio(token,other),95 if phonetic(token) and phonetic(token)==phonetic(other) else 0),other)
                                     for other in right if len(other)>=3),reverse=True)
                    if ranked and ranked[0][0]>=60:
                        counts[token][ranked[0][1]]+=1
                pairs+=1
    finally:
        retriever.close()
    mapping = {token:[other for other,count in counter.most_common(3)
                      if count>=3 and count/sum(counter.values())>=.2]
               for token,counter in counts.items()}
    mapping = {token:values for token,values in mapping.items() if values}
    path = work.parent/'learned_name_variants.json'
    dump(path,{'mapping':mapping,'fit_pairs':pairs,'fit_entities':len(fit),
               'note':'Fit labels only. Held-out true target IDs excluded. No evaluation labels used.'})
    print(json.dumps({'learned_spelling_tokens':len(mapping),'fit_pairs':pairs,'path':str(path)}),flush=True)
    return mapping


class LearnedNameRetriever(Retriever):
    def __init__(self,path,mapping,**kwargs):
        super().__init__(path,**kwargs)
        self.mapping=mapping

    def candidates(self,query):
        original=super().candidates(query)
        result={target['eid']:(target,meta) for target,meta in original}
        words=[w for w in query['name'].split() if w not in SUFFIXES and w not in ('and','the')]
        expansions={v for token in words for v in self.mapping.get(token,[])}
        expansions.update(''.join(words[i:i+2]) for i in range(len(words)-1))
        if 1<len(words)<=5:
            expansions.add(''.join(words))
        ranked=sorted((self.df('name',t),t) for t in expansions)
        terms=[t for df,t in ranked if 0<df<=15000][:6]
        if not terms:
            return original
        hits=self.ranked_hits(self.clause('name',terms),query['country'])
        ids=[h[0] for h in hits]
        if not ids:
            return original
        frequency=self.db.execute('SELECT count(*) FROM records WHERE country=? AND core=?',
                                 (query['country'],query['core'])).fetchone()[0] if query['core'] else 0
        qp=phonetic(query['name'])
        for row in self.db.execute('SELECT * FROM records WHERE rid IN ('+','.join('?' for _ in ids)+')',ids):
            target=dict(row)
            if target['eid'] in result:
                continue
            n=token_sort_ratio(query['name'],target['name'])/100
            a=token_sort_ratio(query['address'],target['address'])/100 if query['address'] and target['address'] else 0
            tp=phonetic(target['name'])
            p=ratio(qp,tp)/100 if qp and tp else 0
            result[target['eid']]=target,{'retrieval_score':max(n,.92*p)*.65+a*.35,
                'name_frequency':frequency,'exact_block':int(bool(frequency) and frequency<=240 and target['core']==query['core'])}
        return list(result.values())


if __name__=='__main__':
    import sys
    learn(*sys.argv[1:])
