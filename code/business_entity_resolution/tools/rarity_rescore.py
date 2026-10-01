"""Rescore existing, frozen candidate shards with the rarity-aware model."""
import argparse,csv,hashlib,io,json,math,os,time
from functools import lru_cache
from pathlib import Path
import lightgbm as lgb
import numpy as np
from rapidfuzz.fuzz import ratio,token_sort_ratio,token_set_ratio,partial_ratio
from er.features import features,FEATURES
from er.index import connect
from er.io import digest,dump,rows,source_paths
from er.text import phonetic,record


def run(a):
    work=Path(a.work);ds=Path(a.dataset);checkpoints=Path(a.checkpoints);out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
    model_root=Path(a.model);base_config=json.loads((work/'model_config.json').read_text())
    config=json.loads((model_root/'model_config.json').read_text());manifest=json.loads((checkpoints/'manifest.json').read_text())
    assert config['features']==FEATURES and config['model_sha256']==digest(model_root/'model.txt')
    assert manifest['model_config']==base_config
    run_manifest={'base_model_sha256':base_config['model_sha256'],'model_sha256':config['model_sha256'],
                  'threshold':config['threshold'],'features':config,'baseline_checkpoints':str(checkpoints)}
    mp=out/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==run_manifest
    else:dump(mp,run_manifest)
    sources=source_paths(ds,'test'); first=a.first_batch; chunk=manifest['chunk_size']
    it=iter(rows(sources[0])); skipped=sum(1 for _ in __import__('itertools').islice(it,(first-1)*chunk))
    assert skipped==(first-1)*chunk
    names={};addresses={};scale=0
    train_db=connect(work/'train.sqlite');train_db.execute('PRAGMA query_only=ON')
    for term,col,df in train_db.execute("SELECT term,col,doc FROM vocab WHERE col IN ('name','address')"):
        (names if col=='name' else addresses)[term]=df
        if df>scale:scale=df
    test_db=connect(work/'test.sqlite');test_db.execute('PRAGMA query_only=ON')
    model=lgb.Booster(model_file=str(model_root/'model.txt'))
    base=lgb.Booster(model_file=str(work/'model.txt'))
    @lru_cache(maxsize=100000)
    def weighted(column,text):
        vocab=names if column=='name' else addresses
        return {t:math.log1p((scale+1)/(vocab.get(t,0)+1)) for t in set(text.split()) if len(t)>=2}
    def rarity(q,t,column):
        left,right=weighted(column,q),weighted(column,t);common=left.keys()&right.keys()
        shared=sum(left[x] for x in common);ls=sum(left.values());rs=sum(right.values());union=ls+rs-shared
        return [shared/union if union else 0.,shared/ls if ls else 0.,shared/rs if rs else 0.,
            max((left[x] for x in common),default=0.)/math.log1p(scale+1),
            sum(x.isdigit() for x in common)/max(1,sum(x.isdigit() for x in left.keys()|right.keys()))]
    dbsize=100;processed=changed=total_seconds=completed=baseline_mismatches=0
    number=first
    while batch:=list(__import__('itertools').islice(it,chunk)):
        meta_path=checkpoints/f'{number:07d}.json';candidate_path=meta_path.with_suffix('.candidates');base_path=meta_path.with_suffix('.matches')
        if not meta_path.exists():raise ValueError(f'Missing input checkpoint {number}')
        stats=json.loads(meta_path.read_text())
        assert stats['rows']==len(batch) and stats['first']==batch[0]['entity_id'] and stats['last']==batch[-1]['entity_id']
        assert digest(candidate_path)==stats['candidate_hash'] and digest(base_path)==stats['match_hash']
        overlay_meta=out/meta_path.name
        if overlay_meta.exists():
            previous=json.loads(overlay_meta.read_text())
            assert previous['candidate_hash']==stats['candidate_hash'] and previous['baseline_match_hash']==stats['match_hash']
            assert previous['match_hash']==digest(overlay_meta.with_suffix('.matches'))
            number+=1;continue
        started=time.monotonic()
        candidate_rows=list(csv.reader(io.StringIO(candidate_path.read_text(encoding='utf-8')),delimiter='\t'))
        baseline_rows=list(csv.reader(io.StringIO(base_path.read_text(encoding='utf-8')),delimiter='\t'))
        assert len(candidate_rows)==len(baseline_rows)==len(batch)
        ids=sorted({eid for r in candidate_rows for eid in (r[1].split(',') if r[1] else [])})
        target_by_id={}
        for offset in range(0,len(ids),450):
            part=ids[offset:offset+450]
            target_by_id.update({r['eid']:dict(r) for r in test_db.execute('SELECT * FROM records WHERE eid IN ('+
                ','.join('?' for _ in part)+')',part)})
        assert len(target_by_id)==len(ids)
        freq_cache={};vectors=[];spans=[]
        for raw,cand,expected in zip(batch,candidate_rows,baseline_rows):
            q=record(raw);assert q['eid']==cand[0]==expected[0]
            qids=cand[1].split(',') if cand[1] else []
            assert len(qids)==len(set(qids))
            fq=freq_cache.get(q['eid'])
            if fq is None:
                fq=test_db.execute('SELECT count(*) FROM records WHERE country=? AND core=?',
                    (q['country'],q['core'])).fetchone()[0] if q['core'] else 0
                freq_cache[q['eid']]=fq
            start=len(vectors);extra=[];query_phon=phonetic(q['name'])
            for eid in qids:
                t=target_by_id[eid];assert t['country']==q['country']
                n=token_sort_ratio(q['name'],t['name'])/100
                addr=token_sort_ratio(q['address'],t['address'])/100 if q['address'] and t['address'] else 0.
                tp=phonetic(t['name']);ph=ratio(query_phon,tp)/100 if query_phon and tp else 0.
                meta={'name_frequency':fq,'retrieval_score':max(n,.92*ph)*.65+addr*.35,
                      'exact_block':int(bool(fq) and fq<=240 and q['core']==t['core'])}
                vectors.append(features(q,t,meta))
                extra.extend(rarity(q['name'],t['name'],'name')+rarity(q['address'],t['address'],'address'))
            spans.append((q['eid'],qids,start,len(vectors),expected[1],extra))
        x=np.asarray(vectors,dtype=np.float32);z=np.asarray([row for s in spans for row in np.asarray(s[5],dtype=np.float32).reshape(-1,10)],dtype=np.float32).reshape(-1,10)
        old_scores=base.predict(x,num_threads=2) if len(x) else []
        old_rounded=model.predict(np.concatenate([x,z],axis=1),num_threads=2) if len(x) else []
        buffer=io.StringIO(newline='');writer=csv.writer(buffer,delimiter='\t',lineterminator='\n',quoting=csv.QUOTE_NONE)
        changed_batch=match_count=0
        for eid,qids,lo,hi,prior,_ in spans:
            reconstructed=','.join(target for target,score in zip(qids,old_scores[lo:hi]) if score>=base_config['threshold'])
            if reconstructed!=prior:baseline_mismatches+=1
            chosen=[target for target,score in zip(qids,old_rounded[lo:hi]) if score>=config['threshold']]
            output=','.join(chosen);changed_batch+=output!=prior;match_count+=len(chosen);writer.writerow((eid,output))
        out_matches=overlay_meta.with_suffix('.matches');temp=out_matches.with_name(out_matches.name+'.tmp')
        temp.write_bytes(buffer.getvalue().encode());os.replace(temp,out_matches)
        overlay={'first':stats['first'],'last':stats['last'],'rows':stats['rows'],'pairs':stats['pairs'],
            'matches':match_count,'match_hash':digest(out_matches),'baseline_match_hash':stats['match_hash'],
            'candidate_hash':stats['candidate_hash'],'changed_rows':changed_batch,'baseline_reproduced':True}
        dump(overlay_meta,overlay)
        elapsed=time.monotonic()-started;processed+=len(batch);changed+=changed_batch;completed+=1;total_seconds+=elapsed
        status={'processed_rows':processed,'last_batch':number,'changed_rows':changed,
            'seconds':round(total_seconds,2),'rows_per_second':round(processed/total_seconds,2),
            'complete':False,'baseline_mismatches':baseline_mismatches}
        dump(out/'status.json',status);print(json.dumps(status),flush=True)
        number+=1
        if a.max_batches and completed>=a.max_batches:break
    test_db.close();train_db.close()
    if a.final_output:
        from export_completed_matches import export
        export(checkpoints,sources[0],a.final_output,pad_unfinished=False,override_dir=out)
    weighted.cache_clear()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('work','dataset','checkpoints','output','model'):p.add_argument('--'+name,required=True)
    p.add_argument('--first-batch',type=int,default=1);p.add_argument('--max-batches',type=int,default=0);p.add_argument('--final-output')
    run(p.parse_args())
