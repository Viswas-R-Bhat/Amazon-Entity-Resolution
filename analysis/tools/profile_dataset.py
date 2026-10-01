"""Read-only, resumable audit of the supplied ER dataset. No network calls."""
from pathlib import Path
import csv
import json
import sys
import time
import unicodedata
import re
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.analysis-deps'))
import duckdb
from rapidfuzz.fuzz import ratio

DATA = ROOT / '6ab10eb3b23ba_student_resource/student_resource/dataset'
OUT = ROOT / 'analysis/reports'
CACHE = ROOT / 'analysis/.cache'
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)
input_stats = {p.relative_to(DATA).as_posix(): [p.stat().st_size, p.stat().st_mtime_ns]
               for p in sorted(DATA.rglob('*.tsv'))}
stamp = CACHE / 'input_stats.json'
if stamp.exists() and json.loads(stamp.read_text(encoding='utf-8')) != input_stats:
    raise RuntimeError('Dataset changed. Use a fresh audit cache before profiling again.')
db = duckdb.connect(str(CACHE / 'audit.duckdb'))
stamp.write_text(json.dumps(input_stats, indent=2), encoding='utf-8')
db.execute("SET memory_limit='1600MB'")
db.execute('SET threads=2')
db.execute('SET preserve_insertion_order=false')
db.execute("SET temp_directory='" + str(CACHE / 'spill').replace('\\', '/') + "'")
report = {'duckdb_version': duckdb.__version__, 'files': {}, 'checks': {}, 'sampling': {}}
start = time.monotonic()


def query(sql, args=None):
    cur = db.execute(sql, args or [])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def save():
    report['elapsed_seconds'] = round(time.monotonic() - start, 1)
    (OUT / 'dataset_profile.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')


def log(message):
    print(f'[{time.monotonic() - start:.1f}s] {message}', flush=True)


tables = []
for split in ['train', 'test']:
    for source in [1, 2, 3]:
        table = f'{split}_source{source}'
        tables.append(table)
        path = DATA / split / (table + '.tsv')
        with path.open(encoding='utf-8', newline='') as handle:
            header = next(csv.reader(handle, delimiter='\t'))
        assert header == ['entity_id', 'business_name', 'business_address', 'country'], (table, header)
        db.execute(f'''CREATE TABLE IF NOT EXISTS {table} AS
            SELECT * FROM read_csv(?, delim='\t', header=true, all_varchar=true,
            nullstr='', strict_mode=true, ignore_errors=false)''', [str(path)])
        overview = query(f'''SELECT count(*) AS rows, count(DISTINCT entity_id) AS unique_ids,
            count(*) FILTER(WHERE entity_id IS NULL OR NOT starts_with(entity_id, 'S{source}-')) AS invalid_ids,
            count(*) FILTER(WHERE country IS NULL OR trim(country)='') AS missing_country
            FROM {table}''')[0]
        by_country = query(f'''SELECT country, count(*) AS rows,
            count(*) FILTER(WHERE business_name IS NULL OR trim(business_name)='') AS missing_name,
            count(*) FILTER(WHERE business_address IS NULL OR trim(business_address)='') AS missing_address,
            count(*) FILTER(WHERE (business_name IS NULL OR trim(business_name)='') AND
            (business_address IS NULL OR trim(business_address)='')) AS missing_both,
            count(*) FILTER(WHERE regexp_matches(business_name, '[^\\x00-\\x7F]')) AS non_ascii_name,
            count(*) FILTER(WHERE regexp_matches(business_address, '[^\\x00-\\x7F]')) AS non_ascii_address,
            count(*) FILTER(WHERE regexp_matches(business_name, '[\\x{{0900}}-\\x{{097F}}]')) AS devanagari_name,
            count(*) FILTER(WHERE regexp_matches(business_name, '[\\x{{0400}}-\\x{{04FF}}]')) AS cyrillic_name,
            round(avg(length(business_name)), 1) AS mean_name_chars,
            round(avg(length(business_address)), 1) AS mean_address_chars
            FROM {table} GROUP BY country ORDER BY country''')
        overview.update({'size_bytes': path.stat().st_size, 'columns': header, 'by_country': by_country})
        report['files'][table] = overview
        log(f'{table}: ' + json.dumps(overview))
        save()

db.execute('CREATE OR REPLACE VIEW all_train AS ' + ' UNION ALL '.join(f'SELECT * FROM train_source{s}' for s in [1,2,3]))
db.execute('CREATE OR REPLACE VIEW targets_train AS SELECT * FROM train_source2 UNION ALL SELECT * FROM train_source3')
gt_path = DATA / 'train/train_ground_truth.tsv'
db.execute('''CREATE TABLE IF NOT EXISTS ground_truth AS
    SELECT source1_entity_id, coalesce(matched_entity_ids, '') AS matched_entity_ids,
    CASE WHEN coalesce(matched_entity_ids, '')='' THEN []::VARCHAR[]
    ELSE string_split(matched_entity_ids, ',') END AS targets
    FROM read_csv(?, delim='\t', header=true, all_varchar=true, strict_mode=true)''', [str(gt_path)])
db.execute('''CREATE TABLE IF NOT EXISTS edges AS
    SELECT source1_entity_id, unnest(targets) AS target_id FROM ground_truth''')
report['ground_truth'] = query('''SELECT count(*) AS rows,
    count(DISTINCT source1_entity_id) AS unique_source1_ids,
    count(*) FILTER(WHERE len(targets)=0) AS singletons,
    sum(len(targets)) AS positive_pairs, min(len(targets)) AS min_matches,
    max(len(targets)) AS max_matches, avg(len(targets)) AS mean_matches,
    count(*) FILTER(WHERE len(targets)<>len(list_distinct(targets))) AS rows_with_duplicate_targets
    FROM ground_truth''')[0]
report['ground_truth']['by_country'] = query('''SELECT s.country, count(*) AS entities,
    count(*) FILTER(WHERE len(g.targets)=0) AS singletons, sum(len(g.targets)) AS positive_pairs,
    avg(len(g.targets)) AS mean_matches
    FROM ground_truth g JOIN train_source1 s ON g.source1_entity_id=s.entity_id
    GROUP BY s.country ORDER BY s.country''')
report['ground_truth']['match_count_distribution'] = query('''SELECT len(targets) AS matches,
    count(*) AS entities FROM ground_truth GROUP BY len(targets) ORDER BY matches''')
report['ground_truth']['source_pattern'] = query('''WITH counts AS (
    SELECT len(list_filter(targets, x -> starts_with(x, 'S2-'))) AS n2,
    len(list_filter(targets, x -> starts_with(x, 'S3-'))) AS n3 FROM ground_truth)
    SELECT count(*) FILTER(WHERE n2>0 AND n3>0) AS both_sources,
    count(*) FILTER(WHERE n2>0 AND n3=0) AS source2_only,
    count(*) FILTER(WHERE n3>0 AND n2=0) AS source3_only,
    count(*) FILTER(WHERE n2>1) AS multiple_source2,
    count(*) FILTER(WHERE n3>1) AS multiple_source3 FROM counts''')[0]
checks = report['checks']
checks['source1_without_ground_truth'] = query('''SELECT count(*) AS n FROM train_source1 s
    ANTI JOIN ground_truth g ON s.entity_id=g.source1_entity_id''')[0]['n']
checks['ground_truth_without_source1'] = query('''SELECT count(*) AS n FROM ground_truth g
    ANTI JOIN train_source1 s ON s.entity_id=g.source1_entity_id''')[0]['n']
checks['unknown_target_ids'] = query('''SELECT count(*) AS n FROM edges e
    ANTI JOIN targets_train t ON e.target_id=t.entity_id''')[0]['n']
checks['targets_assigned_to_multiple_source1'] = query('''SELECT count(*) AS n FROM
    (SELECT target_id FROM edges GROUP BY target_id HAVING count(DISTINCT source1_entity_id)>1)''')[0]['n']
report['ground_truth']['labeled_target_coverage'] = query('''SELECT substr(t.entity_id,1,2) AS source,
    t.country, count(*) AS target_records, count(e.target_id) AS labeled_target_records
    FROM targets_train t LEFT JOIN (SELECT DISTINCT target_id FROM edges) e ON t.entity_id=e.target_id
    GROUP BY source,t.country ORDER BY source,t.country''')
checks['cross_country_positive_pairs'] = query('''SELECT count(*) AS n FROM edges e
    JOIN train_source1 s ON s.entity_id=e.source1_entity_id
    JOIN targets_train t ON t.entity_id=e.target_id WHERE s.country IS DISTINCT FROM t.country''')[0]['n']
log('Ground-truth integrity and distributions complete')
save()

checks['train_test_id_overlap'] = {}
for source in [1,2,3]:
    checks['train_test_id_overlap'][f'S{source}'] = query(f'''SELECT count(*) AS n
        FROM train_source{source} a JOIN test_source{source} b USING(entity_id)''')[0]['n']
report['source1_exact_content_duplicates'] = {}
for split in ['train', 'test']:
    report['source1_exact_content_duplicates'][split] = query(f'''SELECT count(*) AS duplicate_groups,
        coalesce(sum(n-1),0) AS extra_rows FROM (SELECT country,business_name,business_address,count(*) AS n
        FROM {split}_source1 GROUP BY ALL HAVING count(*)>1)''')[0]
checks['train_test_exact_source1_content_overlap'] = query('''SELECT count(*) AS test_rows FROM test_source1 t
    SEMI JOIN train_source1 s ON t.country=s.country AND t.business_name=s.business_name
    AND t.business_address=s.business_address''')[0]['test_rows']
report['comparison_space'] = {}
for split in ['train','test']:
    counts = {s: report['files'][f'{split}_source{s}']['rows'] for s in [1,2,3]}
    country_counts = {s: {r['country']:r['rows'] for r in report['files'][f'{split}_source{s}']['by_country']} for s in [1,2,3]}
    report['comparison_space'][split] = {
        'all_pairs': counts[1]*(counts[2]+counts[3]),
        'same_country_pairs': sum(n*(country_counts[2].get(c,0)+country_counts[3].get(c,0)) for c,n in country_counts[1].items())}
log('Duplicate, overlap and scale checks complete')
save()

# Sample by reference entity so all its labeled matches stay together.
sample = query('''SELECT s.entity_id AS source1_id, t.entity_id AS target_id, s.country,
    s.business_name AS name1,t.business_name AS name2,
    s.business_address AS address1,t.business_address AS address2
    FROM edges e JOIN train_source1 s ON e.source1_entity_id=s.entity_id
    JOIN targets_train t ON e.target_id=t.entity_id WHERE hash(s.entity_id)%1000<10''')


def normalize(value):
    value = unicodedata.normalize('NFKD', value or '').casefold()
    value = ''.join(c for c in value if not unicodedata.combining(c))
    return ' '.join(re.findall(r'\w+', value, flags=re.UNICODE))


aggregates = defaultdict(Counter)
examples = defaultdict(list)
similarities = defaultdict(list)
for pair in sample:
    name1,name2 = normalize(pair['name1']),normalize(pair['name2'])
    addr1,addr2 = normalize(pair['address1']),normalize(pair['address2'])
    metrics = {'pairs':1, 'name_exact_nonempty':int(bool(name1) and name1==name2),
        'address_exact_nonempty':int(bool(addr1) and addr1==addr2),
        'either_name_or_address_exact_nonempty':int((bool(name1) and name1==name2) or (bool(addr1) and addr1==addr2)),
        'either_address_missing':int(not addr1 or not addr2),
        'either_name_missing':int(not name1 or not name2),
        'name_token_overlap_zero':int(not (set(name1.split()) & set(name2.split())))}
    nr = ratio(name1,name2) if name1 and name2 else None
    ar = ratio(addr1,addr2) if addr1 and addr2 else None
    for group in ['all',pair['country'],pair['target_id'][:2]]:
        aggregates[group].update(metrics)
        if nr is not None: similarities[group+' name_ratio'].append(nr)
        if ar is not None: similarities[group+' address_ratio'].append(ar)
    categories = []
    if nr is not None and nr<30: categories.append('low_name_similarity')
    if metrics['either_address_missing']: categories.append('missing_address')
    if name1 and name1==name2 and ar is not None and ar<35: categories.append('same_name_different_address')
    if any('\u0900' <= c <= '\u097f' for c in (pair['name1'] or '')+(pair['name2'] or '')): categories.append('devanagari')
    for category in categories:
        if len(examples[category])<4: examples[category].append(pair)
report['sampling'] = {'method':'DuckDB hash(source1_id) % 1000 < 10; all labeled pairs for selected entities',
    'reference_entities':len({p['source1_id'] for p in sample}),
    'normalization':'Unicode NFKD, remove combining marks, casefold, retain Unicode word tokens; exploratory only',
    'positive_pair_statistics':dict(aggregates),'examples':dict(examples),'similarity_quantiles':{}}
for key, values in similarities.items():
    values.sort()
    report['sampling']['similarity_quantiles'][key] = {f'p{p}':round(values[int((len(values)-1)*p/100)],2) for p in [0,10,25,50,75,90,100]}
save()
db.execute('CHECKPOINT')
log('Audit complete: analysis/reports/dataset_profile.json')
