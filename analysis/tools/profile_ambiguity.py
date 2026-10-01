"""Additional exact aggregates on the local audit cache, without training a model."""
from pathlib import Path
import sys, json
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.analysis-deps'))
import duckdb
db=duckdb.connect(str(ROOT/'analysis/.cache/audit.duckdb'))
db.execute("SET memory_limit='1200MB'")
db.execute('SET threads=2')
db.execute("SET temp_directory='"+str(ROOT/'analysis/.cache/spill').replace('\\','/')+"'")

def q(sql):
    cur=db.execute(sql)
    cols=[d[0] for d in cur.description]
    return [dict(zip(cols,r)) for r in cur.fetchall()]

out={}
for split in ['train','test']:
    key="trim(regexp_replace(lower(strip_accents(business_name)), '[^\\p{L}\\p{N}]+', ' ', 'g'))"
    db.execute(f'CREATE OR REPLACE TEMP TABLE names AS SELECT country,{key} AS name_key,count(*) AS n FROM {split}_source1 GROUP BY ALL')
    out[split+'_name_ambiguity']=q('''SELECT country,sum(n) AS entities,count(*) AS unique_name_keys,
        sum(n) FILTER(WHERE n>1) AS entities_sharing_name_key,max(n) AS largest_group,
        sum(n*(n-1)) AS other_reference_pairs_sharing_name FROM names GROUP BY country ORDER BY country''')
    out[split+'_common_names']=q('SELECT * FROM names WHERE n>1 ORDER BY n DESC LIMIT 5')

out['positive_pair_missing_addresses']=q('''SELECT s.country,substr(e.target_id,1,2) AS source,
    count(*) AS pairs,count(*) FILTER(WHERE t.business_address IS NULL) AS missing_address
    FROM edges e JOIN train_source1 s ON e.source1_entity_id=s.entity_id
    JOIN targets_train t ON e.target_id=t.entity_id GROUP BY ALL ORDER BY country,source''')
out['address_numeric_coverage']=[]
for table in ['train_source1','test_source1']:
    for row in q(f'''SELECT country,count(*) AS rows,
        count(*) FILTER(WHERE regexp_matches(business_address,'[0-9]')) AS address_has_digit,
        count(*) FILTER(WHERE regexp_matches(business_address,'\\b[0-9]{{5,6}}\\b')) AS address_has_5_or_6_digit_token
        FROM {table} GROUP BY country'''):
        row['table']=table
        out['address_numeric_coverage'].append(row)

out['script_name_counts']=[]
scripts={'devanagari':(0x0900,0x097F),'bengali':(0x0980,0x09FF),'gurmukhi':(0x0A00,0x0A7F),
    'gujarati':(0x0A80,0x0AFF),'odia':(0x0B00,0x0B7F),'tamil':(0x0B80,0x0BFF),'telugu':(0x0C00,0x0C7F),
    'kannada':(0x0C80,0x0CFF),'malayalam':(0x0D00,0x0D7F)}
expressions=','.join(f"count(*) FILTER(WHERE regexp_matches(business_name,'[\\x{{{a:04X}}}-\\x{{{b:04X}}}]')) AS {name}" for name,(a,b) in scripts.items())
for table in ['train_source2','train_source3','test_source2','test_source3']:
    row=q(f"SELECT {expressions} FROM {table} WHERE country='India'")[0]
    row['table']=table
    out['script_name_counts'].append(row)
(ROOT/'analysis/reports/ambiguity_profile.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(out,ensure_ascii=True,indent=2))
