"""Bounded-memory, resumable SQLite FTS5 target index and multi-pass retrieval."""
import json
import sqlite3
from functools import lru_cache
from pathlib import Path
from rapidfuzz.fuzz import ratio, token_sort_ratio
from .io import log, rows, signature
from .text import record, phonetic

INDEX_VERSION = 1
RETRIEVAL_VERSION = 1
COLUMNS = "eid,country,name,address,core,uname,uaddress"


def connect(path):
    db = sqlite3.connect(str(path))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA cache_size=-98304")  # 96 MiB, independent of corpus size
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA mmap_size=0")
    return db


def build_index(paths, destination, batch_size=5000):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    db = connect(destination)
    try:
        _build_index(paths, destination, batch_size, db)
    finally:
        db.close()


def _build_index(paths, destination, batch_size, db):
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    expected = {"version": INDEX_VERSION, "inputs": signature(paths)}
    previous = db.execute("SELECT value FROM meta WHERE key='config'").fetchone()
    if previous and json.loads(previous[0]) != expected:
        raise ValueError("Index inputs/version changed. Select a new work directory.")
    db.execute("INSERT OR IGNORE INTO meta VALUES('config',?)", (json.dumps(expected),))
    db.execute(f"CREATE TABLE IF NOT EXISTS records(rid INTEGER PRIMARY KEY, eid TEXT UNIQUE NOT NULL, "
               "country TEXT NOT NULL,name TEXT NOT NULL,address TEXT NOT NULL,core TEXT NOT NULL,"
               "uname TEXT NOT NULL,uaddress TEXT NOT NULL)")
    db.execute("CREATE INDEX IF NOT EXISTS exact_name ON records(country,core)")
    db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(name,address,phon,country,content='',detail=column)")
    db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vocab USING fts5vocab(search,'col')")
    db.commit()
    if db.execute("SELECT value FROM meta WHERE key='complete'").fetchone():
        log(f"Reusing complete index: {destination}")
        return
    for path in paths:
        key = str(Path(path).resolve())
        saved = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        done = int(saved[0]) if saved else 0
        batch = []
        def commit_batch(position):
            with db:
                for item in batch:
                    cur = db.execute(f"INSERT INTO records({COLUMNS}) VALUES(?,?,?,?,?,?,?)",
                                     tuple(item[k] for k in COLUMNS.split(',')))
                    db.execute("INSERT INTO search(rowid,name,address,phon,country) VALUES(?,?,?,?,?)",
                               (cur.lastrowid, item['name'], item['address'], phonetic(item['name']), item['country']))
                db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, str(position)))
            batch.clear()
        for position, row in enumerate(rows(path), 1):
            if position <= done:
                continue
            if not row['entity_id'].startswith(("S2-", "S3-")) or not row['country'].strip():
                raise ValueError(f"Invalid target/country in {path}: {row['entity_id']}")
            batch.append(record(row))
            if len(batch) >= batch_size:
                commit_batch(position)
                if position % 100000 == 0:
                    log(f"Indexed {Path(path).name}: {position:,} rows (safe to resume)")
        if batch:
            commit_batch(position)
    log("Finishing search index...")
    db.execute("INSERT INTO search(search) VALUES('optimize')")
    db.execute("INSERT OR REPLACE INTO meta VALUES('complete','1')")
    db.commit()
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    count = db.execute("SELECT count(*) FROM records").fetchone()[0]
    log(f"Index ready: {count:,} targets")


class Retriever:
    def __init__(self, path, cap=60, per_pass=80, max_df=5000):
        if not Path(path).is_file():
            raise ValueError(f'Index not found: {path}. Run the index stage first.')
        self.db = connect(path)
        if not self.db.execute("SELECT 1 FROM meta WHERE key='complete'").fetchone():
            self.db.close()
            raise ValueError("Index is incomplete; rerun index first.")
        self.cap, self.per_pass, self.max_df = cap, per_pass, max_df

    def close(self):
        self.df.cache_clear()
        self.db.close()

    @lru_cache(maxsize=50000)
    def df(self, column, token):
        row = self.db.execute("SELECT doc FROM vocab WHERE term=? AND col=?", (token, column)).fetchone()
        return row[0] if row else 0

    def terms(self, column, text):
        return sorted(((self.df(column, t), t) for t in set(text.split()) if len(t) >= 2),
                      key=lambda x: (x[0], x[1]))

    @staticmethod
    def clause(column, terms):
        return '(' + ' OR '.join(f'{column}:"{t}"' for t in terms) + ')'

    def ranked_hits(self, expression, country):
        # FTS5's native rank cursor can stop early. Preserve the previous
        # (BM25, rowid) ordering by collecting every tie at the cutoff.
        cursor = self.db.execute(
            "SELECT search.rowid,search.rank FROM search CROSS JOIN records AS r "
            "ON r.rid=search.rowid WHERE search MATCH ? AND r.country=? "
            "AND rank MATCH 'bm25(3.0,1.0,2.0,0.0)' ORDER BY search.rank",
            (expression, country))
        fallback = False
        try:
            hits = cursor.fetchmany(self.per_pass)
            if len(hits) == self.per_pass:
                boundary = hits[-1][1]
                for hit in cursor:
                    if hit[1] > boundary:
                        break
                    hits.append(hit)
                    if len(hits) > max(4096, self.per_pass * 4):
                        fallback = True
                        break
        finally:
            cursor.close()
        if fallback:
            return self.db.execute(
                "SELECT search.rowid FROM search CROSS JOIN records AS r "
                "ON r.rid=search.rowid WHERE search MATCH ? AND r.country=? "
                "ORDER BY bm25(search,3.0,1.0,2.0,0.0),search.rowid LIMIT ?",
                (expression, country, self.per_pass)).fetchall()
        hits.sort(key=lambda hit: (hit[1], hit[0]))
        return hits[:self.per_pass]

    def candidates(self, query):
        # Only normalized alphanumeric tokens enter MATCH expressions.
        found = {}
        exact_count = self.db.execute("SELECT count(*) FROM records WHERE country=? AND core=?",
                                     (query['country'], query['core'])).fetchone()[0] if query['core'] else 0
        if exact_count and exact_count <= self.per_pass * 3:
            for row in self.db.execute("SELECT * FROM records WHERE country=? AND core=? ORDER BY rid",
                                       (query['country'], query['core'])):
                found[row['rid']] = (dict(row), 1)
        names = [(n, t) for n, t in self.terms('name', query['name']) if n]
        addresses = [(n, t) for n, t in self.terms('address', query['address']) if n and not t.isdigit()]
        sounds = [(n, t) for n, t in self.terms('phon', phonetic(query['name'])) if n]
        expressions = []
        for column, terms in [('name', names), ('phon', sounds)]:
            rare = [t for n, t in terms if n <= self.max_df][:3]
            if rare:
                expressions.append(self.clause(column, rare))
            if len(terms) >= 2:
                expressions.append(' AND '.join(f'{column}:"{t}"' for _, t in terms[:2]))
        if names and addresses:
            expressions.append(self.clause('name', [t for _, t in names[:2]]) + ' AND ' +
                               self.clause('address', [t for _, t in addresses[:2]]))
        if len(addresses) >= 2:
            expressions.append(' AND '.join(f'address:"{t}"' for _, t in addresses[:2]))
        candidate_rids = []
        seen_rids = set(found)
        for expression in dict.fromkeys(expressions):
            # Country has zero BM25 weight. Keep it out of MATCH to avoid
            # scanning its multi-million-row posting list for each ranking.
            # CROSS JOIN forces text retrieval before primary-key country checks.
            hits = self.ranked_hits(expression, query['country'])
            for hit in hits:
                rid = hit[0]
                if rid not in seen_rids:
                    seen_rids.add(rid)
                    candidate_rids.append(rid)
        # Fetch all newly retrieved records in a few indexed lookups instead
        # of one SQLite round trip per candidate. Keep expression/rank order
        # identical to the prior insertion order used below.
        fetched = {}
        for offset in range(0, len(candidate_rids), 500):
            ids = candidate_rids[offset:offset + 500]
            placeholders = ','.join('?' for _ in ids)
            fetched.update({row['rid']: dict(row) for row in self.db.execute(
                f"SELECT * FROM records WHERE rid IN ({placeholders})", ids)})
        for rid in candidate_rids:
            found[rid] = (fetched[rid], 0)
        scored = []
        qp = phonetic(query['name'])
        for target, exact in found.values():
            n = token_sort_ratio(query['name'], target['name']) / 100
            a = token_sort_ratio(query['address'], target['address']) / 100 if query['address'] and target['address'] else 0
            tp = phonetic(target['name'])
            p = ratio(qp, tp) / 100 if qp and tp else 0
            rank = max(n, .92 * p) * .65 + a * .35
            scored.append((rank, target['eid'], target, exact))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [(target, {'retrieval_score': rank, 'exact_block': exact,
                          'name_frequency': exact_count}) for rank, _, target, exact in scored[:self.cap]]
