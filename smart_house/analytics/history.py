"""Separate transactional storage for uploaded measurements and evaluations."""
import datetime as dt
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


def encode(value):
    return json.dumps(value, ensure_ascii=False, default=str, allow_nan=False)


class History:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS imports (
                    id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, filename TEXT NOT NULL,
                    file_sha256 TEXT NOT NULL, model_sha256 TEXT NOT NULL, count INTEGER NOT NULL,
                    validation TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS measurements (
                    import_id INTEGER NOT NULL REFERENCES imports(id), position INTEGER NOT NULL,
                    timestamp TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(import_id, position)
                );
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL
                );
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def save_import(self, filename, file_digest, model_digest, report, results):
        with self.connect() as db:
            cur = db.execute('INSERT INTO imports(created_at,filename,file_sha256,model_sha256,count,validation) VALUES(?,?,?,?,?,?)',
                             (dt.datetime.now(dt.timezone.utc).isoformat(), filename, file_digest,
                              model_digest, len(results), encode(report)))
            ident = cur.lastrowid
            db.executemany('INSERT INTO measurements VALUES(?,?,?,?)',
                           [(ident, i, r['timestamp'], encode(r)) for i, r in enumerate(results, 1)])
        return ident

    def list_imports(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute(
                'SELECT id,created_at,filename,count FROM imports ORDER BY id DESC LIMIT 100')]

    def get_import(self, ident):
        with self.connect() as db:
            row = db.execute('SELECT * FROM imports WHERE id=?', (ident,)).fetchone()
            if row is None:
                raise KeyError('Import not found')
            payload = dict(row)
            payload['validation'] = json.loads(payload.pop('validation'))
            payload['readings'] = [json.loads(row[0]) for row in db.execute(
                'SELECT payload FROM measurements WHERE import_id=? ORDER BY position', (ident,))]
            return payload

    def save_evaluation(self, payload):
        with self.connect() as db:
            return db.execute('INSERT INTO evaluations(created_at,payload) VALUES(?,?)',
                              (dt.datetime.now(dt.timezone.utc).isoformat(), encode(payload))).lastrowid

    def list_evaluations(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute('SELECT id,created_at FROM evaluations ORDER BY id DESC LIMIT 100')]

    def get_evaluation(self, ident):
        with self.connect() as db:
            row = db.execute('SELECT payload FROM evaluations WHERE id=?', (ident,)).fetchone()
            if row is None:
                raise KeyError('Evaluation not found')
            return {'id': ident, **json.loads(row[0])}
