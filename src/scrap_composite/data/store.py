"""SQLite queue with exclusive claims and resumable stage checkpoints."""
import json
import sqlite3
import time
from pathlib import Path
from .files import signature


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.dispatch_asset=None
        self.dispatch_count=0
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs (
          id TEXT PRIMARY KEY, ordinal INTEGER NOT NULL, payload TEXT NOT NULL,
          state TEXT NOT NULL DEFAULT 'queued', stage TEXT NOT NULL DEFAULT 'queued',
          checkpoint TEXT NOT NULL DEFAULT '{}', result TEXT, updated REAL NOT NULL);
        """)
        self.db.commit()

    def initialize(self, config_signature, jobs):
        row = self.db.execute("SELECT value FROM meta WHERE key='signature'").fetchone()
        if row and row[0] != config_signature:
            raise ValueError("输出目录属于不同配置；请使用新目录")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('signature',?)", (config_signature,))
            for i, job in enumerate(jobs):
                ident = f"{i+1:04d}-"+signature(job)[:10]
                self.db.execute("INSERT OR IGNORE INTO jobs(id,ordinal,payload,updated) VALUES (?,?,?,?)",
                                (ident, i, json.dumps(job, ensure_ascii=False), time.time()))

    def recover(self):
        # Called only after the exclusive batch process lock is held.
        with self.db:
            self.db.execute("UPDATE jobs SET state='queued' WHERE state='running'")

    def claim(self, group_size=1):
        self.db.execute("BEGIN IMMEDIATE")
        row=None
        if self.dispatch_asset is not None and self.dispatch_count<group_size:
            row=self.db.execute("SELECT * FROM jobs WHERE state='queued' "
                "AND json_extract(payload,'$.asset_id')=? ORDER BY ordinal LIMIT 1",
                (self.dispatch_asset,)).fetchone()
        if row is None:
            row=self.db.execute("SELECT * FROM jobs WHERE state='queued' "
                "ORDER BY (stage<>'queued') DESC, ordinal LIMIT 1").fetchone()
            self.dispatch_count=0
        if row:
            self.db.execute("UPDATE jobs SET state='running',updated=? WHERE id=?", (time.time(), row['id']))
        self.db.commit()
        if row is None:return None
        job=self.decode(row)
        self.dispatch_asset=job['payload'].get('asset_id')
        self.dispatch_count+=1
        return job

    @staticmethod
    def decode(row):
        d = dict(row)
        for k in ('payload', 'checkpoint', 'result'):
            d[k] = json.loads(d[k]) if d[k] else None
        return d

    def checkpoint(self, ident, stage, data):
        with self.db:
            self.db.execute("UPDATE jobs SET stage=?,checkpoint=?,updated=? WHERE id=?",
                            (stage, json.dumps(data, ensure_ascii=False), time.time(), ident))

    def finish(self, ident, state, result):
        if state not in ('accepted_display', 'review', 'failed'):
            raise ValueError('invalid terminal state')
        with self.db:
            self.db.execute("UPDATE jobs SET state=?,stage='done',result=?,updated=? WHERE id=?",
                            (state, json.dumps(result, ensure_ascii=False), time.time(), ident))

    def pause(self, ident):
        with self.db:
            self.db.execute("UPDATE jobs SET state='queued' WHERE id=?", (ident,))

    def rows(self):
        return [self.decode(r) for r in self.db.execute("SELECT * FROM jobs ORDER BY ordinal")]

    def counts(self):
        return dict(self.db.execute("SELECT state,count(*) FROM jobs GROUP BY state"))

    def close(self):
        self.db.close()
