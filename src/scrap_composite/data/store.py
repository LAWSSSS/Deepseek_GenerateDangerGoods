"""SQLite queue with exclusive claims and resumable stage checkpoints."""
import json
import sqlite3
import time
from pathlib import Path
from uuid import uuid4
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
        CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, record TEXT NOT NULL);
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

    def remaining(self):
        return self.db.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0]

    def job_stage(self, ident):
        row = self.db.execute("SELECT stage FROM jobs WHERE id=?", (ident,)).fetchone()
        return row['stage'] if row else None

    def record_run(self, run):
        with self.db:
            self.db.execute(
                "INSERT INTO runs(id, record) VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET record=excluded.record",
                (run['id'], json.dumps(run, ensure_ascii=False, allow_nan=False)),
            )

    def last_run(self):
        row = self.db.execute("SELECT record FROM runs ORDER BY rowid DESC LIMIT 1").fetchone()
        return json.loads(row['record']) if row else None

    def last_interruption(self):
        row = self.db.execute(
            "SELECT record FROM runs WHERE json_extract(record, '$.state') = 'interrupted' "
            "OR json_extract(record, '$.state') GLOB 'paused_*' ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        return json.loads(row['record']) if row else None

    def interrupt_stale_run(self, legacy_run=None):
        """Reconcile an unclosed attempt only while holding the exclusive batch lease."""
        run = self.last_run()
        if run and run['state'] != 'running':
            return run
        active = self.db.execute(
            "SELECT id, stage FROM jobs WHERE state='running' ORDER BY updated DESC, ordinal LIMIT 1"
        ).fetchone()
        if not run:
            if not active and (legacy_run or {}).get('state') != 'running':
                return None
            run = dict(legacy_run or {})
            if not run.get('id'):
                run['id'] = 'legacy-' + uuid4().hex
        if active and not run.get('error'):
            run.update(job_id=active['id'], stage=active['stage'])
        run.setdefault('stage', None)
        run.setdefault('job_id', None)
        counts = self.counts()
        run.update(
            state='interrupted', error=run.get('error') or '运行进程已退出，未留下终态记录。具体退出原因未知。',
            remaining=self.remaining() if counts else run.get('remaining'),
        )
        self.record_run(run)
        return run

    def begin_run(self, run_id, stage):
        # Preserve the previous attempt before recover() requeues its in-flight jobs.
        previous = self.last_run()
        if not previous or previous['id'] != run_id:
            self.interrupt_stale_run()
            previous = {}
        elif previous['state'] != 'running':
            raise ValueError('该次运行已结束，请使用新的运行编号')
        run = dict(id=run_id, state='running', stage=stage, job_id=None, error=None,
                   remaining=self.remaining() if self.counts() else previous.get('remaining'))
        self.record_run(run)
        return run

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
            self.db.execute("UPDATE jobs SET state='queued' WHERE id=? AND state='running'", (ident,))

    def rows(self):
        return [self.decode(r) for r in self.db.execute("SELECT * FROM jobs ORDER BY ordinal")]

    def counts(self):
        return dict(self.db.execute("SELECT state,count(*) FROM jobs GROUP BY state"))

    def close(self):
        self.db.close()
