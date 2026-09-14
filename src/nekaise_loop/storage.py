"""SQLite persistence. JSON payloads evolve independently of indexed lifecycle fields."""
from __future__ import annotations

import json
import fcntl
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class Store:
    def __init__(self, workspace: Path):
        self.path = workspace / "loop.sqlite3"
        # journal_mode cannot be safely changed concurrently, even with SQLite's
        # busy timeout. Serialize initialization before opening the connection.
        with (workspace/"schema.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._initialize()

    def _initialize(self):
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
                INSERT INTO schema_version SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_version);
                CREATE TABLE IF NOT EXISTS campaigns (id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL, config TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, error TEXT);
                CREATE TABLE IF NOT EXISTS rounds (id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL REFERENCES campaigns(id), number INTEGER NOT NULL, status TEXT NOT NULL, stage TEXT, model_before TEXT NOT NULL, checkpoint TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, error TEXT, UNIQUE(campaign_id,number));
                CREATE TABLE IF NOT EXISTS stage_runs (id INTEGER PRIMARY KEY, round_id TEXT NOT NULL REFERENCES rounds(id), stage TEXT NOT NULL, attempt INTEGER NOT NULL, status TEXT NOT NULL, input_hash TEXT NOT NULL, artifact TEXT, started_at TEXT NOT NULL, finished_at TEXT, error TEXT, process_pid INTEGER, process_start TEXT, UNIQUE(round_id,stage,attempt));
                CREATE TABLE IF NOT EXISTS records (id TEXT NOT NULL, round_id TEXT NOT NULL REFERENCES rounds(id), kind TEXT NOT NULL, position INTEGER NOT NULL, data TEXT NOT NULL, PRIMARY KEY(round_id,kind,id));
                CREATE TABLE IF NOT EXISTS metrics (id INTEGER PRIMARY KEY, round_id TEXT NOT NULL REFERENCES rounds(id), attempt INTEGER NOT NULL, step INTEGER NOT NULL, data TEXT NOT NULL, created_at TEXT NOT NULL, UNIQUE(round_id,attempt,step));
                CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, campaign_id TEXT REFERENCES campaigns(id), round_id TEXT REFERENCES rounds(id), kind TEXT NOT NULL, message TEXT NOT NULL, data TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS actions (id INTEGER PRIMARY KEY, campaign_id TEXT NOT NULL REFERENCES campaigns(id), kind TEXT NOT NULL, created_at TEXT NOT NULL, handled_at TEXT);
                CREATE TABLE IF NOT EXISTS teacher_calls (id INTEGER PRIMARY KEY, campaign_id TEXT NOT NULL REFERENCES campaigns(id), round_id TEXT NOT NULL REFERENCES rounds(id), purpose TEXT NOT NULL, status TEXT NOT NULL, usage TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS idx_rounds_campaign ON rounds(campaign_id,number);
                CREATE INDEX IF NOT EXISTS idx_stages_round ON stage_runs(round_id,stage,status);
                CREATE INDEX IF NOT EXISTS idx_events_campaign ON events(campaign_id,id);
                CREATE INDEX IF NOT EXISTS idx_actions_pending ON actions(id) WHERE handled_at IS NULL;
            """)
            # Serialize migrations across API, supervisor and worker startup.
            db.execute("BEGIN IMMEDIATE")
            version = db.execute("SELECT version FROM schema_version").fetchone()[0]
            if version == 1:
                db.execute("ALTER TABLE campaigns ADD COLUMN parent_campaign_id TEXT")
                db.execute("ALTER TABLE campaigns ADD COLUMN context_artifact TEXT")
                db.execute("ALTER TABLE campaigns ADD COLUMN teacher_budget_since TEXT")
                db.execute("ALTER TABLE campaigns ADD COLUMN implementation_hash TEXT")
                db.execute("UPDATE schema_version SET version=2")
                version = 2
            if version != 2:
                raise RuntimeError("Unsupported database schema version")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS recoveries (
                    id INTEGER PRIMARY KEY, campaign_id TEXT NOT NULL REFERENCES campaigns(id),
                    round_id TEXT, stage_id INTEGER, kind TEXT NOT NULL, status TEXT NOT NULL,
                    error TEXT NOT NULL, retry_at TEXT, attempts INTEGER NOT NULL DEFAULT 0,
                    decision TEXT, source_hash TEXT NOT NULL, continuation_id TEXT,
                    process_pid INTEGER, process_start TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_one_open_recovery ON recoveries(campaign_id)
                    WHERE status IN ('pending','running','waiting','decided');
                CREATE INDEX IF NOT EXISTS idx_recovery_due ON recoveries(status,retry_at);
            """)

    @contextmanager
    def connect(self, immediate=False):
        db = sqlite3.connect(self.path, timeout=20)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=20000")
        try:
            if immediate:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def query(self, sql, args=()):
        with self.connect() as db:
            return [dict(row) for row in db.execute(sql, args).fetchall()]

    def one(self, sql, args=()):
        rows = self.query(sql, args)
        return rows[0] if rows else None

    def execute(self, sql, args=()):
        with self.connect() as db:
            return db.execute(sql, args).lastrowid

    def event(self, campaign_id, round_id, kind, message, data=None, db=None):
        args = (campaign_id, round_id, kind, message, encode(data or {}), now())
        sql = "INSERT INTO events(campaign_id,round_id,kind,message,data,created_at) VALUES(?,?,?,?,?,?)"
        return db.execute(sql, args).lastrowid if db else self.execute(sql, args)

    def campaign(self, campaign_id):
        row = self.one("SELECT * FROM campaigns WHERE id=?", (campaign_id,))
        if row is None:
            raise KeyError(campaign_id)
        row["config"] = json.loads(row["config"])
        return row

    def set_status(self, campaign_id, status, error=None):
        with self.connect(immediate=True) as db:
            db.execute("UPDATE campaigns SET status=?,error=?,updated_at=? WHERE id=?", (status, error, now(), campaign_id))
            self.event(campaign_id, None, "campaign", f"Campaign {status}", {"status": status, "error": error}, db=db)

    def recover(self, campaign_id, kind, error, *, retry_at=None):
        from .ownership import source_fingerprint
        with self.connect(immediate=True) as db:
            old = db.execute("SELECT id FROM recoveries WHERE campaign_id=? AND status IN ('pending','running','waiting','decided')", (campaign_id,)).fetchone()
            if old:
                return old["id"]
            stage = db.execute("SELECT s.id,s.round_id FROM stage_runs s JOIN rounds r ON s.round_id=r.id WHERE r.campaign_id=? ORDER BY s.id DESC LIMIT 1", (campaign_id,)).fetchone()
            recovery_id = db.execute("INSERT INTO recoveries(campaign_id,round_id,stage_id,kind,status,error,retry_at,source_hash,created_at,updated_at) VALUES(?,?,?,?,'pending',?,?,?,?,?)", (campaign_id, stage["round_id"] if stage else None, stage["id"] if stage else None, kind, str(error)[-3000:], retry_at, source_fingerprint(), now(), now())).lastrowid
            status = "waiting" if kind in {"quota", "rate_limit", "budget"} else "recovering"
            db.execute("UPDATE campaigns SET status=?,error=?,updated_at=? WHERE id=?", (status, str(error)[-3000:], now(), campaign_id))
            self.event(campaign_id, stage["round_id"] if stage else None, "recovery", "Teacher waiting for availability" if status == "waiting" else "Orchestrator wake requested", {"recovery_id": recovery_id, "kind": kind, "retry_at": retry_at}, db=db)
            return recovery_id

    def records(self, round_id, kind):
        return [json.loads(r["data"]) for r in self.query("SELECT data FROM records WHERE round_id=? AND kind=? ORDER BY position,id", (round_id, kind))]

    def put_records(self, round_id, kind, rows, db=None):
        values = [(r["id"], round_id, kind, n, encode(r)) for n, r in enumerate(rows)]
        sql = "INSERT INTO records(id,round_id,kind,position,data) VALUES(?,?,?,?,?) ON CONFLICT(round_id,kind,id) DO UPDATE SET data=excluded.data,position=excluded.position"
        if db:
            db.executemany(sql, values)
        else:
            with self.connect() as conn:
                conn.executemany(sql, values)

    def metrics(self, round_id, limit=600):
        attempt = self.one("SELECT MAX(attempt) AS n FROM metrics WHERE round_id=?", (round_id,))["n"]
        if attempt is None:
            return []
        rows = self.query("SELECT data FROM metrics WHERE round_id=? AND attempt=? ORDER BY step", (round_id, attempt))
        if len(rows) > limit:
            indices = sorted({round(i * (len(rows)-1)/(limit-1)) for i in range(limit)})
            rows = [rows[i] for i in indices]
        return [json.loads(r["data"]) for r in rows]

    def events(self, campaign_id=None, after=0, limit=100):
        where, args = "id>?", [after]
        if campaign_id:
            where += " AND campaign_id=?"
            args.append(campaign_id)
        rows = self.query(f"SELECT * FROM events WHERE {where} ORDER BY id LIMIT ?", (*args, limit))
        for row in rows:
            row["data"] = json.loads(row["data"])
        return rows
