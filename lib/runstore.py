"""Single-machine experiment store for coding agents.

SQLite is the query/index layer; immutable files are the artifact layer.  The database
is deliberately rebuildable from per-run manifests/state and artifact manifests.  No
daemon and no optional dependency are required.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import socket
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

try:  # Imported both as top-level ``runstore`` and package ``lib.runstore``.
    from workspace import Workspace, validate_name
except ModuleNotFoundError:  # pragma: no cover - depends on caller's sys.path
    from lib.workspace import Workspace, validate_name

REPO = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 2
TERMINAL = {"succeeded", "failed", "aborted"}
STATUSES = {"queued", "running", "trained", "evaluating", *TERMINAL}


def new_run_id(now: float | None = None) -> str:
    """Sortable, collision-resistant ID without an external ULID dependency."""
    t = time.time() if now is None else now
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(t))
    micros = int((t % 1) * 1_000_000)
    return f"{stamp}{micros:06d}Z-{secrets.token_hex(6)}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    os.replace(temporary, path)


class RunStore:
    """Durable local run registry and immutable artifact store."""

    JSON_COLUMNS = {
        "command": "command_json",
        "code": "code_json",
        "environment": "environment_json",
        "metadata": "metadata_json",
        "last_metrics": "last_metrics_json",
    }
    RUN_COLUMNS = {
        "experiment", "stage", "kind", "status", "started", "ended", "heartbeat",
        "pid", "host", "model", "model_revision", "dataset_id", "seed",
        "config_sha256", "error", "exit_code", "checkpoint_digest", "checkpoint_path",
        "latest_step", "event_count", "parent_run_id", "retry_of",
        *JSON_COLUMNS.values(),
    }

    def __init__(
        self, repo: str | Path = REPO,
        workspace: Workspace | str | Path | None = None,
    ):
        self.repo = Path(repo).resolve()
        # ``lib.workspace`` and top-level ``workspace`` can both be import names in
        # mixed CLI/test processes; accept the value by protocol, not class identity.
        if isinstance(workspace, Workspace) or (
            workspace is not None
            and hasattr(workspace, "experiments_dir")
            and hasattr(workspace, "store_dir")
        ):
            self.workspace = workspace
        elif workspace is not None:
            self.workspace = Workspace.resolve(self.repo, workspace)
        elif self.repo == REPO:
            self.workspace = Workspace.resolve(self.repo)
        else:
            # An isolated ``RunStore(tmp_path)`` must ignore any ambient real workspace.
            self.workspace = Workspace.legacy(self.repo)
        self.experiments = self.workspace.experiments_dir
        self.root = self.workspace.store_dir
        self.db_path = self.root / "runs.sqlite3"
        self.artifacts_root = self.root / "artifacts"
        self.root.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def stored_path(self, path: str | Path) -> str:
        """Portable database representation for repo/workspace-owned paths."""
        resolved = Path(path).resolve()
        if self.workspace.external:
            try:
                return "workspace:" + str(resolved.relative_to(self.workspace.root))
            except ValueError:
                pass
            try:
                return "repo:" + str(resolved.relative_to(self.repo))
            except ValueError:
                return str(resolved)
        try:
            return str(resolved.relative_to(self.repo))
        except ValueError:
            return str(resolved)

    def resolve_stored_path(self, value: str | Path) -> Path:
        """Resolve new tagged paths plus legacy repo-relative and absolute paths."""
        raw = str(value)
        if raw.startswith("workspace:"):
            base = self.workspace.root
            candidate = (base / raw.removeprefix("workspace:")).resolve()
            try:
                candidate.relative_to(base)
            except ValueError:
                raise ValueError(f"workspace path escapes its root: {raw!r}") from None
            return candidate
        if raw.startswith("repo:"):
            candidate = (self.repo / raw.removeprefix("repo:")).resolve()
            try:
                candidate.relative_to(self.repo)
            except ValueError:
                raise ValueError(f"repo path escapes its root: {raw!r}") from None
            return candidate
        path = Path(raw)
        if path.is_absolute():
            return path.resolve()
        candidate = (self.repo / path).resolve()
        try:
            candidate.relative_to(self.repo)
        except ValueError:
            raise ValueError(f"legacy path escapes repository: {raw!r}") from None
        return candidate

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    experiment TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created REAL NOT NULL,
                    updated REAL NOT NULL,
                    started REAL,
                    ended REAL,
                    heartbeat REAL,
                    pid INTEGER,
                    host TEXT,
                    model TEXT,
                    model_revision TEXT,
                    dataset_id TEXT,
                    seed INTEGER,
                    config_sha256 TEXT,
                    command_json TEXT,
                    code_json TEXT,
                    environment_json TEXT,
                    metadata_json TEXT,
                    error TEXT,
                    exit_code INTEGER,
                    checkpoint_digest TEXT,
                    checkpoint_path TEXT,
                    latest_step INTEGER,
                    event_count INTEGER NOT NULL DEFAULT 0,
                    last_metrics_json TEXT,
                    parent_run_id TEXT,
                    retry_of TEXT,
                    schema_version INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS runs_experiment_created
                    ON runs(experiment, created DESC);
                CREATE INDEX IF NOT EXISTS runs_status_updated
                    ON runs(status, updated DESC);
                CREATE INDEX IF NOT EXISTS runs_dataset
                    ON runs(dataset_id);

                CREATE TABLE IF NOT EXISTS metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    name TEXT NOT NULL,
                    value REAL,
                    n INTEGER,
                    split TEXT NOT NULL DEFAULT '',
                    split_hash TEXT,
                    evaluator TEXT NOT NULL DEFAULT '',
                    evaluator_version TEXT,
                    diagnostic INTEGER NOT NULL DEFAULT 0,
                    created REAL NOT NULL,
                    metadata_json TEXT,
                    UNIQUE(run_id, name, split, evaluator, diagnostic)
                );
                CREATE INDEX IF NOT EXISTS metrics_lookup
                    ON metrics(name, split, diagnostic, value DESC);

                CREATE TABLE IF NOT EXISTS artifacts (
                    digest TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created REAL NOT NULL,
                    state TEXT NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT
                );
                CREATE TABLE IF NOT EXISTS run_artifacts (
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    digest TEXT NOT NULL REFERENCES artifacts(digest),
                    role TEXT NOT NULL,
                    PRIMARY KEY(run_id, role)
                );
                CREATE TABLE IF NOT EXISTS pointers (
                    experiment TEXT NOT NULL,
                    name TEXT NOT NULL,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    digest TEXT NOT NULL REFERENCES artifacts(digest),
                    updated REAL NOT NULL,
                    PRIMARY KEY(experiment, name)
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
                    baseline_run_id TEXT,
                    metric TEXT NOT NULL,
                    value REAL,
                    best_before REAL,
                    noise_band REAL,
                    verdict TEXT NOT NULL,
                    reason TEXT,
                    created REAL NOT NULL,
                    metadata_json TEXT
                );
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    spec_id TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    recipe_sha256 TEXT,
                    path TEXT NOT NULL,
                    rows INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created REAL NOT NULL,
                    spec_json TEXT NOT NULL,
                    metadata_json TEXT
                );
                CREATE INDEX IF NOT EXISTS datasets_spec ON datasets(spec_id);
                CREATE TABLE IF NOT EXISTS dataset_specs (
                    spec_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    recipe_sha256 TEXT,
                    created REAL NOT NULL,
                    spec_json TEXT NOT NULL,
                    metadata_json TEXT,
                    FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id)
                );
                CREATE INDEX IF NOT EXISTS dataset_specs_dataset ON dataset_specs(dataset_id);
                """
            )
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),),
            )

    def run_dir(self, experiment: str, run_id: str) -> Path:
        return (
            self.experiments / validate_name(experiment, "experiment") / "runs"
            / validate_name(run_id, "run_id")
        )

    def _row(self, run_id: str) -> sqlite3.Row:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown run_id: {run_id}")
        return row

    @staticmethod
    def _decode_row(row: sqlite3.Row) -> dict:
        out = dict(row)
        for public, column in RunStore.JSON_COLUMNS.items():
            raw = out.pop(column, None)
            out[public] = json.loads(raw) if raw else None
        return out

    def get_run(self, run_id: str, *, related: bool = False) -> dict:
        out = self._decode_row(self._row(run_id))
        if related:
            with self._connect() as conn:
                out["metrics"] = [dict(r) for r in conn.execute(
                    "SELECT * FROM metrics WHERE run_id=? ORDER BY created", (run_id,))]
                out["artifacts"] = [dict(r) for r in conn.execute(
                    """SELECT a.*,ra.role FROM artifacts a JOIN run_artifacts ra
                       ON a.digest=ra.digest WHERE ra.run_id=?""", (run_id,))]
                decision = conn.execute(
                    "SELECT * FROM decisions WHERE run_id=?", (run_id,)).fetchone()
                out["decision"] = dict(decision) if decision else None
        return out

    def exists(self, run_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone() is not None

    def create_run(
        self, *, experiment: str, stage: str, kind: str, model: str | None = None,
        seed: int | None = None, dataset_id: str | None = None,
        config_sha256: str | None = None, command: list[str] | None = None,
        code: dict | None = None, environment: dict | None = None,
        metadata: dict | None = None, run_id: str | None = None,
        status: str = "running", allow_existing: bool = False,
    ) -> str:
        if status not in STATUSES:
            raise ValueError(f"invalid run status: {status}")
        validate_name(experiment, "experiment")
        run_id = run_id or new_run_id()
        now = time.time()
        metadata = {
            "workspace": self.workspace.provenance(),
            **(metadata or {}),
        }
        values = (
            run_id, experiment, stage, kind, status, now, now,
            now if status == "running" else None, now, os.getpid(), socket.gethostname(),
            model, dataset_id, seed, config_sha256,
            _json(command or []), _json(code or {}), _json(environment or {}),
            _json(metadata), SCHEMA_VERSION,
        )
        try:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO runs(
                        run_id,experiment,stage,kind,status,created,updated,started,heartbeat,
                        pid,host,model,dataset_id,seed,config_sha256,command_json,code_json,
                        environment_json,metadata_json,schema_version)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    values,
                )
        except sqlite3.IntegrityError:
            if not allow_existing:
                raise
            existing = self.get_run(run_id)
            if existing["experiment"] != experiment or existing["stage"] != stage:
                raise ValueError(f"run_id {run_id} is reserved for another run")
            self.update_run(
                run_id, status=status, started=existing["started"] or now,
                heartbeat=now, pid=os.getpid(), host=socket.gethostname(), model=model,
                dataset_id=dataset_id, seed=seed, config_sha256=config_sha256,
                command=command or existing.get("command"), code=code or existing.get("code"),
                environment=environment or existing.get("environment"),
                metadata={**(existing.get("metadata") or {}), **metadata},
            )
            return run_id

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "experiment": experiment,
            "stage": stage,
            "kind": kind,
            "created": now,
            "model": model,
            "dataset_id": dataset_id,
            "seed": seed,
            "config_sha256": config_sha256,
            "command": command or [],
            "code": code or {},
            "environment": environment or {},
            "metadata": metadata,
        }
        run_dir = self.run_dir(experiment, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(run_dir / "manifest.json", manifest)
        self._write_state(run_id)
        return run_id

    def update_run(self, run_id: str, **fields: Any) -> dict:
        unknown = set(fields) - self.RUN_COLUMNS - set(self.JSON_COLUMNS)
        if unknown:
            raise ValueError(f"unknown run fields: {sorted(unknown)}")
        if "status" in fields and fields["status"] not in STATUSES:
            raise ValueError(f"invalid run status: {fields['status']}")
        encoded: dict[str, Any] = {}
        for key, value in fields.items():
            column = self.JSON_COLUMNS.get(key, key)
            encoded[column] = _json(value) if key in self.JSON_COLUMNS else value
        encoded["updated"] = time.time()
        assignments = ",".join(f"{key}=?" for key in encoded)
        with self._connect() as conn:
            cur = conn.execute(
                f"UPDATE runs SET {assignments} WHERE run_id=?",
                (*encoded.values(), run_id),
            )
            if cur.rowcount != 1:
                raise KeyError(f"unknown run_id: {run_id}")
        self._write_state(run_id)
        return self.get_run(run_id)

    def transition(self, run_id: str, status: str, **fields: Any) -> dict:
        now = time.time()
        fields.update(status=status, heartbeat=now)
        if status in TERMINAL:
            fields.setdefault("ended", now)
        return self.update_run(run_id, **fields)

    def fail(self, run_id: str, error: BaseException | str, *, exit_code: int | None = None) -> dict:
        message = str(error) if isinstance(error, str) else f"{type(error).__name__}: {error}"
        return self.transition(run_id, "failed", error=message[:4000], exit_code=exit_code)

    def _write_state(self, run_id: str) -> None:
        run = self.get_run(run_id)
        _atomic_json(self.run_dir(run["experiment"], run_id) / "state.json", run)

    def log_event(self, run_id: str, step: int, metrics: dict[str, Any]) -> None:
        run = self.get_run(run_id)
        event = {"step": int(step), "t": time.time(), **metrics}
        path = self.run_dir(run["experiment"], run_id) / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (_json(event) + "\n").encode()
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """UPDATE runs SET latest_step=?,event_count=event_count+1,
                   last_metrics_json=?,heartbeat=?,updated=? WHERE run_id=?""",
                (int(step), _json(metrics), now, now, run_id),
            )
            count = conn.execute(
                "SELECT event_count FROM runs WHERE run_id=?", (run_id,)).fetchone()[0]
        if count % 20 == 0:
            self._write_state(run_id)

    def tail_events(self, run_id: str, limit: int = 20) -> list[dict]:
        run = self.get_run(run_id)
        path = self.run_dir(run["experiment"], run_id) / "events.jsonl"
        if not path.exists() or limit <= 0:
            return []
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            pos = handle.tell()
            data = b""
            while pos > 0 and data.count(b"\n") <= limit:
                size = min(8192, pos)
                pos -= size
                handle.seek(pos)
                data = handle.read(size) + data
        lines = [line for line in data.splitlines() if line][-limit:]
        return [json.loads(line) for line in lines]

    def tail_process_log(self, run_id: str, limit: int = 40) -> list[str]:
        run = self.get_run(run_id)
        path = self.run_dir(run["experiment"], run_id) / "process.log"
        if not path.exists() or limit <= 0:
            return []
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            pos = handle.tell()
            data = b""
            while pos > 0 and data.count(b"\n") <= limit:
                size = min(8192, pos)
                pos -= size
                handle.seek(pos)
                data = handle.read(size) + data
        return [line.decode(errors="replace") for line in data.splitlines()[-limit:]]

    def list_runs(
        self, *, experiment: str | None = None, status: str | None = None,
        stage: str | None = None, limit: int = 50, offset: int = 0,
        detail: bool = False,
    ) -> list[dict]:
        clauses, values = [], []
        for column, value in (("experiment", experiment), ("status", status), ("stage", stage)):
            if value is not None:
                clauses.append(f"{column}=?")
                values.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        projection = "*" if detail else """run_id,experiment,stage,kind,status,
            created,updated,started,ended,model,model_revision,dataset_id,seed,error,
            exit_code,checkpoint_digest,checkpoint_path,latest_step,event_count,
            last_metrics_json,parent_run_id,retry_of"""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {projection} FROM runs {where} "
                "ORDER BY created DESC LIMIT ? OFFSET ?",
                (*values, max(1, min(limit, 1000)), max(0, offset)),
            ).fetchall()
        if detail:
            return [self._decode_row(row) for row in rows]
        out = []
        for row in rows:
            item = dict(row)
            item["last_metrics"] = json.loads(item.pop("last_metrics_json") or "{}")
            out.append(item)
        return out

    def record_metric(
        self, run_id: str, *, name: str, value: float | None, n: int | None = None,
        split: str = "", split_hash: str | None = None, evaluator: str = "",
        evaluator_version: str | None = None, diagnostic: bool = False,
        metadata: dict | None = None,
    ) -> None:
        now = time.time()
        record = {
            "run_id": run_id, "name": name, "value": value, "n": n, "split": split,
            "split_hash": split_hash, "evaluator": evaluator,
            "evaluator_version": evaluator_version, "diagnostic": bool(diagnostic),
            "created": now, "metadata": metadata or {},
        }
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO metrics(run_id,name,value,n,split,split_hash,evaluator,
                   evaluator_version,diagnostic,created,metadata_json)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(run_id,name,split,evaluator,diagnostic) DO UPDATE SET
                   value=excluded.value,n=excluded.n,split_hash=excluded.split_hash,
                   evaluator_version=excluded.evaluator_version,created=excluded.created,
                   metadata_json=excluded.metadata_json""",
                (run_id, name, value, n, split, split_hash, evaluator,
                 evaluator_version, int(diagnostic), now, _json(metadata or {})),
            )
        run = self.get_run(run_id)
        key = hashlib.sha256(_json({
            "name": name, "split": split, "evaluator": evaluator,
            "diagnostic": bool(diagnostic),
        }).encode()).hexdigest()[:16]
        _atomic_json(self.run_dir(run["experiment"], run_id) / "metrics" / f"{key}.json",
                     record)

    def get_metric(self, run_id: str, name: str, split: str = "dev",
                   *, diagnostic: bool = False) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM metrics WHERE run_id=? AND name=? AND split=?
                   AND diagnostic=? ORDER BY created DESC LIMIT 1""",
                (run_id, name, split, int(diagnostic)),
            ).fetchone()
        return dict(row) if row else None

    def best_metric(self, experiment: str, name: str, split: str = "dev",
                    *, exclude_run_id: str | None = None,
                    direction: str = "max") -> dict | None:
        if direction not in {"max", "min"}:
            raise ValueError("direction must be max or min")
        exclusion = "AND m.run_id<>?" if exclude_run_id else ""
        values: tuple = (experiment, name, split, exclude_run_id) if exclude_run_id \
            else (experiment, name, split)
        with self._connect() as conn:
            row = conn.execute(
                """SELECT m.*,r.experiment FROM metrics m JOIN runs r USING(run_id)
                   WHERE r.experiment=? AND m.name=? AND m.split=? AND m.diagnostic=0
                   """ + exclusion + f" ORDER BY m.value {'DESC' if direction == 'max' else 'ASC'} LIMIT 1",
                values,
            ).fetchone()
        return dict(row) if row else None

    def record_decision(
        self, run_id: str, *, metric: str, value: float | None, verdict: str,
        reason: str, best_before: float | None = None, noise_band: float | None = None,
        baseline_run_id: str | None = None, metadata: dict | None = None,
    ) -> None:
        created = time.time()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO decisions(run_id,baseline_run_id,metric,value,best_before,
                   noise_band,verdict,reason,created,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(run_id) DO UPDATE SET baseline_run_id=excluded.baseline_run_id,
                   metric=excluded.metric,value=excluded.value,best_before=excluded.best_before,
                   noise_band=excluded.noise_band,verdict=excluded.verdict,
                   reason=excluded.reason,created=excluded.created,
                   metadata_json=excluded.metadata_json""",
                (run_id, baseline_run_id, metric, value, best_before, noise_band, verdict,
                 reason, created, _json(metadata or {})),
            )
        run = self.get_run(run_id)
        _atomic_json(self.run_dir(run["experiment"], run_id) / "decision.json", {
            "run_id": run_id, "baseline_run_id": baseline_run_id, "metric": metric,
            "value": value, "best_before": best_before, "noise_band": noise_band,
            "verdict": verdict, "reason": reason, "created": created,
            "metadata": metadata or {},
        })

    def list_decisions(self, *, experiment: str | None = None,
                       verdict: str | None = None) -> list[dict]:
        clauses, values = [], []
        if experiment is not None:
            clauses.append("r.experiment=?")
            values.append(experiment)
        if verdict is not None:
            clauses.append("d.verdict=?")
            values.append(verdict)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT d.*,r.experiment,r.stage,r.dataset_id FROM decisions d
                    JOIN runs r USING(run_id) {where} ORDER BY d.created""",
                values,
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            out.append(item)
        return out

    def artifact_temp(self, kind: str, run_id: str) -> Path:
        root = self.artifacts_root / ".tmp"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{kind}-{run_id}-{secrets.token_hex(4)}"
        path.mkdir()
        return path

    def commit_artifact(
        self, temporary: Path, *, kind: str, run_id: str, role: str,
        metadata: dict | None = None,
    ) -> dict:
        temporary = Path(temporary)
        entries = []
        for path in sorted(p for p in temporary.rglob("*") if p.is_file()):
            if path.name == "artifact.json":
                continue
            entries.append({
                "path": str(path.relative_to(temporary)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        identity = {"kind": kind, "files": entries}
        digest = hashlib.sha256(_json(identity).encode()).hexdigest()
        size_bytes = sum(item["size"] for item in entries)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "digest": digest,
            "kind": kind,
            "size_bytes": size_bytes,
            "created": time.time(),
            "producer_run_id": run_id,
            "role": role,
            "files": entries,
            "metadata": metadata or {},
        }
        _atomic_json(temporary / "artifact.json", manifest)
        destination = self.artifacts_root / f"{kind}s" / digest
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, destination)
        rel = self.stored_path(destination)
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO artifacts(digest,kind,path,size_bytes,created,state,metadata_json)
                   VALUES(?,?,?,?,?,'present',?) ON CONFLICT(digest) DO UPDATE SET state='present'""",
                (digest, kind, rel, size_bytes, manifest["created"], _json(metadata or {})),
            )
            conn.execute(
                """INSERT INTO run_artifacts(run_id,digest,role) VALUES(?,?,?)
                   ON CONFLICT(run_id,role) DO UPDATE SET digest=excluded.digest""",
                (run_id, digest, role),
            )
            if kind == "checkpoint":
                conn.execute(
                    """UPDATE runs SET checkpoint_digest=?,checkpoint_path=?,updated=?
                       WHERE run_id=?""",
                    (digest, rel, time.time(), run_id),
                )
        self._write_state(run_id)
        return {**manifest, "path": rel}

    def set_pointer(self, experiment: str, name: str, run_id: str, digest: str) -> None:
        validate_name(experiment, "experiment")
        updated = time.time()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO pointers(experiment,name,run_id,digest,updated) VALUES(?,?,?,?,?)
                   ON CONFLICT(experiment,name) DO UPDATE SET run_id=excluded.run_id,
                   digest=excluded.digest,updated=excluded.updated""",
                (experiment, name, run_id, digest, updated),
            )
        key = hashlib.sha256(name.encode()).hexdigest()[:16]
        _atomic_json(self.root / "pointers" / experiment / f"{key}.json", {
            "experiment": experiment, "name": name, "run_id": run_id,
            "digest": digest, "updated": updated,
        })

    def resolve_checkpoint(self, run_id: str) -> Path:
        run = self.get_run(run_id)
        if not run.get("checkpoint_path"):
            raise KeyError(f"run {run_id} has no checkpoint")
        path = self.resolve_stored_path(run["checkpoint_path"])
        if not path.is_dir():
            raise FileNotFoundError(f"checkpoint artifact missing for {run_id}: {path}")
        return path

    def resolve_pointer(self, experiment: str, name: str = "latest") -> tuple[str, Path]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT run_id FROM pointers WHERE experiment=? AND name=?",
                (experiment, name),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown pointer: {experiment}/{name}")
        return row["run_id"], self.resolve_checkpoint(row["run_id"])

    def resolve_model_ref(self, ref: str) -> str:
        if ref.startswith("run:"):
            return str(self.resolve_checkpoint(ref.split(":", 1)[1]))
        if ref.startswith("latest:"):
            return str(self.resolve_pointer(ref.split(":", 1)[1], "latest")[1])
        if ref.startswith("best:"):
            _, experiment, metric = ref.split(":", 2)
            return str(self.resolve_pointer(experiment, f"best:{metric}")[1])
        path = Path(ref).expanduser()
        local = path if path.is_absolute() else self.workspace.resolve_input(path)
        return str(local) if local.exists() else ref

    def register_dataset(
        self, *, dataset_id: str, spec_id: str, content_sha256: str,
        recipe_sha256: str | None, path: Path, rows: int, size_bytes: int,
        spec: dict, metadata: dict | None = None,
    ) -> None:
        stored_path = self.stored_path(path)
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO datasets(dataset_id,spec_id,content_sha256,recipe_sha256,
                   path,rows,size_bytes,created,spec_json,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(dataset_id) DO NOTHING""",
                (dataset_id, spec_id, content_sha256, recipe_sha256, stored_path, rows,
                 size_bytes, time.time(), _json(spec), _json(metadata or {})),
            )
            conn.execute(
                """INSERT INTO dataset_specs(spec_id,dataset_id,recipe_sha256,created,
                   spec_json,metadata_json) VALUES(?,?,?,?,?,?)
                   ON CONFLICT(spec_id) DO UPDATE SET dataset_id=excluded.dataset_id,
                   recipe_sha256=excluded.recipe_sha256,spec_json=excluded.spec_json,
                   metadata_json=excluded.metadata_json""",
                (spec_id, dataset_id, recipe_sha256, time.time(), _json(spec),
                 _json(metadata or {})),
            )

    def integrity(self) -> dict:
        with self._connect() as conn:
            sqlite_ok = conn.execute("PRAGMA integrity_check").fetchone()[0]
            counts = {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("runs", "metrics", "artifacts", "decisions", "datasets",
                              "dataset_specs")
            }
            missing = []
            for row in conn.execute("SELECT digest,path FROM artifacts WHERE state='present'"):
                if not self.resolve_stored_path(row["path"]).exists():
                    missing.append(row["digest"])
        return {"schema_version": SCHEMA_VERSION, "sqlite": sqlite_ok,
                "counts": counts, "missing_artifacts": missing,
                "ok": sqlite_ok == "ok" and not missing}

    def reindex(self, *, reset: bool = False) -> dict:
        """Rebuild the SQLite query index from immutable/local record files."""
        if reset:
            for path in (self.db_path, self.db_path.with_name(self.db_path.name + "-wal"),
                         self.db_path.with_name(self.db_path.name + "-shm")):
                if path.exists():
                    path.unlink()
            self._init_schema()
        counts = {"runs": 0, "metrics": 0, "artifacts": 0,
                  "decisions": 0, "datasets": 0, "dataset_specs": 0, "pointers": 0}
        for state_path in sorted(self.experiments.glob("*/runs/*/state.json")):
            state = json.loads(state_path.read_text())
            values = {
                "command_json": _json(state.get("command") or []),
                "code_json": _json(state.get("code") or {}),
                "environment_json": _json(state.get("environment") or {}),
                "metadata_json": _json(state.get("metadata") or {}),
                "last_metrics_json": _json(state.get("last_metrics") or {}),
            }
            scalar_columns = [
                "run_id", "experiment", "stage", "kind", "status", "created", "updated",
                "started", "ended", "heartbeat", "pid", "host", "model", "model_revision",
                "dataset_id", "seed", "config_sha256", "error", "exit_code",
                "checkpoint_digest", "checkpoint_path", "latest_step", "event_count",
                "parent_run_id", "retry_of", "schema_version",
            ]
            payload = {column: state.get(column) for column in scalar_columns}
            payload.update(values)
            columns = list(payload)
            updates = ",".join(
                f"{column}=excluded.{column}" for column in columns if column != "run_id")
            with self._connect() as conn:
                conn.execute(
                    f"INSERT INTO runs({','.join(columns)}) "
                    f"VALUES({','.join('?' for _ in columns)}) "
                    f"ON CONFLICT(run_id) DO UPDATE SET {updates}",
                    [payload[column] for column in columns],
                )
            counts["runs"] += 1
            for metric_path in sorted(state_path.parent.glob("metrics/*.json")):
                metric = json.loads(metric_path.read_text())
                with self._connect() as conn:
                    conn.execute(
                        """INSERT OR REPLACE INTO metrics(run_id,name,value,n,split,split_hash,
                           evaluator,evaluator_version,diagnostic,created,metadata_json)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (metric["run_id"], metric["name"], metric.get("value"), metric.get("n"),
                         metric.get("split", ""), metric.get("split_hash"),
                         metric.get("evaluator", ""), metric.get("evaluator_version"),
                         int(metric.get("diagnostic", False)), metric["created"],
                         _json(metric.get("metadata") or {})),
                    )
                counts["metrics"] += 1
            decision_path = state_path.parent / "decision.json"
            if decision_path.is_file():
                decision = json.loads(decision_path.read_text())
                with self._connect() as conn:
                    conn.execute(
                        """INSERT OR REPLACE INTO decisions(run_id,baseline_run_id,metric,value,
                           best_before,noise_band,verdict,reason,created,metadata_json)
                           VALUES(?,?,?,?,?,?,?,?,?,?)""",
                        (decision["run_id"], decision.get("baseline_run_id"), decision["metric"],
                         decision.get("value"), decision.get("best_before"),
                         decision.get("noise_band"), decision["verdict"],
                         decision.get("reason"), decision["created"],
                         _json(decision.get("metadata") or {})),
                    )
                counts["decisions"] += 1

        for artifact_path in sorted(self.artifacts_root.glob("*/*/artifact.json")):
            artifact = json.loads(artifact_path.read_text())
            directory = artifact_path.parent
            rel = self.stored_path(directory)
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO artifacts(digest,kind,path,size_bytes,created,
                       state,pinned,metadata_json) VALUES(?,?,?,?,?,'present',0,?)
                       ON CONFLICT(digest) DO UPDATE SET path=excluded.path,
                       size_bytes=excluded.size_bytes,state='present',
                       metadata_json=excluded.metadata_json""",
                    (artifact["digest"], artifact["kind"], rel, artifact["size_bytes"],
                     artifact["created"], _json(artifact.get("metadata") or {})),
                )
                producer = artifact.get("producer_run_id")
                if producer and conn.execute(
                        "SELECT 1 FROM runs WHERE run_id=?", (producer,)).fetchone():
                    conn.execute(
                        "INSERT OR REPLACE INTO run_artifacts(run_id,digest,role) VALUES(?,?,?)",
                        (producer, artifact["digest"], artifact.get("role", artifact["kind"])),
                    )
            counts["artifacts"] += 1

        # An artifact manifest names its first producer, but content-addressed weights
        # may be shared by later runs.  Every run's state is therefore the authority for
        # reconstructing its checkpoint edge after a full index reset.
        with self._connect() as conn:
            for row in conn.execute(
                    """SELECT run_id,checkpoint_digest FROM runs
                       WHERE checkpoint_digest IS NOT NULL"""):
                if conn.execute(
                        "SELECT 1 FROM artifacts WHERE digest=?",
                        (row["checkpoint_digest"],)).fetchone():
                    conn.execute(
                        """INSERT OR REPLACE INTO run_artifacts(run_id,digest,role)
                           VALUES(?,?,'checkpoint')""",
                        (row["run_id"], row["checkpoint_digest"]),
                    )

        for pointer_path in sorted((self.root / "pointers").glob("*/*.json")):
            pointer = json.loads(pointer_path.read_text())
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO pointers(experiment,name,run_id,digest,updated) "
                    "VALUES(?,?,?,?,?)",
                    (pointer["experiment"], pointer["name"], pointer["run_id"],
                     pointer["digest"], pointer["updated"]),
                )
            counts["pointers"] += 1

        for provenance_path in sorted(self.experiments.glob(
                "*/data/objects/*/provenance.json")):
            provenance = json.loads(provenance_path.read_text())
            self.register_dataset(
                dataset_id=provenance["dataset_id"], spec_id=provenance["spec_id"],
                content_sha256=provenance["content_sha256"],
                recipe_sha256=provenance.get("recipe_sha256"), path=provenance_path.parent,
                rows=provenance["n"], size_bytes=provenance["size_bytes"],
                spec=provenance["spec"], metadata={"stats": provenance.get("stats", {})},
            )
            counts["datasets"] += 1
        for pointer_path in sorted(self.experiments.glob("*/data/specs/*.json")):
            pointer = json.loads(pointer_path.read_text())
            provenance = pointer.get("provenance")
            if not provenance:
                continue
            object_path = pointer_path.parents[1] / "objects" / provenance["dataset_id"]
            self.register_dataset(
                dataset_id=provenance["dataset_id"], spec_id=provenance["spec_id"],
                content_sha256=provenance["content_sha256"],
                recipe_sha256=provenance.get("recipe_sha256"), path=object_path,
                rows=provenance["n"], size_bytes=provenance["size_bytes"],
                spec=provenance["spec"], metadata={"stats": provenance.get("stats", {})},
            )
            counts["dataset_specs"] += 1
        return {"reset": reset, "indexed": counts, "integrity": self.integrity()}

    def gc_candidates(self, keep_last: int = 2) -> list[dict]:
        """Unpinned checkpoint artifacts not retained by pointers or latest N runs."""
        keep: set[str] = set()
        with self._connect() as conn:
            keep.update(r[0] for r in conn.execute("SELECT digest FROM pointers"))
            experiments = [r[0] for r in conn.execute("SELECT DISTINCT experiment FROM runs")]
            for experiment in experiments:
                keep.update(r[0] for r in conn.execute(
                    """SELECT checkpoint_digest FROM runs WHERE experiment=?
                       AND checkpoint_digest IS NOT NULL ORDER BY created DESC LIMIT ?""",
                    (experiment, max(0, keep_last)),
                ))
            rows = conn.execute(
                """SELECT * FROM artifacts WHERE kind='checkpoint' AND state='present'
                   AND pinned=0 ORDER BY created"""
            ).fetchall()
        return [dict(row) for row in rows if row["digest"] not in keep]

    def gc(self, *, keep_last: int = 2, apply: bool = False) -> dict:
        candidates = self.gc_candidates(keep_last)
        reclaimed = 0
        if apply:
            for item in candidates:
                path = self.resolve_stored_path(item["path"])
                if path.is_dir() and self.artifacts_root in path.parents:
                    shutil.rmtree(path)
                reclaimed += item["size_bytes"]
                with self._connect() as conn:
                    conn.execute(
                        "UPDATE artifacts SET state='deleted' WHERE digest=?",
                        (item["digest"],),
                    )
        return {"apply": apply, "keep_last": keep_last,
                "candidates": [{"digest": r["digest"], "path": r["path"],
                                "size_bytes": r["size_bytes"]} for r in candidates],
                "reclaimed_bytes": reclaimed}
