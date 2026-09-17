"""Explicit orchestrator decisions about checkpoint bytes, with immutable receipts.

Metadata and training artifacts remain authoritative after bytes have been retired.
The source lock excludes training; the retention lock also excludes external readers.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .artifacts import atomic_write, canonical
from .storage import now


class CheckpointRetention(BaseModel):
    model_config = ConfigDict(extra='forbid')
    path: str
    manifest_sha256: str = Field(pattern='^[a-f0-9]{64}$')
    disposition: Literal['keep', 'weights', 'summary']
    summary: str = Field(min_length=1, max_length=6000)
    reason: str = Field(min_length=1, max_length=2000)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def receipt(root, manifest=None):
    path = Path(root)/'.retention'
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    manifest = Path(root)/'checkpoint.json' if manifest is None else manifest
    if value.get('schema_version') != 1 or value.get('manifest_sha256') != sha(manifest):
        raise ValueError('Checkpoint retention receipt does not match immutable manifest')
    return value


@contextmanager
def retention_lock(workspace):
    with (workspace/'checkpoint-retention.lock').open('a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def checkpoint_path(workspace, relative):
    path = workspace/relative
    if (Path(relative).is_absolute() or '..' in Path(relative).parts or
            not path.is_relative_to(workspace/'runs') or path.name != 'checkpoint' or
            path.resolve() != path.absolute()):
        raise ValueError('Retention requires a real checkpoint directory inside workspace/runs')
    return path


def storage_status(workspace, checkpoint=None):
    free = shutil.disk_usage(workspace).free
    # Room for a full checkpoint, temporary output, and database/log operations.
    weight_bytes = 0
    if checkpoint and Path(checkpoint).is_dir():
        weight_bytes = sum(p.stat().st_size for p in Path(checkpoint).glob('*.safetensors'))
    required = max(5 * 1024**3, weight_bytes * 6 + 2 * 1024**3)
    return {'free_bytes': free, 'required_bytes': required, 'pressure': free < required}


def inventory(service):
    workspace, store = service.settings.workspace, service.store
    records = {}
    rows = store.query("SELECT s.artifact,r.id AS round_id,r.campaign_id,r.number,r.status FROM stage_runs s JOIN rounds r ON r.id=s.round_id WHERE s.stage='train' AND s.status='complete' ORDER BY s.id")
    tips = {}
    for row in rows:
        result = service.artifacts.get(row['artifact'])
        path = Path(result['checkpoint'])
        tips[row['campaign_id']] = str(path)
        if result.get('trained') is False or not path.is_relative_to(workspace/'runs'):
            continue
        relative = path.relative_to(workspace).as_posix()
        checkpoint_path(workspace, relative)
        manifest = path/'checkpoint.json'
        if json.loads(manifest.read_text()) != result['manifest']:
            raise ValueError('Checkpoint manifest differs from completed training artifact')
        prior = receipt(path)
        files = result['manifest']['files']
        sizes = {name: (path/name).stat().st_size for name in files if (path/name).is_file()}
        records[str(path)] = {**row, 'path': relative, 'manifest_sha256': sha(manifest),
            'bytes': sum(sizes.values()), 'optimizer_bytes': sizes.get('training_state.pt', 0),
            'weight_bytes': sum(size for name, size in sizes.items() if name.endswith('.safetensors') or name.startswith('pytorch_model')),
            'protected_resumable': [], 'protected_weights': [], 'retention': prior,
            'parent': result['manifest'].get('parent'), 'tokens': result['manifest'].get('tokens')}
    def protect(path, reason, level='protected_resumable'):
        if path in records:
            records[path][level].append(reason)
    for cid, path in tips.items():
        protect(path, f'Latest completed training checkpoint of {cid}')
    campaigns = store.query('SELECT id,status,config FROM campaigns')
    for c in campaigns:
        config = json.loads(c['config'])
        protect(config.get('student_model'), f'Configured starting checkpoint of {c["id"]}')
        if c['status'] not in {'complete', 'stopped', 'failed', 'interrupted'}:
            pending = store.query("SELECT id,model_before,checkpoint FROM rounds WHERE campaign_id=? AND status!='complete'", (c['id'],))
            for r in pending:
                protect(r['model_before'], f'Unfinished round input {r["id"]}')
                protect(r['checkpoint'], f'Unfinished round output {r["id"]}')
                # Frozen diagnostic references remain runnable through the next review.
                for stage in store.query("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='select' AND status='complete'", (r['id'],)):
                    data = service.artifacts.get(stage['artifact'])
                    def references(value):
                        if isinstance(value, dict):
                            for v in value.values(): references(v)
                        elif isinstance(value, list):
                            for v in value: references(v)
                        elif isinstance(value, str): protect(value, f'Frozen diagnostic in {r["id"]}', 'protected_weights')
                    references(data)
    # Operational leases contain paths only, never evaluation questions or scores.
    leases = service.settings.root.parent/'nekaise-bench/workspace/monitor/protected-checkpoints.json'
    if leases.exists():
        for path in json.loads(leases.read_text()).get('paths', []):
            protect(path, 'Pending or running independent reader', 'protected_weights')
    latest = next((json.loads(c['config']).get('student_model') for c in reversed(campaigns)), None)
    return {'storage': storage_status(workspace, latest), 'checkpoints': list(records.values()),
            'scope': 'keep=full state, weights=inference only, summary=records only; original manifests, datasets, lessons and lineage remain. No score-based automatic deletion.'}


def apply(service, recovery_id, decisions):
    if not decisions:
        return []
    workspace = service.settings.workspace
    choices = [CheckpointRetention.model_validate(x) for x in decisions]
    if len({x.path for x in choices}) != len(choices):
        raise ValueError('Duplicate checkpoint retention decisions')
    with retention_lock(workspace):
        current = {x['path']: x for x in inventory(service)['checkpoints']}
        plans = []
        for choice in choices:
            item = current.get(choice.path)
            if not item or item['manifest_sha256'] != choice.manifest_sha256:
                raise ValueError('Checkpoint retention identity changed')
            root = checkpoint_path(workspace, choice.path)
            old = receipt(root)
            if old and old['recovery_id'] == recovery_id:
                if old['decision'] != choice.model_dump():
                    raise ValueError('Checkpoint retention decision changed during retry')
                plans.append((root, old));continue
            if choice.disposition != 'keep' and item['protected_resumable']:
                raise ValueError(f'Checkpoint is needed for resumption: {choice.path}')
            if choice.disposition == 'summary' and item['protected_weights']:
                raise ValueError(f'Checkpoint is needed for inference: {choice.path}')
            if old and old['disposition'] == 'summary' and choice.disposition != 'summary':
                raise ValueError('Deleted weights cannot be restored by a retention label')
            if old and old['disposition'] == 'weights' and choice.disposition == 'keep':
                raise ValueError('Deleted optimizer cannot be restored by a retention label')
            manifest = json.loads((root/'checkpoint.json').read_text())
            names = [n for n in manifest['files'] if choice.disposition != 'keep' and
                     (n == 'training_state.pt' or (choice.disposition == 'summary' and
                     (n.endswith('.safetensors') or n.startswith('pytorch_model'))))]
            deleted = dict(old.get('deleted', {})) if old else {}
            sizes = {}
            for name in names:
                p = root/name
                if Path(name).name != name or p.is_symlink():
                    raise ValueError('Unsafe checkpoint file path')
                if p.exists():
                    if sha(p) != manifest['files'][name]:
                        raise ValueError(f'Checkpoint integrity failed before retirement: {name}')
                    sizes[name] = p.stat().st_size
                elif deleted.get(name) != manifest['files'][name]:
                    raise ValueError('Missing checkpoint file has no retirement receipt')
                deleted[name] = manifest['files'][name]
            record = {'schema_version': 1, 'manifest_sha256': choice.manifest_sha256,
                'recovery_id': recovery_id, 'decision': choice.model_dump(), 'disposition': choice.disposition,
                'summary': choice.summary, 'reason': choice.reason, 'deleted': deleted,
                'remove': names, 'bytes_released': sum(sizes.values()), 'status': 'planned', 'created_at': now()}
            plans.append((root, record))
        # All choices validated before any unlink. Receipts survive a crash mid-batch.
        result = []
        for root, record in plans:
            journal = workspace/'recoveries'/str(recovery_id)/'checkpoint-retention'/f'{hashlib.sha256(str(root).encode()).hexdigest()}.json'
            if record['status'] != 'complete':
                atomic_write(journal, canonical(record))
                atomic_write(root/'.retention', canonical(record))
                for name in record['remove']:
                    p = root/name
                    if p.exists():
                        if p.is_symlink() or sha(p) != record['deleted'][name]:
                            raise ValueError('Checkpoint changed while retiring bytes')
                        p.unlink()
                directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
                try: os.fsync(directory)
                finally: os.close(directory)
                record.update(status='complete', completed_at=now())
                atomic_write(root/'.retention', canonical(record))
                atomic_write(journal, canonical(record))
            result.append({'path': root.relative_to(workspace).as_posix(), **record})
        return result
