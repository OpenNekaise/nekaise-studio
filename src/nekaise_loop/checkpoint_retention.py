"""Explicit orchestrator decisions about checkpoint bytes, with immutable receipts.

Metadata and training artifacts remain authoritative after bytes have been retired.
The source lock excludes training; retiring bytes also excludes external readers.
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
from .checkpoint_lineage import current_campaign_id, latest_completed_snapshot
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
def retention_lock(workspace, *, exclusive=True):
    # Serialize receipt writers even when their byte-access lock can be shared
    # with inference readers. The source lock remains the outer execution guard.
    with (workspace/'checkpoint-retention-writer.lock').open('a+') as writer:
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (workspace/'checkpoint-retention.lock').open('a+') as f:
            fcntl.flock(f, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
            yield


def checkpoint_path(workspace, relative):
    path = workspace/relative
    if (Path(relative).is_absolute() or '..' in Path(relative).parts or
            not path.is_relative_to(workspace/'runs') or path.name != 'checkpoint' or
            path.resolve() != path.absolute()):
        raise ValueError('Retention requires a real checkpoint directory inside workspace/runs')
    return path


def recorded_checkpoint_path(workspace, reference):
    """Map a workspace-root alias without admitting symlinks inside runs.

    Historical artifacts retain their original absolute spelling. Inventory and
    every dependency lookup must use the same identity, while decisions still
    accept only canonical relative paths through checkpoint_path.
    """
    if not isinstance(reference, str):
        return None
    path = Path(reference)
    if not path.is_absolute() or path.name != 'checkpoint' or '..' in path.parts:
        return None
    if path.is_relative_to(workspace):
        return checkpoint_path(workspace, path.relative_to(workspace))
    for ancestor in path.parents:
        if ancestor.resolve() == workspace:
            return checkpoint_path(workspace, path.relative_to(ancestor))
    return None


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
        path = recorded_checkpoint_path(workspace, result['checkpoint'])
        tips[row['campaign_id']] = result['checkpoint']
        if result.get('trained') is False or path is None:
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
            'protected_resumable': [], 'protected_weights': [], 'historical_references': [], 'retention': prior,
            'parent': result['manifest'].get('parent'), 'tokens': result['manifest'].get('tokens')}
    def protect(path, reason, level='protected_resumable'):
        path = recorded_checkpoint_path(workspace, path)
        path = str(path) if path is not None else None
        if path in records:
            records[path][level].append(reason)
    campaigns = store.query('SELECT id,status,config,parent_campaign_id,operator_hold,created_at FROM campaigns ORDER BY created_at,id')
    # Execution dependencies are not a keep-forever rule for every historical
    # campaign. Retirement still requires the orchestrator's explicit decision.
    active = {'running', 'queued', 'pausing', 'stopping', 'waiting', 'recovering'}
    operational = {c['id'] for c in campaigns if c['status'] in active}
    operational.update(r['campaign_id'] for r in store.query(
        'SELECT DISTINCT campaign_id FROM actions WHERE handled_at IS NULL'))
    operational.update(r['campaign_id'] for r in store.query(
        "SELECT DISTINCT campaign_id FROM recoveries WHERE status IN ('pending','running','waiting','decided')"))
    superseded = {c['parent_campaign_id'] for c in campaigns if c['parent_campaign_id']}
    operational.update(c['id'] for c in campaigns if c['operator_hold'] == 'pause' and c['id'] not in superseded)
    by_id = {c['id']: c for c in campaigns}
    current = by_id.get(current_campaign_id(store))
    if current:
        operational.add(current['id'])  # Keep the current paused/stopped run resumable too.
    # Model chat reads the latest fully completed round in the current lineage;
    # an unfinished round may already have a newer trained checkpoint.
    snapshot = latest_completed_snapshot(service, current['id']) if current else None
    if snapshot and snapshot[0]['config'].get('student_format') == 'chat_template':
        protect(snapshot[2]['checkpoint'], 'Current Model chat snapshot', 'protected_weights')
    for c in campaigns:
        config = json.loads(c['config'])
        # Preserve lineage references as evidence, without pretending they are
        # current resumption dependencies after a successor has taken over.
        for path, reason in ((tips.get(c['id']), f'Historical campaign tip {c["id"]}'),
                             (config.get('student_model'), f'Configured origin of {c["id"]}')):
            protect(path, reason, 'historical_references')
        if c['id'] in operational:
            if c['id'] in tips:
                protect(tips[c['id']], f'Current resumption checkpoint of {c["id"]}')
            else:
                protect(config.get('student_model'), f'Unstarted current campaign input {c["id"]}')
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
    for monitor in ('monitor', 'monitor-v2'):
        leases = service.settings.root.parent/f'nekaise-bench/workspace/{monitor}/protected-checkpoints.json'
        if leases.exists():
            for path in json.loads(leases.read_text()).get('paths', []):
                protect(path, 'Pending or running independent reader', 'protected_weights')
    latest = (tips.get(current['id']) or json.loads(current['config']).get('student_model')) if current else None
    return {'storage': storage_status(workspace, latest), 'checkpoints': list(records.values()),
            'current_campaign_id': current['id'] if current else None,
            'operational_campaign_ids': sorted(operational),
            'scope': 'keep=full state, weights=inference only, summary=records only; original manifests, datasets, lessons and lineage remain. No score-based automatic deletion.'}


def apply(service, recovery_id, decisions):
    if not decisions:
        return []
    workspace = service.settings.workspace
    choices = [CheckpointRetention.model_validate(x) for x in decisions]
    if len({x.path for x in choices}) != len(choices):
        raise ValueError('Duplicate checkpoint retention decisions')
    # The caller's exclusive source lock serializes Studio metadata writers.
    # Keeping bytes only replaces atomic receipts and can coexist with inference
    # readers. Any potentially destructive batch still excludes every reader.
    with retention_lock(workspace, exclusive=any(x.disposition != 'keep' for x in choices)):
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
