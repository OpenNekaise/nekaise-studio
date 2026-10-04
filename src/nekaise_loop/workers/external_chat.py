"""Own one local GPU chat server while training is explicitly paused."""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import sqlite3
import subprocess
import sys
import time

import httpx

from nekaise_loop.config import Settings
from nekaise_loop.external_chat import read_config
from nekaise_loop.ownership import source_lock
from nekaise_loop.processes import process_start, stop_child
from nekaise_loop.storage import now


def training_paused(workspace):
    """Read only. Queued execution and recorded live model children take precedence."""
    with sqlite3.connect(f'file:{workspace / "loop.sqlite3"}?mode=ro', uri=True) as db:
        latest = db.execute("SELECT status,operator_hold FROM campaigns ORDER BY created_at DESC LIMIT 1").fetchone()
        if not latest or latest != ('paused', 'pause'):
            return False
        if db.execute("SELECT 1 FROM campaigns WHERE status IN ('running','queued','pausing','stopping','recovering') LIMIT 1").fetchone():
            return False
        if db.execute("SELECT 1 FROM actions a JOIN campaigns c ON c.id=a.campaign_id WHERE a.handled_at IS NULL AND a.kind IN ('start','resume') AND c.operator_hold IS NULL LIMIT 1").fetchone():
            return False
        for table in ('trainer_sessions', 'stage_runs'):
            for pid, start in db.execute(f'SELECT process_pid,process_start FROM {table} WHERE process_pid IS NOT NULL'):
                if start and process_start(pid) == start:
                    return False
    return True


def write_runtime(workspace, value):
    path = workspace / 'model-chat-runtime.json'
    temporary = path.with_suffix('.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as handle:
        json.dump(value, handle)
    os.replace(temporary, path)


def server_command(config, key_path):
    return [str(config.server_binary), '--model', str(config.model_path),
            '--alias', config.id, '--host', '127.0.0.1', '--port', str(config.port),
            '--api-key-file', str(key_path), '--ctx-size', str(config.context_tokens),
            '--n-gpu-layers', '99', '--fit', 'off', '--parallel', '1',
            '--threads', '8', '--flash-attn', 'on', '--jinja', '--reasoning', 'off',
            '--no-context-shift', '--cache-ram', '0', '--no-webui', '--log-disable']


def run(settings):
    config = read_config(settings.workspace)
    if config is None:
        raise ValueError('Configure workspace/model-chat.json before starting GPU chat')
    # The existing worker lock is the GPU lease. Training, profiling and chat
    # already enter through workers, so there is no second scheduler or queue.
    # Inherit this descriptor into llama-server: even an orphan holds exclusion.
    with source_lock(), (settings.workspace / 'worker.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not training_paused(settings.workspace):
            raise ValueError('GPU chat requires an explicitly paused, quiescent training campaign')
        identity = {'pid': os.getpid(), 'start': process_start(os.getpid()), 'kind': 'external_model_chat'}
        lease.seek(0); lease.truncate(); json.dump(identity, lease); lease.flush()
        state = {'model_id': config.id, 'status': 'loading', 'worker_pid': identity['pid'],
                 'worker_start': identity['start'], 'started_at': now()}
        write_runtime(settings.workspace, state)
        closing = [False]
        signal.signal(signal.SIGTERM, lambda *_: closing.__setitem__(0, True))
        signal.signal(signal.SIGINT, lambda *_: closing.__setitem__(0, True))
        proc = None
        key_path = settings.workspace / 'model-chat-server.key'
        try:
            with config.model_path.open('rb') as model:
                actual = hashlib.file_digest(model, 'sha256').hexdigest()
            if actual != config.sha256:
                raise ValueError('Chat model SHA-256 does not match the configured artifact')
            if closing[0] or not training_paused(settings.workspace):
                return
            key = secrets.token_urlsafe(32)
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, 'w') as handle:
                handle.write(key + '\n')
            env = {**os.environ, 'PYTHONPATH': str(settings.root / 'src'), 'CUDA_VISIBLE_DEVICES': '0'}
            # The exec wrapper installs parent-death protection before exec, with
            # a parent PID check to close the death-before-prctl race.
            proc = subprocess.Popen([sys.executable, '-m', 'nekaise_loop.workers.external_chat', '--child', str(os.getpid()),
                                     *server_command(config, key_path)], cwd=settings.root, env=env,
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, start_new_session=True,
                                    pass_fds=(lease.fileno(),))
            state.update(server_pid=proc.pid, server_start=process_start(proc.pid), api_key=key)
            write_runtime(settings.workspace, state)
            deadline = time.monotonic() + 240
            with httpx.Client(base_url=config.url, headers={'Authorization': 'Bearer ' + key},
                              timeout=2, trust_env=False) as client:
                while not closing[0]:
                    if not training_paused(settings.workspace) or read_config(settings.workspace) != config:
                        state['reason'] = 'Training requested execution or the selected model changed'
                        break
                    if proc.poll() is not None:
                        raise RuntimeError(f'Local model server exited with status {proc.returncode}')
                    healthy = False
                    try:
                        health = client.get('/health')
                        if health.status_code == 200:
                            models = client.get('/v1/models')
                            healthy = models.status_code == 200 and any(
                                model.get('id') == config.id for model in models.json().get('data', []))
                    except (httpx.HTTPError, ValueError):
                        pass
                    if healthy:
                        deadline = time.monotonic() + 30
                        if state['status'] != 'ready':
                            state.update(status='ready', ready_at=now())
                            write_runtime(settings.workspace, state)
                    elif time.monotonic() > deadline:
                        raise RuntimeError('Local model server did not become healthy')
                    time.sleep(0.5)
        except Exception:
            state['reason'] = 'Local model worker failed; inspect the service status'
            raise
        finally:
            state['status'] = 'stopping'
            write_runtime(settings.workspace, state)
            if proc is not None:
                stop_child(proc)
            state.update(status='stopped', stopped_at=now())
            state.pop('api_key', None)
            write_runtime(settings.workspace, state)
            key_path.unlink(missing_ok=True)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == '--child':
        parent = int(sys.argv[2])
        if ctypes.CDLL(None).prctl(1, signal.SIGTERM) != 0:
            raise RuntimeError('Cannot establish model process ownership')
        if os.getppid() != parent:
            return
        os.execv(sys.argv[3], sys.argv[3:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path)
    args = parser.parse_args()
    run(Settings(args.workspace))


if __name__ == '__main__':
    main()
