"""Read-only model chat, isolated from the teaching ledger and GPU worker."""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import signal
from pathlib import Path
from typing import Literal

import anyio
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .artifacts import digest
from . import ownership

MAX_PROMPT_TOKENS = 2048
MAX_NEW_TOKENS = 256
TIMEOUT_SECONDS = 180


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[ChatMessage] = Field(min_length=1, max_length=31)

    @model_validator(mode="after")
    def conversation(self):
        if len(self.messages) % 2 != 1 or any(
            m.role != ("user" if i % 2 == 0 else "assistant")
            for i, m in enumerate(self.messages)
        ):
            raise ValueError("Conversation must alternate user/assistant and end with a user message")
        if any(not m.content.strip() for m in self.messages):
            raise ValueError("Messages cannot be blank")
        if sum(len(m.content) for m in self.messages) > 12000:
            raise ValueError("Conversation is too long; start a new chat")
        return self


def latest_snapshot(service):
    """Follow the current lineage, ignoring drafts and unfinished round outputs."""
    campaign = service.store.one("""SELECT id FROM campaigns
        WHERE status IN ('running','queued','pausing','stopping','waiting','recovering')
        ORDER BY created_at DESC LIMIT 1""")
    if not campaign:
        campaign = service.store.one("""SELECT c.id FROM campaigns c
            WHERE c.status!='ready' AND (c.parent_campaign_id IS NOT NULL OR
                EXISTS (SELECT 1 FROM rounds r WHERE r.campaign_id=c.id))
            ORDER BY c.created_at DESC LIMIT 1""")
    seen = set()
    while campaign and campaign['id'] not in seen:
        cid = campaign['id']
        seen.add(cid)
        row = service.store.one("""SELECT r.id,r.number,r.updated_at,s.artifact
            FROM rounds r JOIN stage_runs s ON s.round_id=r.id
            WHERE r.campaign_id=? AND r.status='complete' AND s.stage='train'
                AND s.status='complete' ORDER BY r.number DESC,s.attempt DESC LIMIT 1""", (cid,))
        config = service.store.campaign(cid)
        if row:
            result = service.artifacts.get(row['artifact'])
            path = Path(result['checkpoint'])
            # Do not quietly substitute older weights when the latest is missing.
            if not path.is_dir() or not any(path.glob('*.safetensors')):
                raise ValueError("The latest completed model weights are unavailable")
            if config['config'].get('student_format', 'raw_text') != 'chat_template':
                raise ValueError("The current model has no native chat interface")
            identity = digest({'checkpoint': str(path), 'manifest': result['manifest']})
            return {'id': identity, 'campaign_id': cid, 'run_name': config['name'],
                    'round_id': row['id'], 'round_number': row['number'],
                    'completed_at': row['updated_at'], 'device': 'cpu',
                    'max_prompt_tokens': MAX_PROMPT_TOKENS, 'max_new_tokens': MAX_NEW_TOKENS}, result
        parent = config.get('parent_campaign_id')
        campaign = {'id': parent} if parent else None
    raise ValueError("A completed chat-model iteration is not available yet")


def status(service):
    try:
        snapshot, _ = latest_snapshot(service)
        return {'available': True, 'model': snapshot}
    except (ValueError, OSError, KeyError) as exc:
        return {'available': False, 'message': str(exc)}


def worker_command(service):
    return [service.settings.model_python, '-m', 'nekaise_loop.workers.chat']


async def stop_process(proc):
    if proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(proc.wait(), 3)
    except asyncio.TimeoutError:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await proc.wait()


async def stream_chat(service, body):
    """Own exactly one bounded CPU child, including disconnect and timeout cleanup.

    No prompts, answers or usage are written to the training database or files.
    The child inherits the singleton lock, so API restarts cannot overlap readers.
    """
    proc = launch = None
    deadline = asyncio.get_running_loop().time() + TIMEOUT_SECONDS
    def remaining():
        return max(0, deadline - asyncio.get_running_loop().time())
    workspace = service.settings.workspace
    with (workspace/'model-chat.lock').open('a+') as slot:
        try:
            fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield {'type': 'error', 'message': 'The model is answering another request. Try again shortly.'}
            return
        ownership.LOCK_DIRECTORY.mkdir(parents=True, exist_ok=True)
        with (ownership.LOCK_DIRECTORY/'source.lock').open('a+') as source_lock, (workspace/'checkpoint-retention.lock').open('a+') as checkpoint_lock:
            try:
                # Never wait for cleanup or prevent it for the duration of a chat.
                fcntl.flock(source_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                fcntl.flock(checkpoint_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                metadata, result = latest_snapshot(service)
                env = {**os.environ, 'PYTHONPATH': str(service.settings.root/'src'),
                       'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4',
                       'MKL_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false',
                       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'NEKAISE_CHAT_PARENT_PID': str(os.getpid())}
                payload = {'snapshot': result, 'messages': body.model_dump()['messages'],
                           'max_prompt_tokens': MAX_PROMPT_TOKENS,
                           'max_new_tokens': MAX_NEW_TOKENS,
                           'checkpoint_lock_fd': checkpoint_lock.fileno(), 'source_lock_fd': source_lock.fileno()}
                launch = asyncio.create_task(asyncio.create_subprocess_exec(*worker_command(service),
                    cwd=service.settings.root, env=env, stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                    start_new_session=True, pass_fds=(slot.fileno(), checkpoint_lock.fileno(), source_lock.fileno())))
                proc = await asyncio.wait_for(asyncio.shield(launch), remaining())
                proc.stdin.write((json.dumps(payload, ensure_ascii=False)+'\n').encode())
                await asyncio.wait_for(proc.stdin.drain(), remaining())
                proc.stdin.close()
                yield {'type': 'model', 'model': metadata}
                received, finished = 0, False
                while line := await asyncio.wait_for(proc.stdout.readline(), remaining()):
                    received += len(line)
                    if received > 128000:
                        raise ValueError('Model response exceeded its size limit')
                    message = json.loads(line)
                    if message['type'] == 'ready':
                        # Child has verified and loaded all model/tokenizer files.
                        fcntl.flock(checkpoint_lock, fcntl.LOCK_UN)
                        fcntl.flock(source_lock, fcntl.LOCK_UN)
                    if message['type'] == 'done':
                        finished = True
                    yield message
                    if message['type'] == 'error':
                        return
                code = await asyncio.wait_for(proc.wait(), remaining())
                if code or not finished:
                    raise RuntimeError('Model worker ended before completing the reply')
            except BlockingIOError:
                yield {'type': 'error', 'message': 'Model deployment or storage maintenance is in progress. Try again shortly.'}
            except TimeoutError:
                yield {'type': 'error', 'message': 'Reply timed out. Try a shorter conversation.'}
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                yield {'type': 'error', 'message': str(exc)[:400]}
            finally:
                if launch is not None:
                    # Shield cleanup against ASGI request cancellation; keep ownership
                    # until the child is reaped, before another chat can acquire slot.
                    with anyio.CancelScope(shield=True):
                        if proc is None:
                            try:
                                proc = await launch
                            except Exception:
                                pass  # Spawn failure was already returned as an error.
                        if proc is not None:
                            task = asyncio.create_task(stop_process(proc))
                            try:
                                await asyncio.shield(task)
                            except asyncio.CancelledError:
                                await task
                                raise
