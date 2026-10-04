"""Machine-local chat configuration and HTTP transport; no ML or teaching writes."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .artifacts import digest
from .processes import process_start


class ExternalChatConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    display_name: str = Field(min_length=1, max_length=100)
    model_source: str
    revision: str
    quantization: str
    model_path: Path
    sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    server_binary: Path
    port: int = Field(default=8791, ge=1024, le=65535)
    context_tokens: int = Field(default=8192, ge=4096, le=32768)
    max_new_tokens: int = Field(default=2048, ge=256, le=3072)

    @property
    def id(self):
        return digest(self.model_dump(mode='json'))

    @property
    def url(self):
        return f'http://127.0.0.1:{self.port}'

    def metadata(self):
        return {'id': self.id, 'kind': 'external', 'display_name': self.display_name,
                'model_source': self.model_source, 'revision': self.revision,
                'quantization': self.quantization, 'device': 'cuda', 'runtime': 'llama.cpp',
                'max_prompt_tokens': self.context_tokens - self.max_new_tokens,
                'max_new_tokens': self.max_new_tokens, 'thinking': False}


def read_config(workspace):
    path = workspace / 'model-chat.json'
    return ExternalChatConfig.model_validate_json(path.read_text()) if path.exists() else None


def read_runtime(workspace, config):
    try:
        runtime = json.loads((workspace / 'model-chat-runtime.json').read_text())
        if runtime['model_id'] != config.id or runtime['status'] != 'ready':
            raise ValueError('The selected chat model is not ready')
        for prefix in ('worker', 'server'):
            if not runtime.get(f'{prefix}_start') or process_start(runtime[f'{prefix}_pid']) != runtime[f'{prefix}_start']:
                raise ValueError('The local chat worker is not running')
        return runtime
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('The local chat worker is not ready. Training may have resumed.') from exc


def status(workspace, config):
    try:
        read_runtime(workspace, config)
        return {'available': True, 'model': config.metadata()}
    except ValueError as exc:
        return {'available': False, 'model': config.metadata(), 'message': str(exc)}


def upstream_error(response):
    if response.is_success:
        return
    # Do not echo provider error bodies: they may include the private prompt.
    if response.status_code in (400, 413):
        raise ValueError('Conversation exceeds the model limits. Start a new chat or shorten the message.')
    raise ValueError(f'Local chat server is unavailable (HTTP {response.status_code})')


async def stream(service, body, config):
    """Closing the iterator closes the upstream response and cancels generation."""
    runtime = read_runtime(service.settings.workspace, config)
    if body.model_id and body.model_id != config.id:
        raise ValueError('The chat model changed. Start a new chat.')
    yield {'type': 'model', 'model': config.metadata()}
    started = time.monotonic()
    payload = {'model': config.id, 'messages': body.model_dump()['messages'],
               'stream': True, 'stream_options': {'include_usage': True},
               'max_tokens': config.max_new_tokens, 'temperature': 0.7, 'top_p': 0.8,
               'top_k': 20, 'chat_template_kwargs': {'enable_thinking': False},
               'cache_prompt': False}
    async with asyncio.timeout(180), httpx.AsyncClient(
        base_url=config.url, headers={'Authorization': 'Bearer ' + runtime['api_key']},
        timeout=httpx.Timeout(60, connect=2), trust_env=False,
    ) as client:
        # Count the native rendered prompt, including all turns. Never truncate it.
        rendered = await client.post('/apply-template', json=payload)
        upstream_error(rendered)
        tokenized = await client.post('/tokenize', json={
            'content': rendered.json()['prompt'], 'add_special': True, 'parse_special': True})
        upstream_error(tokenized)
        prompt_tokens = len(tokenized.json()['tokens'])
        if prompt_tokens > config.context_tokens - config.max_new_tokens:
            raise ValueError('Conversation is too long. Start a new chat or shorten the message.')
        text, usage, timings, finish, ended = '', None, {}, None, False
        async with client.stream('POST', '/v1/chat/completions', json=payload) as response:
            upstream_error(response)
            yield {'type': 'ready', 'prompt_tokens': prompt_tokens}
            received = 0
            async for line in response.aiter_lines():
                received += len(line.encode())
                if received > 2_000_000:
                    raise ValueError('Model response exceeded its size limit')
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    ended = True
                    break
                event = json.loads(data)
                if event.get('error'):
                    raise ValueError('Local model generation failed. Please retry.')
                if event.get('usage'):
                    usage = event['usage']
                if event.get('timings'):
                    timings = event['timings']
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {}).get('content') or ''
                    if delta:
                        text += delta
                        yield {'type': 'delta', 'text': delta}
                    finish = choice.get('finish_reason') or finish
        if not ended or finish not in ('stop', 'length') or not text.strip():
            raise ValueError('The model ended before completing a reply. Please retry.')
        if not usage or not isinstance(usage.get('completion_tokens'), int):
            raise ValueError('The local server did not report response usage')
        yield {'type': 'done', 'text': text, 'tokens': usage['completion_tokens'],
               'seconds': time.monotonic() - started, 'stop_reason': 'length' if finish == 'length' else 'eos',
               'device': 'cuda', 'tokens_per_second': timings.get('predicted_per_second'),
               'prompt_tokens': usage.get('prompt_tokens')}
