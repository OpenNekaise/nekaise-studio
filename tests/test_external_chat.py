"""External chat fixtures test transport and ownership, never model quality."""
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import httpx
import pytest

from nekaise_loop import external_chat, model_chat
from nekaise_loop.model_chat import ChatRequest, stream_chat
from nekaise_loop.processes import process_start, stop_child
from nekaise_loop.workers.external_chat import training_paused, write_runtime


@pytest.fixture
def selected(setup_loop, tmp_path):
    settings, service, campaign, _ = setup_loop
    service.store.execute("UPDATE campaigns SET status='paused',operator_hold='pause' WHERE id=?", (campaign['id'],))
    model = tmp_path / 'model.gguf'
    model.write_bytes(b'fixture, not real model weights')
    config = external_chat.ExternalChatConfig(
        display_name='Qwen test', model_source='fixture/model', revision='revision', quantization='Q6_K',
        model_path=model, sha256=hashlib.sha256(model.read_bytes()).hexdigest(), server_binary=Path(sys.executable))
    (settings.workspace / 'model-chat.json').write_text(config.model_dump_json())
    write_runtime(settings.workspace, {'model_id': config.id, 'status': 'ready',
        'worker_pid': os.getpid(), 'worker_start': process_start(os.getpid()),
        'server_pid': os.getpid(), 'server_start': process_start(os.getpid()), 'api_key': 'test-local-key'})
    return settings, service, campaign, config


def test_external_identity_and_dead_or_changed_worker_fail_closed(selected):
    settings, service, _, config = selected
    status = model_chat.status(service)
    assert status['available'] and status['model']['kind'] == 'external'
    assert 'round_number' not in status['model'] and 'identity' not in status['model']
    assert 'api_key' not in json.dumps(status)
    path = settings.workspace / 'model-chat-runtime.json'
    runtime = json.loads(path.read_text())
    runtime['server_start'] = 'wrong-process-generation'
    write_runtime(settings.workspace, runtime)
    assert not model_chat.status(service)['available']
    runtime.update(server_start=process_start(os.getpid()), model_id='different')
    write_runtime(settings.workspace, runtime)
    assert not model_chat.status(service)['available']
    assert path.stat().st_mode & 0o777 == 0o600


def install_transport(monkeypatch, config, *, finished=True, prompt_tokens=3, hang=False):
    requests, closed = [], []
    class Bytes(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield ('data: ' + json.dumps({'choices': [{'delta': {'content': '热阻🙂'}, 'finish_reason': None}]}, ensure_ascii=False) + '\n\n').encode()
            if hang:
                await asyncio.sleep(60)
            if finished:
                yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":4,"prompt_tokens":3},"timings":{"predicted_per_second":25}}\n\ndata: [DONE]\n\n'
        async def aclose(self):
            closed.append(True)
    def handle(request):
        assert request.headers['authorization'] == 'Bearer test-local-key'
        assert str(request.url).startswith(config.url)
        payload = json.loads(request.content)
        requests.append((request.url.path, payload))
        if request.url.path == '/apply-template':
            return httpx.Response(200, json={'prompt': 'native fixture'})
        if request.url.path == '/tokenize':
            return httpx.Response(200, json={'tokens': [1] * prompt_tokens})
        return httpx.Response(200, stream=Bytes())
    client_class = httpx.AsyncClient
    monkeypatch.setattr(external_chat.httpx, 'AsyncClient', lambda **kwargs: client_class(**kwargs, transport=httpx.MockTransport(handle)))
    return requests, closed


def test_external_stream_multiturn_usage_privacy_and_no_training_writes(selected, monkeypatch):
    _, service, _, config = selected
    before = {t: service.store.one(f'SELECT count(*) n FROM {t}')['n'] for t in ['actions','events','records','metrics','teacher_calls']}
    requests, closed = install_transport(monkeypatch, config)
    messages = [{'role':'user','content':'Hello'}, {'role':'assistant','content':'Hi'}, {'role':'user','content':'Explain heat resistance'}]
    async def run():
        return [e async for e in stream_chat(service, ChatRequest(messages=messages, model_id=config.id))]
    events = asyncio.run(run())
    assert [e['type'] for e in events] == ['model','ready','delta','done']
    assert events[-1]['text'] == '热阻🙂' and events[-1]['tokens_per_second'] == 25
    assert requests[-1][1]['messages'] == messages
    assert requests[-1][1]['chat_template_kwargs'] == {'enable_thinking': False}
    assert closed
    assert before == {t: service.store.one(f'SELECT count(*) n FROM {t}')['n'] for t in before}


@pytest.mark.parametrize('mode', ['truncated', 'context', 'changed'])
def test_external_rejects_incomplete_context_overflow_and_model_switch(selected, monkeypatch, mode):
    _, service, _, config = selected
    requests, _ = install_transport(monkeypatch, config, finished=mode != 'truncated',
        prompt_tokens=config.context_tokens if mode == 'context' else 3)
    async def run():
        return [e async for e in stream_chat(service, ChatRequest(messages=[{'role':'user','content':'Hi'}],
            model_id='changed' if mode == 'changed' else config.id))]
    events = asyncio.run(run())
    assert events[-1]['type'] == 'error'
    assert not any(e['type'] == 'done' for e in events)
    if mode != 'truncated':
        assert not any(path == '/v1/chat/completions' for path, _ in requests)


def test_external_disconnect_closes_upstream_and_releases_slot(selected, monkeypatch):
    settings, service, _, config = selected
    _, closed = install_transport(monkeypatch, config, hang=True)
    async def run():
        stream = stream_chat(service, ChatRequest(messages=[{'role':'user','content':'Hi'}]))
        assert (await anext(stream))['type'] == 'model'
        assert (await anext(stream))['type'] == 'ready'
        assert (await anext(stream))['type'] == 'delta'
        await stream.aclose()
        with (settings.workspace/'model-chat.lock').open('a+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    asyncio.run(run())
    assert closed


def wait_for(predicate, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.05)
    pytest.fail('Owned worker did not reach expected state')


def launch_fixture_worker(selected, tmp_path):
    settings, _, _, config = selected
    server = tmp_path / 'server.py'
    server.write_text(f'#!{sys.executable}\n' + '''import http.server,json,sys
args=sys.argv
alias=args[args.index('--alias')+1]
class H(http.server.BaseHTTPRequestHandler):
 def log_message(self,*a): pass
 def do_GET(self):
  body=json.dumps({'data':[{'id':alias}]} if self.path=='/v1/models' else {'status':'ok'}).encode()
  self.send_response(200); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
http.server.HTTPServer(('127.0.0.1',int(args[args.index('--port')+1])),H).serve_forever()
''')
    server.chmod(0o700)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port = sock.getsockname()[1]
    config = config.model_copy(update={'server_binary': server, 'port': port})
    (settings.workspace/'model-chat.json').write_text(config.model_dump_json())
    code = ('from pathlib import Path; from nekaise_loop import ownership; '
            f'ownership.LOCK_DIRECTORY=Path({str(tmp_path)!r}); '
            'from nekaise_loop.workers.external_chat import run; from nekaise_loop.config import Settings; '
            f'run(Settings({str(settings.workspace)!r}))')
    proc = subprocess.Popen([sys.executable,'-c',code], env={**os.environ,'PYTHONPATH':str(settings.root/'src')},
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
    return proc, config


def test_worker_lease_blocks_training_and_resume_stops_owned_server(selected, tmp_path):
    settings, service, campaign, _ = selected
    assert training_paused(settings.workspace)
    proc, config = launch_fixture_worker(selected, tmp_path)
    try:
        wait_for(lambda: external_chat.status(settings.workspace, config)['available'])
        runtime = external_chat.read_runtime(settings.workspace, config)
        with (settings.workspace/'worker.lock').open('a+') as lease:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        service.action(campaign['id'], 'resume', spawn=False)
        assert service.store.one("SELECT id FROM actions WHERE handled_at IS NULL AND kind='resume'")
        assert not training_paused(settings.workspace)
        assert proc.wait(timeout=10) == 0
        assert process_start(runtime['server_pid']) is None
        assert not model_chat.status(service)['available']
        with (settings.workspace/'worker.lock').open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from nekaise_loop.supervisor import tick
        assert tick(service) == ['worker']
    finally:
        stop_child(proc)


def test_worker_cannot_overlap_training_lease(selected, tmp_path):
    settings, _, _, _ = selected
    with (settings.workspace/'worker.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proc, _ = launch_fixture_worker(selected, tmp_path)
        try:
            assert proc.wait(timeout=10) != 0
            assert b'BlockingIOError' in proc.stderr.read()
        finally:
            stop_child(proc)


def test_server_inherits_lease_and_dies_with_worker(selected, tmp_path):
    settings, _, _, _ = selected
    proc, config = launch_fixture_worker(selected, tmp_path)
    runtime = None
    try:
        wait_for(lambda: external_chat.status(settings.workspace, config)['available'])
        runtime = external_chat.read_runtime(settings.workspace, config)
        # Suspend child so it cannot process parent-death SIGTERM yet.
        os.kill(runtime['server_pid'], signal.SIGSTOP)
        proc.kill(); proc.wait(timeout=5)
        with (settings.workspace/'worker.lock').open('a+') as lease:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.kill(runtime['server_pid'], signal.SIGCONT)
        from nekaise_loop.ownership import locked
        wait_for(lambda: not locked(settings.workspace/'worker.lock'))
    finally:
        stop_child(proc)
        if runtime and process_start(runtime['server_pid']) == runtime['server_start']:
            os.kill(runtime['server_pid'], signal.SIGKILL)
