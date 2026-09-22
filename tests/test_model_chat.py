"""Interactive chat fixtures exercise isolation, not student learning."""
import asyncio
import fcntl
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nekaise_loop.api import create_app
from nekaise_loop.model_chat import ChatRequest, latest_snapshot, stream_chat
from nekaise_loop import model_chat, ownership


@pytest.fixture
def chat_loop(setup_loop):
    settings, service, campaign, engine = setup_loop
    engine.run(campaign['id'])
    service.store.execute("UPDATE campaigns SET config=json_set(config,'$.student_format','chat_template') WHERE id=?", (campaign['id'],))
    return settings, service, campaign


def test_latest_completed_current_lineage_ignores_drafts_and_incomplete(chat_loop):
    settings, service, campaign = chat_loop
    snapshot, result = latest_snapshot(service)
    assert snapshot['round_number'] == 2
    last = snapshot['round_id']
    service.store.execute("UPDATE rounds SET status='running' WHERE id=?", (last,))
    assert latest_snapshot(service)[0]['round_number'] == 1
    service.store.execute("UPDATE rounds SET status='complete' WHERE id=?", (last,))
    from nekaise_loop.config import CampaignConfig
    config = CampaignConfig.model_validate(service.store.campaign(campaign['id'])['config'])
    child = service.create('Continuation', config, parent_id=campaign['id'])
    service.store.execute("UPDATE campaigns SET status='queued' WHERE id=?", (child['id'],))
    assert latest_snapshot(service)[0]['id'] == snapshot['id']
    service.create('Unused draft', config)
    assert latest_snapshot(service)[0]['id'] == snapshot['id']
    Path(result['checkpoint'], 'model.safetensors').unlink()
    with pytest.raises(ValueError, match='unavailable'):
        latest_snapshot(service)


@pytest.mark.parametrize('messages', [[], [{'role':'assistant','content':'x'}], [{'role':'user','content':' '}],
    [{'role':'user','content':'x'}, {'role':'user','content':'y'}, {'role':'user','content':'z'}],
    [{'role':'system','content':'x'}], [{'role':'user','content':'x'*4001}]])
def test_bounded_native_conversation(messages):
    with pytest.raises(ValidationError):
        ChatRequest(messages=messages)


def mock_worker(monkeypatch, tmp_path, mode='success'):
    # A real subprocess allows meaningful cancellation and file-lock checks.
    script = tmp_path/'child.py'
    script.write_text('''import os,sys,json,time,fcntl
p=json.loads(sys.stdin.readline())
assert os.environ['CUDA_VISIBLE_DEVICES']==''
assert os.environ['OMP_NUM_THREADS']=='4'
assert os.environ['HF_HUB_OFFLINE']=='1'
print(json.dumps({'type':'ready','pid':os.getpid()}),flush=True)
''' + ("time.sleep(60)\n" if mode == 'hang' else "print(json.dumps({'type':'delta','text':'热阻🙂'}),flush=True)\nprint(json.dumps({'type':'done','text':'热阻🙂','tokens':4,'seconds':0.1,'stop_reason':'eos'}),flush=True)\n"))
    monkeypatch.setattr(model_chat, 'worker_command', lambda service: [sys.executable, str(script)])


def test_stream_has_no_training_side_effects_and_releases_load_locks(chat_loop, monkeypatch, tmp_path):
    _, service, _ = chat_loop
    before = {t: service.store.one(f'SELECT count(*) n FROM {t}')['n'] for t in ['records','events','actions','teacher_calls','metrics']}
    mock_worker(monkeypatch, tmp_path)
    async def run():
        events = []
        async for event in stream_chat(service, ChatRequest(messages=[{'role':'user','content':'Private question'}])):
            events.append(event)
            if event['type']=='ready':
                for path in [service.settings.workspace/'checkpoint-retention.lock',ownership.LOCK_DIRECTORY/'source.lock']:
                    with path.open('a+') as f:
                        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return events
    events = asyncio.run(run())
    assert [e['type'] for e in events]==['model','ready','delta','done']
    assert events[-1]['text']=='热阻🙂'
    assert before == {t: service.store.one(f'SELECT count(*) n FROM {t}')['n'] for t in before}
    assert not (service.settings.workspace/'model-chat').exists()


def test_busy_and_disconnect_reap_only_owned_child(chat_loop, monkeypatch, tmp_path):
    _, service, _ = chat_loop
    mock_worker(monkeypatch,tmp_path,'hang')
    body=ChatRequest(messages=[{'role':'user','content':'Hello'}])
    async def run():
        stream=stream_chat(service, body)
        assert (await anext(stream))['type']=='model'
        ready=await anext(stream)
        assert ready['type']=='ready'
        second=[e async for e in stream_chat(service,body)]
        assert 'another request' in second[0]['message']
        await stream.aclose()
        assert not Path(f"/proc/{ready['pid']}").exists()
        with (service.settings.workspace/'model-chat.lock').open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    asyncio.run(run())


def test_timeout_reaps_child(chat_loop,monkeypatch,tmp_path):
    _,service,_=chat_loop
    mock_worker(monkeypatch,tmp_path,'hang')
    monkeypatch.setattr(model_chat,'TIMEOUT_SECONDS',0.3)
    async def run():
        return [e async for e in stream_chat(service,ChatRequest(messages=[{'role':'user','content':'Hello'}]))]
    events=asyncio.run(run())
    assert 'timed out' in events[-1]['message']
    assert not Path(f"/proc/{next(e['pid'] for e in events if e['type']=='ready')}").exists()


def test_maintenance_yields_busy_without_spawning(chat_loop,monkeypatch):
    _,service,_=chat_loop
    monkeypatch.setattr(model_chat,'worker_command',lambda _: pytest.fail('Must not spawn'))
    async def run():
        return [e async for e in stream_chat(service,ChatRequest(messages=[{'role':'user','content':'Hello'}]))]
    for path in [service.settings.workspace/'checkpoint-retention.lock',ownership.LOCK_DIRECTORY/'source.lock']:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert 'maintenance' in asyncio.run(run())[0]['message']


def test_api_status_chat_and_origin_guard(chat_loop,monkeypatch,tmp_path):
    settings,service,_=chat_loop
    mock_worker(monkeypatch,tmp_path)
    with TestClient(create_app(settings)) as client:
        result=client.get('/api/model')
        assert result.json()['model']['round_number']==2
        assert result.headers['cache-control']=='no-store'
        body={'messages':[{'role':'user','content':'Hi'}]}
        assert client.post('/api/model/chat',json=body,headers={'Origin':'https://bad.example'}).status_code==403
        assert client.post('/api/model/chat',json={**body,'checkpoint':'/tmp/other'}).status_code==422
        response=client.post('/api/model/chat',json=body)
        assert [json.loads(line)['type'] for line in response.text.splitlines()]==['model','ready','delta','done']


def test_native_multiturn_no_silent_truncation():
    from nekaise_loop.workers.chat import render_conversation
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            assert messages[1]['reasoning_content']==''
            assert kwargs=={'tokenize':False,'add_generation_prompt':True,'enable_thinking':False}
            return 'native-prefix'
        def encode(self,text,**kwargs):
            assert kwargs=={'add_special_tokens':False}
            return list(range(12))
    messages=[{'role':'user','content':'a'},{'role':'assistant','content':'b'},{'role':'user','content':'c'}]
    assert len(render_conversation(Tokenizer(),messages,12))==12
    with pytest.raises(ValueError,match='Start a new chat'):
        render_conversation(Tokenizer(),messages,11)


def test_asgi_send_disconnect_closes_iterator_and_reaps_child(chat_loop,monkeypatch,tmp_path):
    from contextlib import aclosing
    from nekaise_loop.api import ChatStreamResponse
    from starlette.requests import ClientDisconnect
    _,service,_=chat_loop
    mock_worker(monkeypatch,tmp_path,'hang')
    async def run():
        pid=None
        async def events():
            nonlocal pid
            async with aclosing(stream_chat(service,ChatRequest(messages=[{'role':'user','content':'Hello'}]))) as stream:
                async for event in stream:
                    if event['type']=='ready': pid=event['pid']
                    yield json.dumps(event)
        async def send(message):
            if pid: raise OSError('Client disconnected')
        async def receive(): return {'type':'http.request','body':b''}
        with pytest.raises(ClientDisconnect):
            await ChatStreamResponse(events())({'type':'http','asgi':{'spec_version':'2.4'}},receive,send)
        assert pid and not Path(f'/proc/{pid}').exists()
    asyncio.run(run())


def test_level_cancellation_still_cleans_up_child(chat_loop,monkeypatch,tmp_path):
    import anyio
    _,service,_=chat_loop
    mock_worker(monkeypatch,tmp_path,'hang')
    async def run():
        pid=None
        with anyio.CancelScope() as scope:
            async for event in stream_chat(service,ChatRequest(messages=[{'role':'user','content':'Hello'}])):
                if event['type']=='ready':
                    pid=event['pid']
                    scope.cancel()
        assert pid and not Path(f'/proc/{pid}').exists()
    anyio.run(run)


def test_cancellation_during_spawn_still_reaps_child(chat_loop,monkeypatch,tmp_path):
    _,service,_=chat_loop
    mock_worker(monkeypatch,tmp_path,'hang')
    real_spawn=asyncio.create_subprocess_exec
    children=[]
    async def run():
        starting=asyncio.Event()
        async def delayed_spawn(*args,**kwargs):
            starting.set()
            await asyncio.sleep(.05)
            proc=await real_spawn(*args,**kwargs)
            children.append(proc)
            return proc
        monkeypatch.setattr(asyncio,'create_subprocess_exec',delayed_spawn)
        async def consume():
            return [e async for e in stream_chat(service,ChatRequest(messages=[{'role':'user','content':'Hello'}]))]
        task=asyncio.create_task(consume())
        await starting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        assert len(children)==1 and children[0].returncode is not None
    asyncio.run(run())


def test_slow_consumer_deadline_does_not_cancel_calling_task(chat_loop,monkeypatch,tmp_path):
    _,service,_=chat_loop
    mock_worker(monkeypatch,tmp_path,'hang')
    monkeypatch.setattr(model_chat,'TIMEOUT_SECONDS',.15)
    async def run():
        stream=stream_chat(service,ChatRequest(messages=[{'role':'user','content':'Hello'}]))
        await anext(stream)
        await asyncio.sleep(.2)  # HTTP send backpressure, outside the generator.
        assert 'timed out' in (await anext(stream))['message']
        await stream.aclose()
    asyncio.run(run())


def test_stalled_asgi_send_has_bounded_cleanup(chat_loop,monkeypatch,tmp_path):
    from contextlib import aclosing
    from nekaise_loop.api import ChatStreamResponse
    _,service,_=chat_loop
    mock_worker(monkeypatch,tmp_path,'hang')
    async def run():
        pid=None
        async def events():
            nonlocal pid
            async with aclosing(stream_chat(service,ChatRequest(messages=[{'role':'user','content':'Hello'}]))) as stream:
                async for event in stream:
                    if event['type']=='ready': pid=event['pid']
                    yield json.dumps(event)
        async def send(message):
            if pid: await asyncio.sleep(60)
        async def receive(): return {'type':'http.request','body':b''}
        response=ChatStreamResponse(events());response.deadline_seconds=.3
        with pytest.raises(TimeoutError):
            await response({'type':'http','asgi':{'spec_version':'2.4'}},receive,send)
        assert pid and not Path(f'/proc/{pid}').exists()
    asyncio.run(run())


def test_missing_runtime_returns_error_and_releases_slot(chat_loop,monkeypatch):
    _,service,_=chat_loop
    monkeypatch.setattr(model_chat,'worker_command',lambda _:['/nonexistent/nekaise-chat-python'])
    async def run():
        return [e async for e in stream_chat(service,ChatRequest(messages=[{'role':'user','content':'Hello'}]))]
    events=asyncio.run(run())
    assert len(events)==1 and events[0]['type']=='error'
    with (service.settings.workspace/'model-chat.lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
