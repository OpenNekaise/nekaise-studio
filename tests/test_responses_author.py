"""Protocol/worker fixtures; no network calls or student learning claims."""
import asyncio
import json

import httpx
import pytest

from nekaise_loop.author_config import AuthorPool, AuthorSpec
from nekaise_loop.material_jobs import CandidateValidationError, candidates
from nekaise_loop.providers.material import AuthorHTTPError, AuthorWaiting, OpenAIResponsesAuthor
from nekaise_loop.providers.responses_material import reported_usage
from test_material_authors import configured, response as chat_fixture


def author(**changes):
    return AuthorSpec(**dict({'id':'a', 'label':'Responses fixture', 'transport':'openai_responses',
        'base_url':'https://fixture.invalid/v1', 'model':'fixture-model'}, **changes))


def message(text='{"rows": []}'):
    return {'type':'message','role':'assistant','status':'completed',
            'content':[{'type':'output_text','text':text}]}


def terminal(**changes):
    return {'type':'response.completed','response':dict({'id':'response-1','model':'served-fixture',
        'status':'completed','output':[{'type':'reasoning','summary':[{'text':'Not teaching text'}]},message()],
        'usage':{'input_tokens':30,'output_tokens':20,'total_tokens':50,
                 'input_tokens_details':{'cached_tokens':10},'output_tokens_details':{'reasoning_tokens':12}}}, **changes)}


def frame(event, newline='\n'):
    return (': keepalive'+newline+'event: ignored'+newline+'data: '+json.dumps(event,ensure_ascii=False)+newline+newline).encode()


class Chunks(httpx.AsyncByteStream):
    def __init__(self, data, size=1, error=None):
        self.data, self.size, self.error, self.closed = data, size, error, False
    async def __aiter__(self):
        for i in range(0,len(self.data),self.size):
            yield self.data[i:i+self.size]
        if self.error:
            raise self.error
    async def aclose(self):
        self.closed = True


def generate(tmp_path, stream, *, spec=None, status=200, headers=None, cap=2000000):
    seen=[]
    def handler(request):
        seen.append(request)
        return httpx.Response(status,stream=stream,headers=headers or {'content-type':'text/event-stream'})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await OpenAIResponsesAuthor(client).generate(spec or author(),
                {'model':'fixture-model','messages':[{'role':'system','content':'JSON only'},{'role':'user','content':'Fixture'}],
                 'max_tokens':128,'stream':False,'response_format':{'type':'json_object'}},
                 env_file=tmp_path/'.env',max_bytes=cap)
    return asyncio.run(run()), seen


def test_stream_fragments_utf8_usage_final_only_and_exact_request(tmp_path):
    end=terminal(output=[message('Hej, 你好!')])
    stream=Chunks(frame({'type':'response.created','response':{'id':'response-1'}})+'data: {"type":\ndata: "response.output_text.delta", "delta":"ignored"}\n\n'.encode()+frame(end,'\r\n'),error=AssertionError('must close at terminal'))
    result,seen=generate(tmp_path,stream)
    assert stream.closed and result.complete and result.content=='Hej, 你好!'
    assert result.model=='served-fixture'
    assert result.usage=={'prompt_tokens':30,'completion_tokens':20,'total_tokens':50,'prompt_cache_hit_tokens':10}
    assert result.raw['response']==end['response'] and result.raw['event_count']==3
    body=json.loads(seen[0].content)
    assert str(seen[0].url)=='https://fixture.invalid/v1/responses'
    assert body=={'model':'fixture-model','input':[{'role':'system','content':'JSON only'},{'role':'user','content':'Fixture'}],
                  'max_output_tokens':128,'stream':True,'store':False,'text':{'format':{'type':'json_object'}}}
    assert seen[0].headers['accept']=='text/event-stream'
    assert len(result.raw['request_sha256'])==64


@pytest.mark.parametrize('tail',[b'',b'data: [DONE]\n\n',b'data: {"type":"response.completed"}'])
def test_no_terminal_never_accepts_even_valid_json_delta(tmp_path,tail):
    with pytest.raises(AuthorHTTPError,match='missing_terminal_response'):
        generate(tmp_path,Chunks(frame({'type':'response.output_text.delta','delta':'{"rows":[]}'} )+tail))


@pytest.mark.parametrize('kind,status', [('response.incomplete','incomplete'),('response.failed','failed'),('response.completed','incomplete'),('response.done','incomplete')])
def test_incomplete_terminals_retain_usage_not_material(tmp_path,kind,status):
    event=terminal(status=status); event['type']=kind
    result,_=generate(tmp_path,Chunks(frame(event)))
    assert not result.complete and result.usage['completion_tokens']==20
    with pytest.raises(CandidateValidationError,match='response_incomplete'):
        candidates(result,{'sources':{},'job':{'seed_ids':[]}})


def test_documented_done_alias_needs_completed_envelope(tmp_path):
    event=terminal(); event['type']='response.done'
    result,_=generate(tmp_path,Chunks(frame(event)))
    assert result.complete


@pytest.mark.parametrize('output', [[],[message(),message()], [message(),{'type':'function_call'}],
    [dict(message(),role='user')], [dict(message(),status='in_progress')],
    [dict(message(),content=[{'type':'refusal','refusal':'No'}])], [message('')], [None]])
def test_unsupported_output_boundaries_do_not_become_candidates(tmp_path,output):
    result,_=generate(tmp_path,Chunks(frame(terminal(output=output))))
    assert not result.complete and result.content is None


@pytest.mark.parametrize('data,expected',[
    (b'data: {bad}\n\n','invalid_event_encoding_or_json'),
    (b'data: {"type":"\xff"}\n\n','invalid_event_encoding_or_json'),
    (b'data: {"type":"x","n":NaN}\n\n','invalid_event_encoding_or_json'),
    (b'data: []\n\n','invalid_event'),
    (b'data: {"type":"response.completed"}\n\n','missing_terminal_envelope'),
    (frame({'type':'error','error':{'message':'fixture failure'}}),'error event'),
    (frame({'type':'response.created','response':{'id':'other'}})+frame(terminal()),'conflicting_response_ids')])
def test_protocol_errors_are_durable(tmp_path,data,expected):
    with pytest.raises(AuthorHTTPError,match=expected) as error:
        generate(tmp_path,Chunks(data))
    assert error.value.evidence


@pytest.mark.parametrize('status,expected',[(402,AuthorWaiting),(429,AuthorWaiting),(401,AuthorHTTPError),(500,AuthorHTTPError)])
def test_http_errors_without_retry(tmp_path,status,expected):
    with pytest.raises(expected) as error:
        generate(tmp_path,Chunks(b'{"error":{"message":"fixture failure"}}'),status=status,
                 headers={'content-type':'application/json','retry-after':'60'})
    assert error.value.evidence['status']==status
    if status==429:
        assert error.value.retry_seconds==60


def test_non_sse_oversize_and_disconnect_are_unknown_usage(tmp_path):
    for stream,kwargs,expected in [(Chunks(b'{}'),{'headers':{'content-type':'application/json'}},'expected_event_stream'),
                                  (Chunks(b':'*300),{'cap':128},'response_exceeded_byte_budget'),
                                  (Chunks(frame({'type':'response.created','response':{'id':'response-1'}}),error=httpx.ReadError('do not echo transport text')),{},'ReadError')]:
        with pytest.raises(AuthorHTTPError,match=expected) as error:
            generate(tmp_path,stream,**kwargs)
        assert not getattr(error.value,'usage',None)
        assert 'do not echo' not in str(error.value)
        assert stream.closed


@pytest.mark.parametrize('escaped',[False,True])
def test_echoed_credential_is_never_persisted(tmp_path,escaped):
    key='fixture-secret-123'
    (tmp_path/'.env').write_text('AUTHOR_KEY='+key+'\n')
    data=frame(terminal(output=[message(key)]))
    if escaped:
        data=data.replace(key.encode(),b'\\u0066ixture-secret-123')
    with pytest.raises(AuthorHTTPError) as error:
        generate(tmp_path,Chunks(data),spec=author(api_key_env='AUTHOR_KEY'))
    assert error.value.evidence=={'transport_error':'credential_echo_withheld'}
    assert key not in str(error.value)


def test_reported_usage_does_not_invent_or_double_count():
    assert reported_usage({})=={}
    assert reported_usage({'usage':{'input_tokens':True,'output_tokens':-1,'total_tokens':'8'}})=={}
    assert reported_usage({'usage':{'output_tokens':5,'output_tokens_details':{'reasoning_tokens':4}}})=={'completion_tokens':5}


def test_budget_overrun_preserves_usage_for_recovery(tmp_path):
    with pytest.raises(AuthorHTTPError,match='exceeded its reservation') as error:
        generate(tmp_path,Chunks(frame(terminal(usage={'input_tokens':30,'output_tokens':129}))))
    assert error.value.usage=={'prompt_tokens':30,'completion_tokens':129}
    assert error.value.evidence['response']['usage']['output_tokens']==129


@pytest.mark.parametrize('option',['input','instructions','max_output_tokens','text','tools','tool_choice',
    'previous_response_id','conversation','background','store','include','truncation','stream_options'])
def test_options_cannot_bypass_job_or_transport(option):
    with pytest.raises(ValueError,match='cannot override'):
        author(options={option:'override'})


def test_real_worker_contract_with_responses_transport(setup_loop):
    requests=[]
    def handler(req):
        body=json.loads(req.content); requests.append(body)
        # Reuse fixture candidates, not any real student or provider result.
        fixture=chat_fixture(httpx.Request('POST','https://fixture.invalid',json={'messages':body['input']})).json()
        return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Chunks(frame(terminal(
            output=[message(fixture['choices'][0]['message']['content'])])),size=7))
    settings,service,campaign,engine=configured(setup_loop,handler,pool=AuthorPool(authors=[author()]))
    engine.run(campaign['id'])
    assert service.store.campaign(campaign['id'])['status']=='complete'
    detail=service.snapshot(campaign['id'])['round']
    assert len(detail['materials'])==2
    assert all(m['material_origin']['model']=='served-fixture' for m in detail['materials'])
    assert detail['learning_work']['material_author_work']['reported_output_tokens']==40
    calls=service.store.query('SELECT * FROM material_calls')
    assert len(calls)==2 and all(c['status']=='complete' for c in calls)
    for c in calls:
        artifact=service.artifacts.get(c['artifact'])
        assert artifact['response']['transport']=='openai_responses'
        assert service.artifacts.get(artifact['request_artifact'])['request']['max_tokens']==512
    assert all(r['max_output_tokens']==512 for r in requests)


def test_escaped_credential_in_malformed_http_error_is_withheld(tmp_path):
    key='fixture-secret-123'
    (tmp_path/'.env').write_text('AUTHOR_KEY='+key+'\n')
    with pytest.raises(AuthorHTTPError) as error:
        generate(tmp_path,Chunks(b'{"error":"\\u0066ixture-secret-123'),status=500,
                 spec=author(api_key_env='AUTHOR_KEY'))
    evidence=error.value.evidence['response']
    assert evidence['transport_error']=='invalid_json_error_body'
    assert 'fixture' not in json.dumps(evidence)


def test_cancellation_closes_the_stream(tmp_path):
    class Hanging(httpx.AsyncByteStream):
        closed=False
        async def __aiter__(self):
            yield b': keepalive\n\n'
            started.set()
            await asyncio.Event().wait()
        async def aclose(self):
            self.closed=True
    async def run():
        nonlocal started
        started=asyncio.Event()
        stream=Hanging()
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req:httpx.Response(200,stream=stream,headers={'content-type':'text/event-stream'}))) as client:
            task=asyncio.create_task(OpenAIResponsesAuthor(client).generate(author(),
                {'model':'fixture-model','messages':[],'max_tokens':128},env_file=tmp_path/'.env',max_bytes=1000))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert stream.closed
    started=None
    asyncio.run(run())


@pytest.mark.parametrize('nested_escape',[False,True])
def test_credential_split_across_final_parts_is_withheld(tmp_path,nested_escape):
    key='fixture-secret-123'
    (tmp_path/'.env').write_text('AUTHOR_KEY='+key+'\n')
    content=json.dumps({'secret':key})
    if nested_escape:
        content=content.replace('fixture',r'\u0066ixture')
    parts=[{'type':'output_text','text':content[:12]},{'type':'output_text','text':content[12:]}]
    with pytest.raises(AuthorHTTPError) as error:
        generate(tmp_path,Chunks(frame(terminal(output=[dict(message(),content=parts)]))),spec=author(api_key_env='AUTHOR_KEY'))
    assert error.value.evidence=={'transport_error':'credential_echo_withheld'}


def test_only_supported_common_text_request_shapes_are_mapped():
    from nekaise_loop.providers.responses_material import wire_request
    base={'model':'fixture','messages':[{'role':'user','content':'text'}],'max_tokens':128}
    assert 'text' not in wire_request(base)
    with pytest.raises(ValueError,match='text-only'):
        wire_request(dict(base,messages=[{'role':'user','content':[{'type':'text','text':'image later'}]}]))
    with pytest.raises(ValueError,match='json_object'):
        wire_request(dict(base,response_format={'type':'json_schema','json_schema':{}}))
