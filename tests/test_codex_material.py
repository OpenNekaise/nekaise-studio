import json
import time

import pytest

from nekaise_loop.author_config import AuthorPool, AuthorSpec, catalog
from nekaise_loop.material_allowance import allowance_totals
from nekaise_loop.processes import process_start
from test_claude_material import fake_cli
from test_material_authors import configured


def setup_codex(setup_loop, tmp_path, fault="", timeout=5):
    script = '''import json, sys, time
from pathlib import Path
args = sys.argv[1:]
assert args[args.index('-m')+1] == 'gpt-5.6-terra'
for flag in ['--strict-config','--ignore-user-config','--ignore-rules','--ephemeral']:
    assert flag in args
assert 'features.shell_tool=false' in args and 'features.multi_agent=false' in args
assert 'project_doc_max_bytes=0' in args
schema=json.loads(Path(args[args.index('--output-schema')+1]).read_text())
assert schema['required'] == list(schema['properties'])
prompt=sys.stdin.read()
payload=json.loads(prompt[prompt.index('\\n{')+1:])
task=payload['task']
row={'id':'variant','kind':'sft','concept':'Resistance','training_tokenization':'full_text',
     'training_text':'Doubling resistance halves heat flow.',
     'source_keys':list(task['sources'])[:1],'seed_ids':task['job']['seed_ids'],
     'rationale':'Fixture example'}
print(json.dumps({'type':'thread.started','thread_id':'fixture'}),flush=True)
print(json.dumps({'type':'turn.started'}),flush=True)
usage={'input_tokens':100,'cached_input_tokens':60,'output_tokens':40,'reasoning_output_tokens':10}
'''
    if fault == "quota":
        script += "print(json.dumps({'type':'turn.failed','error':{'message':'usage limit; retry-after: 61 seconds'}}),flush=True)\nsys.exit(1)\n"
    elif fault == "hang":
        script += "time.sleep(30)\n"
    elif fault == "oversize":
        script += "print('x'*2000001,flush=True)\ntime.sleep(30)\n"
    elif fault == "tool":
        script += "print(json.dumps({'type':'item.started','item':{'type':'command_execution'}}),flush=True)\ntime.sleep(30)\n"
    elif fault == "second_turn":
        script += "print(json.dumps({'type':'turn.started'}),flush=True)\ntime.sleep(30)\n"
    elif fault == "overrun":
        script += "usage['output_tokens']=513\n"
    elif fault == "missing_usage":
        script += "usage={}\n"
    elif fault == "schema":
        script += "row['extra_private']='do not echo'\n"
    elif fault == "error_item":
        script += "print(json.dumps({'type':'item.completed','item':{'type':'error','message':'stream disconnected'}}),flush=True)\n"
    script += "print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':json.dumps({'rows':[row]})}}),flush=True)\n"
    if fault != "no_completion":
        script += "print(json.dumps({'type':'turn.completed','usage':usage}),flush=True)\n"
    author = AuthorSpec(id="a", label="Fixture Terra", transport="codex_code", model="gpt-5.6-terra", timeout_seconds=timeout)
    settings, service, campaign, engine = configured(setup_loop, pool=AuthorPool(authors=[author]), review_policy="trusted_author_v1")
    settings.codex = fake_cli(tmp_path, script)
    return settings, service, campaign, engine


def test_codex_trusted_material_usage_and_budget_contract(setup_loop, tmp_path):
    _, service, campaign, engine = setup_codex(setup_loop, tmp_path)
    engine.run(campaign['id'])
    assert service.store.campaign(campaign['id'])['status'] == 'complete'
    details = service.snapshot(campaign['id'])['round']
    work = details['learning_work']['material_author_work']
    assert work['reported_input_tokens'] == 200
    assert work['reported_output_tokens'] == 80  # Reasoning is already included.
    assert work['reserved_output_tokens_all_attempts'] == 1024
    assert details['learning_work']['teacher_efficiency']['teacher_tokens'] is None
    assert all(m['material_origin']['model'] == 'gpt-5.6-terra' for m in details['materials'])
    for call in service.store.query('SELECT * FROM material_calls'):
        assert call['process_pid'] is None
        raw=service.artifacts.get(call['artifact'])['response']
        assert raw['execution']['provider_output_token_cap'] is None
        assert raw['execution']['observed_turns'] == 1
    pool=AuthorPool.model_validate(campaign['config']['material_authors'])
    assert catalog(pool,tmp_path/'.env')[0]['max_response_output_tokens'] is None


@pytest.mark.parametrize('fault',['quota','no_completion','schema','overrun','tool','second_turn','hang','oversize','error_item'])
def test_codex_failures_preserve_reservations_and_join_children(setup_loop,tmp_path,fault):
    _,service,campaign,engine=setup_codex(setup_loop,tmp_path,fault,timeout=1)
    started=time.monotonic()
    engine.run(campaign['id'])
    assert time.monotonic()-started < 8
    assert service.store.campaign(campaign['id'])['status'] == ('waiting' if fault=='quota' else 'failed')
    assert not service.store.query("SELECT * FROM stage_runs WHERE stage='train'")
    calls=service.store.query('SELECT * FROM material_calls')
    assert calls and all(c['process_pid'] is None and c['reserved_tokens']==512 and c['artifact'] for c in calls)
    if fault in {'tool','second_turn'}:
        assert any('cancelled with' in (c['error'] or '') for c in calls)
        assert any('response_error' in service.artifacts.get(c['artifact']) for c in calls)
    if fault=='overrun':
        assert any(json.loads(c['usage']).get('completion_tokens')==513 for c in calls)
        pool=AuthorPool.model_validate(campaign['config']['material_authors'])
        rid=service.store.one('SELECT id FROM rounds WHERE campaign_id=?',(campaign['id'],))['id']
        with service.store.connect() as db:
            totals=allowance_totals(db,rid,campaign['teacher_budget_since'] or campaign['created_at'],pool)
        assert totals['reported_output_overrun_tokens'] > 0
        assert totals['remaining_output_tokens']==pool.max_output_tokens_per_round-totals['reserved_tokens']-totals['reported_output_overrun_tokens']


def test_codex_missing_usage_is_unknown(setup_loop,tmp_path):
    _,service,campaign,engine=setup_codex(setup_loop,tmp_path,'missing_usage')
    engine.run(campaign['id'])
    assert service.snapshot(campaign['id'])['round']['learning_work']['material_author_work']['calls_without_usage']==2


def test_codex_cancellation_joins_concurrent_processes(setup_loop,tmp_path):
    _,service,campaign,engine=setup_codex(setup_loop,tmp_path,'hang',timeout=20)
    owned=[]
    def cancelled():
        if not service.store.one("SELECT name FROM sqlite_master WHERE name='material_calls'"):
            return False
        calls=service.store.query('SELECT process_pid,process_start FROM material_calls WHERE process_pid IS NOT NULL')
        if len(calls)==2:
            owned.extend(calls)
            return True
        return False
    engine.run(campaign['id'],controls=cancelled)
    assert owned and service.store.campaign(campaign['id'])['status']=='stopped'
    assert all(process_start(p['process_pid'])!=p['process_start'] for p in owned)
    assert all(c['process_pid'] is None for c in service.store.query('SELECT * FROM material_calls'))


@pytest.mark.parametrize('changes',[{'base_url':'https://example.org'},{'api_key_env':'OPENAI_API_KEY'},
    {'model':'terra'},{'options':{'effort':'automatic'}},{'options':{'max_tokens':100}}])
def test_codex_config_rejects_ambiguous_overrides(changes):
    data=dict(id='a',label='Terra',transport='codex_code',model='gpt-5.6-terra')
    with pytest.raises(ValueError):
        AuthorSpec.model_validate({**data,**changes})
