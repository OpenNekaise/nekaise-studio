"""Author adapter fixtures; no provider calls or student-learning claims."""
import json

import httpx
import pytest

from nekaise_loop.config import CampaignConfig
from nekaise_loop.material_response import NoTrainingContent, normalize
from nekaise_loop.material_types import CandidateBatch
from nekaise_loop.providers.material import AuthorResult
from conftest import FakeModel
from test_material_authors import AuthorTeacher, configured


def spec(scope='general_chat'):
    return {'job':{'id':'job', 'seed_ids':['seed'], 'material_scope':scope, 'expected_items':8}, 'sources':{'source':{}}}


def recover(content, complete=True, kind='final', scope='general_chat'):
    return normalize(AuthorResult(content if isinstance(content,str) else json.dumps(content),complete,'fixture',{},None,kind),spec(scope))


def test_metadata_damage_retains_exact_content_and_unresolved_claims():
    item={'id':'bad id', 'kind':'incorrect', 'training_tokenization':None, 'student_prompt':'  Hej?\n',
          'training_response':'  Hej!\n你好。 ', 'training_tokenization_note':'',
          'source_keys':['source','unknown',None], 'seed_ids':['seed','unknown']}
    rows,audit=recover({'rows':[item,item]})
    CandidateBatch(rows=rows)
    assert len(rows)==2 and len({r['id'] for r in rows})==2
    assert rows==recover({'rows':[item,item]})[0]
    for r in rows:
        assert r['student_prompt']==item['student_prompt'] and r['training_response']==item['training_response']
        assert r['kind']=='sft' and r['source_keys']==['source'] and r['seed_ids']==['seed']
        receipt=audit['rows'][r['id']]
        assert receipt['input_id']=='bad id'
        assert receipt['unresolved_citations']=={'source_keys':['unknown',None], 'seed_ids':['unknown']}
        assert receipt['metadata_defaults']==['concept','rationale']
        assert receipt['ignored_fields']==['training_tokenization_note']


@pytest.mark.parametrize('content',[
    {'candidates':[{'question':'Why?', 'answer':'Because.'}]},
    [{'prompt':'Why?', 'response':'Because.'}],
    {'student_prompt':'Why?', 'training_response':'Because.'},
    {'messages':[{'role':'system','content':'Never train this'}, {'role':'user','content':'Why?'},
                 {'role':'assistant','content':'Because.'}, {'role':'tool','content':'Never train this'}]},
    '```json\n{"rows":[{"question":"Why?","answer":"Because."}]}\n```',
])
def test_common_wrappers_and_chat_boundaries(content):
    rows,_=recover(content)
    assert len(rows)==1 and rows[0]['student_prompt']=='Why?' and rows[0]['training_response']=='Because.'


def test_prose_fallback_never_invents_a_chat_prompt_or_claims_chat_targets():
    for item in [{'training_response':'Actual standalone prose.'}, {'training_text':'Actual standalone prose.'}, 'Actual standalone prose.']:
        rows,audit=recover(item)
        assert rows[0]['training_text']=='Actual standalone prose.'
        assert rows[0]['training_tokenization']=='full_text' and rows[0]['kind']=='cpt'
        assert rows[0]['student_prompt']==''
        assert audit['rows'][rows[0]['id']]['material_scope']=='general_prose'
        assert audit['rows'][rows[0]['id']]['planned_material_scope']=='general_chat'


def test_partial_batch_uses_closed_fields_and_never_invents_eos_for_open_answer():
    text='{"rows":[{"question":"one?","answer":"one."},{"question":"two?","answer":"two."},{"question":"three?","answer":"unfinished'
    rows,audit=recover(text,complete=False)
    assert [r['training_response'] for r in rows]==['one.','two.']
    assert not audit['provider_complete']
    rows,_=recover('{"rows":[{"question":"one?","answer":"one.","rationale":"unfinished',complete=False)
    assert rows[0]['training_response']=='one.'


def test_full_text_alias_preserves_prose_and_normalization_provenance():
    item = {'id':'prose', 'kind':'cpt', 'student_prompt':'', 'training_text':'',
            'training_response':'', 'training_tokenization':'full_text',
            'full_text':'  Exact prose.\nNästa stycke. ', 'source_keys':['source','unknown']}
    rows,audit = recover({'rows':[item]})
    CandidateBatch(rows=rows)
    assert len(rows)==1 and rows[0]['training_text']==item['full_text']
    assert rows[0]['training_tokenization']=='full_text' and rows[0]['training_response']==''
    receipt = audit['rows'][rows[0]['id']]
    assert receipt['text_field']=='full_text' and 'full_text' not in receipt['ignored_fields']
    assert receipt['planned_material_scope']=='general_chat'
    assert receipt['material_scope']=='general_prose'
    assert rows[0]['source_keys']==['source']
    assert receipt['unresolved_citations']=={'source_keys':['unknown']}
    # Alternate copies of the same target remain one target.
    item['training_text'] = item['full_text']
    assert len(recover({'rows':[item]})[0])==1


def test_full_text_alias_recovers_closed_targets_but_never_an_open_string():
    text = '{"rows":[{"full_text":"Complete prose."},{"full_text":"Unfinished'
    rows,audit = recover(text,complete=False,scope='general_prose')
    assert [r['training_text'] for r in rows]==['Complete prose.']
    assert audit['parser']=='row_fragments' and not audit['provider_complete']
    for kind,content in [('final','{"rows":[{"full_text":"Unfinished'),
                         ('nonfinal','{"rows":[{"full_text":"Provider error"}]}')]:
        with pytest.raises(NoTrainingContent):
            recover(content,complete=False,kind=kind)


def test_broken_middle_row_does_not_discard_later_whole_rows_or_train_a_false_prefix():
    text='{"rows":[{"question":"one?","answer":"one."},{"question":"two?","answer":"wrong "quotes" here"},{"question":"three?","answer":"three."}]}'
    rows,audit=recover(text)
    assert [r['training_response'] for r in rows]==['one.','three.']
    assert audit['unusable_items']


@pytest.mark.parametrize('content,complete,kind',[
    ('I should plan the requested curriculum before writing JSON.',False,'final'),
    ('{"rows":[{"question":"why?","answer":"open',False,'final'),
    ({'rows':[]},True,'final'),
    ({'rows':[{'concept':'Only metadata','rationale':'No teaching text'}]},True,'final'),
    ({'task':{'seeds':[{'training_text':'Echoed seed'}]},'rows':[{'training_text':'Echoed task'}]},True,'final'),
    ('{"task":{"rows":[{"training_text":"Echoed task"}',False,'final'),
    ({'role':'tool','content':'Tool result'},True,'final'),
    ({'role':'user','content':'Prompt only'},True,'final'),
    ({'rows':[{'training_text':'Provider error'}]},True,'nonfinal'),
])
def test_non_teaching_or_unbounded_content_is_never_promoted(content,complete,kind):
    with pytest.raises(NoTrainingContent):
        recover(content,complete,kind)


def test_empty_rows_do_not_discard_neighbours_and_distinct_targets_are_preserved():
    rows,audit=recover({'rows':[{}, {'student_prompt':'Q', 'training_response':'Answer',
        'training_text':'Separate prose.'}, {'student_prompt':'Q','training_response':''}]})
    assert len(rows)==2 and len(audit['unusable_items'])==2
    assert [r['training_tokenization'] for r in rows]==['chat_response','full_text']


@pytest.mark.parametrize('plain',[False,True,'full_text'])
def test_worker_freezes_partial_batch_once_with_exact_provenance(setup_loop,plain):
    requests=[]
    class Teacher(AuthorTeacher):
        def curriculum(self,brief):
            p=super().curriculum(brief)
            for j in p['expansion_jobs']:
                j['material_scope']='general_chat'
            return p
    original={'rows':[{}, {'id':'bad id', 'student_prompt':'Hej?', 'training_response':'  Hej, hur går det?\n',
                          'source_keys':['unresolved-source'], 'note':False}]}
    content = (json.dumps({'rows':[{'training_text':'', 'training_response':'',
                                   'full_text':'Actual returned prose.'}]}) if plain=='full_text'
               else 'Actual returned prose.' if plain else json.dumps(original))
    def handler(req):
        requests.append(json.loads(req.content))
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':content}}],
                                     'usage':{'prompt_tokens':10,'completion_tokens':20}})
    settings,service,old,engine=configured(setup_loop,handler,teacher=Teacher,review_policy='trusted_author_v1')
    class Model(FakeModel):
        def prepare(self,checkpoint,rows):
            from test_chat_serialization import ChatTokenizer
            from nekaise_loop.training import prepare_dataset
            return prepare_dataset(rows,ChatTokenizer(),self.config.model_dump(),eos_ids=[4])
    engine.model_factory=Model
    c=service.create('Salvage fixture',CampaignConfig.model_validate({**old['config'],'material_response_policy':'salvage_v1'}))
    engine.run(c['id'])
    assert service.store.campaign(c['id'])['status']=='complete'
    assert len(requests)==2  # No paid rewriting or top-up despite expected_items=2.
    detail=service.snapshot(c['id'])['round']
    rows=detail['materials']
    assert len(rows)==2
    for row in rows:
        assert row['material_scope']==('general_prose' if plain else 'general_chat')
        assert row['training_text' if plain else 'training_response']==('Actual returned prose.' if plain else '  Hej, hur går det?\n')
        origin=row['material_origin']
        receipt=service.artifacts.get(origin['normalization_artifact'])
        raw=service.artifacts.get(receipt['response_artifact'])
        assert raw['response']['choices'][0]['message']['content']==content
        assert origin['unresolved_citations']==({} if plain else {'source_keys':['unresolved-source']})
        assert origin['planned_material_scope']=='general_chat'
    trained=[r for r in FakeModel.datasets[-1] if r.get('material_origin')]
    assert len(trained)==2 and {r['id'] for r in trained}=={r['id'] for r in rows}
    calls=service.store.query('SELECT * FROM material_calls')
    assert len(calls)==2 and all(c['status']=='complete' for c in calls)


def test_empty_author_remains_recoverable_incident_without_new_budget_or_silent_success(setup_loop):
    def handler(req):
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'{"rows":[]}'}}]})
    _,service,old,engine=configured(setup_loop,handler,review_policy='trusted_author_v1')
    c=service.create('Empty fixture',CampaignConfig.model_validate({**old['config'],'material_response_policy':'salvage_v1'}))
    engine.run(c['id'])
    assert service.store.campaign(c['id'])['status']=='failed'
    calls=service.store.query('SELECT * FROM material_calls')
    assert len(calls)==2 and all(c['status']=='failed' for c in calls)
    assert not service.store.query("SELECT * FROM stage_runs WHERE stage='train'")
    assert all('no_recoverable_teaching_text' in c['error'] for c in calls)


def test_recovery_preserves_selected_response_policy(setup_loop):
    from nekaise_loop.service import Conflict
    _,service,old,_=setup_loop
    c=service.create('Salvage policy',CampaignConfig.model_validate({**old['config'],'material_response_policy':'salvage_v1'}))
    with pytest.raises(Conflict,match='response policy'):
        service.continue_campaign(c['id'],{'material_response_policy':'strict_v1'},actor='orchestrator')
    child=service.continue_campaign(c['id'],actor='orchestrator')
    assert child['config']['material_response_policy']=='salvage_v1'


@pytest.mark.parametrize('finish,extra,usable',[
    ('stop',{},True),('length',{},True),('tool_calls',{},False),
    ('content_filter',{},False),(None,{},False),('stop',{'tool_calls':[{}]},False),
])
def test_chat_transport_preserves_the_final_message_boundary(tmp_path,finish,extra,usable):
    import asyncio
    from nekaise_loop.providers.material import OpenAIChatAuthor
    from test_material_authors import author
    async def run():
        def handler(req):
            return httpx.Response(200,json={'choices':[{'finish_reason':finish,'message':{
                'role':'assistant','content':'{"rows":[{"question":"why?","answer":"because."}]}',**extra}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await OpenAIChatAuthor(client).generate(author(),{'model':'fixture','messages':[],'max_tokens':128},
                env_file=tmp_path/'.env',max_bytes=2000,execution={'material_response_policy':'salvage_v1'})
    result=asyncio.run(run())
    if usable:
        assert normalize(result,spec())[0][0]['training_response']=='because.'
    else:
        with pytest.raises(NoTrainingContent):
            normalize(result,spec())
