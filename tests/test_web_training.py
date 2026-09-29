import hashlib
from types import SimpleNamespace

import pytest

from nekaise_loop.artifacts import Artifacts
from nekaise_loop.web_training import collect, training_rows, coverage, in_collection
from nekaise_loop.curriculum_types import CurriculumLoop
from nekaise_loop.config import CampaignConfig
from nekaise_loop.progressive_preparation import prepare_progressive
from conftest import FakeTokenizer


def context(tmp_path):
    artifacts = Artifacts(tmp_path)
    return SimpleNamespace(artifacts=artifacts, cancelled=lambda:False,
        engine=SimpleNamespace(settings=SimpleNamespace(workspace=tmp_path)),
        config=SimpleNamespace(curriculum_loop=CurriculumLoop(namespace='test', projection_artifact='a'*64,
            web_training=True, span_chars=128)))


def request():
    return {'url':'https://example.org/book/start','title':'Fixture book','purpose':'Teach this unit',
        'training':{'license':'CC-BY-4.0', 'license_url':'https://example.org/license',
        'evidence_quote':'This book is licensed under CC BY 4.0',
        'scope_reason':'Fixture license applies to the complete book path.',
        'collection_prefix':'https://example.org/book', 'max_pages':3}}


def configure(monkeypatch):
    import nekaise_loop.web_training as web
    monkeypatch.setattr(web,'permitted_source_url',lambda url:None)
    monkeypatch.setattr(web,'wait_turn',lambda *a:None)
    monkeypatch.setattr(web,'robots',lambda url:(SimpleNamespace(can_fetch=lambda *a:True),1,{'fixture':True}))
    calls=[]
    def fetch(req, **kwargs):
        calls.append(req['url'])
        text = 'This book is licensed under CC BY 4.0. '+('Textbook content about science. '*15)
        if req['url'].endswith('license'):
            text = 'This book is licensed under CC BY 4.0'
        # Second page deliberately duplicates exact first text; third link escapes prefix.
        return {'id':req['url'], 'url':req['url'], 'title':req['title'], 'text':text,
            'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'source_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'links':['/book/second','/outside/third'], 'license':'research_reference_only'}
    monkeypatch.setattr(web,'fetch_source',fetch)
    return calls


def test_licensed_collection_is_deduplicated_cached_and_prefix_bounded(tmp_path, monkeypatch):
    ctx=context(tmp_path)
    calls=configure(monkeypatch)
    key=collect(ctx,request())
    result=ctx.artifacts.get(key)
    assert len(result['pages'])==1
    assert not any('outside' in u for u in calls)
    before=list(calls)
    assert collect(ctx,request())==key and calls==before
    source=ctx.artifacts.get(result['pages'][0]['artifact'])
    assert source['training_eligible'] and source['license']=='CC-BY-4.0'
    assert source['license_evidence_artifact']


def test_missing_license_evidence_never_admits_training(tmp_path, monkeypatch):
    ctx=context(tmp_path)
    configure(monkeypatch)
    req=request()
    req['training']['evidence_quote']='An invented public-domain permission'
    with pytest.raises(ValueError,match='quote is absent'):
        collect(ctx,req)


@pytest.mark.parametrize('policy', ['license_evidence_v1', 'teacher_selected_v1'])
def test_web_targets_are_fresh_full_spans_and_never_teacher_generated(tmp_path, monkeypatch, policy):
    ctx=context(tmp_path)
    ctx.config.curriculum_loop=ctx.config.curriculum_loop.model_copy(update={'web_training_policy':policy})
    configure(monkeypatch)
    key=collect(ctx,request())
    work={'before':{},'unit':{'id':'science'}}
    references={'training_collections':[key]}
    web=training_rows(ctx,work,references)
    rows=[{'id':'generated','stream':'cpt','text':'fresh teaching', 'learning_track':'gpc','curriculum_unit_id':'science'},
          *web, {'id':'raw','stream':'corpus','text':'corpus '*100,'learning_track':'corpus','curriculum_span':True}]
    config=CampaignConfig(curriculum_loop=ctx.config.curriculum_loop).model_dump()
    config['curriculum_loop'].update(raw_target_tokens=100,web_target_tokens=180)
    frozen=prepare_progressive(rows,FakeTokenizer(),config)
    used=[r for r in frozen['rows'] if r.get('web_span')]
    assert len(used)==2
    assert frozen['progressive_preparation']['web_targets_prepared']>=180
    assert frozen['progressive_preparation']['targets']['teacher_gpc']<180
    next_key,chars=coverage(ctx,work,frozen['rows'])
    assert chars==256
    later=training_rows(ctx,{'before':{'web_coverage_artifact':next_key},'unit':{'id':'science2'}},references)
    assert later[0]['span_start']==256
    assert set(r['id'] for r in later).isdisjoint(r['id'] for r in used)
    broken=[{**used[0],'span_start':3}]
    with pytest.raises(ValueError,match='exact fresh contiguous'):
        coverage(ctx,work,broken)
    assert training_rows(ctx,work,{'sources':[{'license':'research_reference_only'}]})==[]


def test_collection_prefix_has_path_boundary():
    assert in_collection('https://example.org/book/1','https://example.org/book')
    assert not in_collection('https://example.org/books/1','https://example.org/book')
    assert not in_collection('https://other.org/book/1','https://example.org/book')


def test_readable_extractor_omits_navigation_and_preserves_code_indentation():
    from nekaise_loop.curriculum_research import TrainingPageText
    p=TrainingPageText()
    p.feed('<nav>menu</nav><main><p>Subject</p><pre>if yes:\n    code()</pre></main><footer>footer</footer>')
    text=''.join(p.main_parts)
    assert 'menu' not in text and 'footer' not in text and '    code()' in text


@pytest.mark.parametrize('quote,text,license_url', [
    ('licensed under a Creative Commons Attribution', 'This is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license.', 'https://example.org/license'),
    ('CC BY 4.0', 'CC BY 4.0', 'https://creativecommons.org/licenses/by/4.0/'),
])
def test_license_prefix_or_unrelated_deed_cannot_grant_permission(quote,text,license_url):
    from nekaise_loop.web_training import verify_permission
    permission=request()['training']
    permission.update(evidence_quote=quote,license_url=license_url)
    with pytest.raises(ValueError):
        verify_permission(permission,{'text':text,'url':license_url},request()['url'])


def test_source_admission_failure_reaches_teacher_as_reference_only(tmp_path,monkeypatch):
    from nekaise_loop.web_training import add_collections
    ctx=context(tmp_path)
    configure(monkeypatch)
    req=request();req['training']['evidence_quote']='unsupported permission claim'
    original={'plan':{'sources':[req]},'sources':[{'id':'reference','license':'research_reference_only'}]}
    result=add_collections(ctx,original)
    assert not result['training_collections']
    assert result['training_admission_failures'][0]['url']==req['url']
    assert result['sources']==original['sources']


def test_single_page_redirect_cannot_inherit_another_publishers_permission(tmp_path,monkeypatch):
    import nekaise_loop.web_training as web
    ctx=context(tmp_path);configure(monkeypatch)
    original=web.fetch_source
    def redirect(req,**kwargs):
        source=original(req,**kwargs)
        if not req['url'].endswith('license'):
            source['url']='https://different.org/book'
        return source
    monkeypatch.setattr(web,'fetch_source',redirect)
    req=request();req['training'].update(collection_prefix='',max_pages=1)
    with pytest.raises(ValueError,match='no usable pages'):
        collect(ctx,req)


def test_teacher_selected_sources_need_no_license_or_training_object(tmp_path, monkeypatch):
    from nekaise_loop.web_training import add_collections
    ctx=context(tmp_path)
    ctx.config.curriculum_loop=ctx.config.curriculum_loop.model_copy(update={'web_training_policy':'teacher_selected_v1'})
    calls=configure(monkeypatch)
    req=request();req['training']=None
    original={'plan':{'sources':[req]},'sources':[{'id':'reference','license':'research_reference_only'}],
              'training_collections':[], 'training_admission_failures':[{'error':'old license rejection'}]}
    result=add_collections(ctx,original)
    assert len(result['training_collections'])==1 and not result['training_admission_failures']
    assert calls==[req['url']]
    assert original['training_collections']==[] and original['training_admission_failures']
    collection=ctx.artifacts.get(result['training_collections'][0])
    assert collection['admission_policy']=='teacher_selected_v1'
    assert not collection['license_checked'] and collection['license_evidence_artifact'] is None
    source=ctx.artifacts.get(collection['pages'][0]['artifact'])
    assert source['license']=='not_assessed' and source['training_eligible']
    assert source['text_sha256']==hashlib.sha256(source['text'].encode()).hexdigest()
    assert result['sources']==original['sources']


def test_teacher_selection_ignores_restrictive_metadata_and_unreachable_license_url(tmp_path, monkeypatch):
    ctx=context(tmp_path)
    ctx.config.curriculum_loop=ctx.config.curriculum_loop.model_copy(update={'web_training_policy':'teacher_selected_v1'})
    calls=configure(monkeypatch);req=request()
    req['training'].update(license='CC-BY-NC-ND-4.0',license_url='https://unreachable.invalid/terms',
                          evidence_quote='Not verified and not used for admission')
    key=collect(ctx,req);collection=ctx.artifacts.get(key)
    assert collection['pages'] and not collection['license_checked']
    assert all('unreachable' not in url for url in calls)
    assert collection['permission']['license']=='CC-BY-NC-ND-4.0'
    assert ctx.artifacts.get(collection['pages'][0]['artifact'])['license']=='not_assessed'


def test_admission_policy_has_distinct_cache_and_preserves_license_history(tmp_path, monkeypatch):
    ctx=context(tmp_path);calls=configure(monkeypatch);req=request()
    old_key=collect(ctx,req);old=ctx.artifacts.get(old_key)
    ctx.config.curriculum_loop=ctx.config.curriculum_loop.model_copy(update={'web_training_policy':'teacher_selected_v1'})
    new_key=collect(ctx,req)
    assert new_key!=old_key and ctx.artifacts.get(old_key)==old
    assert old['license_checked'] and not ctx.artifacts.get(new_key)['license_checked']
    assert calls.count(req['training']['license_url'])==1


def test_recovery_preserves_operator_web_admission_policy(tmp_path):
    from nekaise_loop.config import Settings
    from nekaise_loop.service import Service, Conflict
    service=Service(Settings(tmp_path))
    loop=context(tmp_path).config.curriculum_loop.model_copy(update={'web_training_policy':'teacher_selected_v1'})
    parent=service.create('web policy',CampaignConfig(curriculum_loop=loop))
    updates={'curriculum_loop':loop.model_copy(update={'web_training_policy':'license_evidence_v1'}).model_dump()}
    with pytest.raises(Conflict,match='preserve forward progression'):
        service.continue_campaign(parent['id'],updates,actor='orchestrator')
    child=service.continue_campaign(parent['id'],reason='Verified source repair',actor='orchestrator')
    assert child['config']['curriculum_loop']['web_training_policy']=='teacher_selected_v1'
