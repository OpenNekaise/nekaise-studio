import base64
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from nekaise_loop.artifacts import Artifacts
from nekaise_loop.curriculum_types import CurriculumLoop, ResearchPlan
from nekaise_loop.processes import Cancelled
from nekaise_loop.web_http import WebSession, response_bytes
from nekaise_loop.web_inventory import collect, research_sources, fresh_supply, catalog
from nekaise_loop.web_training import training_rows, coverage


def context(tmp_path, cancelled=lambda:False):
    return SimpleNamespace(artifacts=Artifacts(tmp_path),cancelled=cancelled,
        engine=SimpleNamespace(settings=SimpleNamespace(workspace=tmp_path)),
        config=SimpleNamespace(curriculum_loop=CurriculumLoop(namespace='test',projection_artifact='a'*64,
            web_training=True,web_training_policy='teacher_selected_v1',web_crawl_policy='inventory_v1',span_chars=128)))


def source(url='https://example.org/book/one', **training):
    return {'url':url,'title':'Selected teaching source','purpose':'Explain mechanisms',
        'training':{'collection_prefix':'https://example.org/book/','max_pages':8,**training}}


def page(label, links=''):
    return '<main><p>'+label+' substantive teaching paragraph. '*20+'</p>'+links+'</main>'


def network(monkeypatch, tmp_path, handler):
    monkeypatch.setattr('nekaise_loop.web_access.public_url',lambda url:None)
    monkeypatch.setattr('nekaise_loop.web_http.wait_turn',lambda *a,**k:None)
    calls=[]
    def route(request):
        calls.append(str(request.url))
        return handler(request)
    factory=lambda **kw:httpx.Client(transport=httpx.MockTransport(route),**kw)
    return WebSession(tmp_path,client_factory=factory),calls


def test_multipage_https_upgrade_robots_redirect_and_raw_cache(tmp_path,monkeypatch):
    def handler(r):
        if r.url.path=='/robots.txt':return httpx.Response(301,headers={'location':'/access.txt'})
        if r.url.path=='/access.txt':return httpx.Response(200,text='User-agent: *\nAllow: /')
        text=page(r.url.path,'<a href="http://example.org/book/two">next</a><a href="/outside/three">out</a><a href="/book/gpqa/data">excluded</a>')
        return httpx.Response(200,text=text,headers={'content-type':'text/html'})
    session,calls=network(monkeypatch,tmp_path,handler);ctx=context(tmp_path)
    key=collect(ctx,source(),session);data=ctx.artifacts.get(key)
    assert len(data['pages'])==2 and data['complete']
    assert calls.count('https://example.org/robots.txt')==1
    assert calls.count('https://example.org/access.txt')==1
    assert all('outside' not in u and 'gpqa' not in u for u in calls)
    assert data['url_mappings'][0]['fetch_url']=='https://example.org/book/two'
    before=list(calls);assert collect(ctx,source(),session)==key and calls==before
    # A different collection reuses the response, not a second body download.
    other=collect(ctx,{**source(),'purpose':'Another approved teaching use'},session)
    assert len(ctx.artifacts.get(other)['pages'])==2 and calls==before
    raw=ctx.artifacts.get(ctx.artifacts.get(data['pages'][0]['artifact'])['raw_response_artifact'])
    assert hashlib.sha256(base64.b64decode(raw['body_base64'])).hexdigest()==raw['sha256']


def test_sitemap_index_discovery_stays_in_selected_section(tmp_path,monkeypatch):
    def handler(r):
        if r.url.path=='/robots.txt':return httpx.Response(404)
        if r.url.path=='/sitemap.xml':return httpx.Response(200,text='<sitemapindex><sitemap><loc>https://example.org/child.xml</loc></sitemap></sitemapindex>')
        if r.url.path=='/child.xml':return httpx.Response(200,text='<urlset><url><loc>https://example.org/book/two</loc></url><url><loc>https://elsewhere.org/book/private</loc></url><url><loc>https://example.org/outside</loc></url></urlset>')
        return httpx.Response(200,text=page(r.url.path),headers={'content-type':'text/html'})
    session,calls=network(monkeypatch,tmp_path,handler);ctx=context(tmp_path)
    data=ctx.artifacts.get(collect(ctx,source(sitemap_urls=['https://example.org/sitemap.xml']),session))
    assert len(data['pages'])==2 and len(data['discovery_visited'])==2
    assert not any('elsewhere' in u or 'outside' in u for u in calls)


def test_redirect_is_checked_before_fetch_and_403_is_not_retried(tmp_path,monkeypatch):
    def handler(r):
        if r.url.path=='/robots.txt':return httpx.Response(404)
        if r.url.path=='/book/one':return httpx.Response(302,headers={'location':'https://elsewhere.org/private'})
        return httpx.Response(403)
    session,calls=network(monkeypatch,tmp_path,handler);ctx=context(tmp_path)
    with pytest.raises(ValueError,match='no usable pages'):collect(ctx,source(),session)
    assert all('elsewhere' not in u for u in calls)
    with pytest.raises(ValueError,match='no usable pages'):collect(ctx,source('https://example.org/book/denied'),session)
    assert calls.count('https://example.org/book/denied')==1


def test_retry_after_cache_integrity_and_benchmark_exclusions(tmp_path,monkeypatch):
    attempts=[]
    def handler(r):
        attempts.append(1)
        if len(attempts)==1:return httpx.Response(429,headers={'retry-after':'0'})
        return httpx.Response(200,text=page('retry'),headers={'content-type':'text/html'})
    session,calls=network(monkeypatch,tmp_path,handler)
    raw=response_bytes(tmp_path,'https://example.org/book/one',client_factory=session.client_factory)
    assert len(calls)==2
    p=next((tmp_path/'curriculum/http').glob('*.json'));cached=json.loads(p.read_text());cached['body_base64']=base64.b64encode(b'changed').decode();p.write_text(json.dumps(cached))
    with pytest.raises(ValueError,match='integrity'):response_bytes(tmp_path,raw['url'],client_factory=session.client_factory)
    for url in ('https://example.org/datasets/TIGER-Lab/MMLU-Pro','https://example.org/gpqa','https://example.org/fineweb'):
        with pytest.raises(ValueError,match='excluded'):response_bytes(tmp_path,url,client_factory=session.client_factory)
    assert len(calls)==2


def test_declared_alternative_catalog_reuse_and_exact_fresh_coverage(tmp_path,monkeypatch):
    def handler(r):
        if r.url.path=='/robots.txt':return httpx.Response(404)
        if r.url.path=='/missing':return httpx.Response(404)
        return httpx.Response(200,text=page('actual source'),headers={'content-type':'text/html'})
    session,calls=network(monkeypatch,tmp_path,handler)
    monkeypatch.setattr('nekaise_loop.web_inventory.WebSession',lambda *args:session)
    ctx=context(tmp_path)
    req={'url':'https://example.org/missing','title':'Primary','purpose':'Teach the same concept',
        'alternatives':[{'url':'https://example.org/book/one','title':'Declared alternative','training':None}]}
    plan=ResearchPlan(sources=[req],teaching_direction='Use actual selected bytes').model_dump()
    result=research_sources(ctx,plan,'unit-a');assert result['source_selections'][0]['used_alternative']
    assert result['sources'][0]['url']=='https://example.org/book/one'
    ref=result['sources'][0];c=ctx.artifacts.get(result['training_collections'][0]);full=ctx.artifacts.get(c['pages'][0]['artifact'])
    assert ref['source_sha256']==full['source_sha256'] and not ref['training_eligible'] and full['training_eligible']
    work={'before':{},'unit':{'id':'unit-a'}};rows=training_rows(ctx,work,result)
    key,chars=coverage(ctx,work,rows[:1]);positions=ctx.artifacts.get(key)['positions']
    supply=fresh_supply(ctx,result,positions)
    assert supply[0]['fresh_chars_after_frontier']==len(full['text'])-128
    before=calls.count('https://example.org/book/one')
    reused=research_sources(ctx,{'sources':[],'reuse_source_ids':result['source_catalog_ids']},'unit-b')
    assert reused['sources'][0]['text']==ref['text'] and calls.count('https://example.org/book/one')==before
    cat=catalog(ctx,['unit-b'],positions);assert cat['total_sources']==1 and len(cat['matching_units'])==1
    assert cat['matching_units'][0]['selected_for_unit_ids']==['unit-a','unit-b']
    assert chars==128
    with pytest.raises(ValueError,match='unknown'):research_sources(ctx,{'reuse_source_ids':['f'*64]},'unit-b')


def test_interruption_preserves_first_page_and_resumes_frontier(tmp_path,monkeypatch):
    state={'cancel':False}
    ctx=context(tmp_path,lambda:state['cancel'])
    def handler(r):
        if r.url.path=='/robots.txt':return httpx.Response(404)
        return httpx.Response(200,text=page(r.url.path,'<a href="/book/two">next</a>'),headers={'content-type':'text/html'})
    session,calls=network(monkeypatch,tmp_path,handler)
    from nekaise_loop import web_inventory as inventory
    original=inventory.extract_page
    def extract(raw,req):
        value=original(raw,req);state['cancel']=True;return value
    monkeypatch.setattr(inventory,'extract_page',extract)
    with pytest.raises(Cancelled):collect(ctx,source(),session)
    saved=json.loads(next((tmp_path/'curriculum/web').glob('*.json')).read_text())
    assert len(saved['pages'])==1 and saved['pending']==['https://example.org/book/two']
    state['cancel']=False;monkeypatch.setattr(inventory,'extract_page',original)
    result=ctx.artifacts.get(collect(ctx,source(),session))
    assert len(result['pages'])==2 and calls.count('https://example.org/book/one')==1


def test_inventory_contract_rejects_disabled_web_and_unscoped_discovery(tmp_path):
    with pytest.raises(ValueError,match='enabled Teacher-selected'):
        CurriculumLoop(namespace='test',projection_artifact='a'*64,web_crawl_policy='inventory_v1')
    with pytest.raises(ValueError,match='requires a selected path'):
        collect(context(tmp_path),{'url':'https://example.org/one','title':'x','purpose':'y','training':{'max_pages':2}})


def test_host_pacing_respects_deadline(monkeypatch):
    import time
    from nekaise_loop.web_access import wait_turn, _NEXT
    _NEXT['pace.example.org']=time.monotonic()+60
    with pytest.raises(TimeoutError,match='pacing'):
        wait_turn('https://pace.example.org/book',1,lambda:False,deadline=time.monotonic()+1)


def test_time_slice_returns_useful_pages_and_keeps_unfetched_frontier(tmp_path,monkeypatch):
    import time
    clock=[time.time()]
    monkeypatch.setattr('nekaise_loop.web_inventory.wall_time',lambda:clock[0])
    def handler(r):
        if r.url.path=='/robots.txt':return httpx.Response(404)
        return httpx.Response(200,text=page(r.url.path,'<a href="/book/two">next</a>'),headers={'content-type':'text/html'})
    session,calls=network(monkeypatch,tmp_path,handler);ctx=context(tmp_path)
    original=session.fetch
    def limited(url,*a,**kw):
        if url.endswith('/two'):raise TimeoutError('Acquisition slice exhausted')
        return original(url,*a,**kw)
    monkeypatch.setattr(session,'fetch',limited)
    first=ctx.artifacts.get(collect(ctx,source(),session))
    assert len(first['pages'])==1 and not first['complete']
    assert first['pending']==['https://example.org/book/two'] and first['bounded_by']=='cooldown'
    clock[0]+=31
    monkeypatch.setattr(session,'fetch',original)
    second=ctx.artifacts.get(collect(ctx,source(),session))
    assert len(second['pages'])==2 and second['complete']
    assert calls.count('https://example.org/book/one')==1


def test_sitemap_timeout_is_not_mislabeled_as_complete(tmp_path,monkeypatch):
    import time
    clock=[time.time()]
    monkeypatch.setattr('nekaise_loop.web_inventory.wall_time',lambda:clock[0])
    def handler(r):
        if r.url.path=='/robots.txt':return httpx.Response(404)
        if r.url.path=='/sitemap.xml':return httpx.Response(200,text='<urlset><url><loc>https://example.org/book/two</loc></url></urlset>')
        return httpx.Response(200,text=page(r.url.path),headers={'content-type':'text/html'})
    session,calls=network(monkeypatch,tmp_path,handler);ctx=context(tmp_path);original=session.fetch
    def limited(url,*a,**kw):
        if url.endswith('.xml'):raise TimeoutError('Sitemap slice exhausted')
        return original(url,*a,**kw)
    monkeypatch.setattr(session,'fetch',limited)
    req=source(sitemap_urls=['https://example.org/sitemap.xml'])
    first=ctx.artifacts.get(collect(ctx,req,session))
    assert not first['complete'] and first['discovery_pending']==['https://example.org/sitemap.xml']
    clock[0]+=31
    monkeypatch.setattr(session,'fetch',original)
    second=ctx.artifacts.get(collect(ctx,req,session))
    assert second['complete'] and len(second['pages'])==2


def test_cancelled_sitemap_remains_pending_and_resumes(tmp_path,monkeypatch):
    def handler(r):
        if r.url.path=='/robots.txt':return httpx.Response(404)
        if r.url.path=='/sitemap.xml':return httpx.Response(200,text='<urlset><url><loc>https://example.org/book/two</loc></url></urlset>')
        return httpx.Response(200,text=page(r.url.path),headers={'content-type':'text/html'})
    session,calls=network(monkeypatch,tmp_path,handler);ctx=context(tmp_path);original=session.fetch
    def cancel(url,*a,**kw):raise Cancelled('Operator cancelled sitemap')
    monkeypatch.setattr(session,'fetch',cancel)
    req=source(sitemap_urls=['https://example.org/sitemap.xml'])
    with pytest.raises(Cancelled):collect(ctx,req,session)
    saved=json.loads(next((tmp_path/'curriculum/web').glob('*.json')).read_text())
    assert saved['discovery_pending']==['https://example.org/sitemap.xml'] and saved['discovery_visited']==[]
    monkeypatch.setattr(session,'fetch',original)
    assert len(ctx.artifacts.get(collect(ctx,req,session))['pages'])==2


def test_slow_head_does_not_block_other_pages_and_retry_count_is_bounded(tmp_path,monkeypatch):
    import time
    clock=[time.time()];attempts=[]
    monkeypatch.setattr('nekaise_loop.web_inventory.wall_time',lambda:clock[0])
    def handler(r):
        if r.url.path=='/robots.txt':return httpx.Response(404)
        return httpx.Response(200,text=page(r.url.path),headers={'content-type':'text/html'})
    session,calls=network(monkeypatch,tmp_path,handler);ctx=context(tmp_path);original=session.fetch
    def slow(url,*a,**kw):
        if url.endswith('/one'):
            attempts.append(url);raise TimeoutError('Per-request deadline')
        return original(url,*a,**kw)
    monkeypatch.setattr(session,'fetch',slow)
    req=source(seed_urls=['https://example.org/book/two'])
    first=ctx.artifacts.get(collect(ctx,req,session))
    assert len(first['pages'])==1 and first['pending']==['https://example.org/book/one']
    assert first['pages'][0]['url'].endswith('/two')
    collect(ctx,req,session);assert len(attempts)==1  # No early cooldown retry.
    for _ in range(2):
        clock[0]+=31;last=ctx.artifacts.get(collect(ctx,req,session))
    assert last['complete'] and last['failures'][-1]['retry_exhausted']
    assert len(attempts)==3


def test_rate_limit_is_deferred_and_retry_after_is_honored(tmp_path,monkeypatch):
    import time
    clock=[time.time()];limited=[True]
    monkeypatch.setattr('nekaise_loop.web_inventory.wall_time',lambda:clock[0])
    def handler(r):
        if r.url.path=='/robots.txt':return httpx.Response(404)
        if r.url.path=='/book/one' and limited[0]:return httpx.Response(429,headers={'retry-after':'60'})
        return httpx.Response(200,text=page(r.url.path),headers={'content-type':'text/html'})
    session,calls=network(monkeypatch,tmp_path,handler);ctx=context(tmp_path)
    req=source(seed_urls=['https://example.org/book/two'])
    first=ctx.artifacts.get(collect(ctx,req,session))
    assert not first['complete'] and first['next_retry_at']==clock[0]+60
    clock[0]+=31;collect(ctx,req,session)
    assert calls.count('https://example.org/book/one')==1
    clock[0]+=30;limited[0]=False
    last=ctx.artifacts.get(collect(ctx,req,session))
    assert last['complete'] and len(last['pages'])==2


def test_revised_extraction_preserves_failed_journal_and_reuses_raw_bytes(tmp_path,monkeypatch):
    from nekaise_loop import web_inventory as inventory
    def handler(r):
        if r.url.path=='/robots.txt':return httpx.Response(404)
        return httpx.Response(200,text='<aside><nav><div><nav>menu</div></aside>'+page('body'),
                              headers={'content-type':'text/html'})
    session,calls=network(monkeypatch,tmp_path,handler);ctx=context(tmp_path)
    original=inventory.extract_page;version=inventory.TrainingPageText.version
    def failed_extract(*args):raise ValueError('Research page contains insufficient readable text')
    monkeypatch.setattr(inventory.TrainingPageText,'version','readable_body_v1')
    monkeypatch.setattr(inventory,'extract_page',failed_extract)
    with pytest.raises(ValueError,match='no usable pages'):collect(ctx,source(),session)
    old=next((tmp_path/'curriculum/web').glob('*.json'))
    legacy=json.loads(old.read_text());legacy.pop('extractor');old.write_text(json.dumps(legacy))
    before=old.read_bytes();requests=list(calls)
    monkeypatch.setattr(inventory.TrainingPageText,'version',version)
    monkeypatch.setattr(inventory,'extract_page',original)
    result=ctx.artifacts.get(collect(ctx,source(),session))
    assert result['complete'] and len(result['pages'])==1 and calls==requests
    assert old.read_bytes()==before and len(list((tmp_path/'curriculum/web').glob('*.json')))==2
    text=ctx.artifacts.get(result['pages'][0]['artifact'])
    assert text['extractor']==version and 'menu' not in text['text'] and 'body' in text['text']
    assert ctx.artifacts.get(collect(ctx,source(),session))==result and calls==requests


@pytest.mark.parametrize('status',[200,403])
def test_extractor_upgrade_does_not_reset_successful_or_http_failed_collections(tmp_path,monkeypatch,status):
    from nekaise_loop import web_inventory as inventory
    def handler(r):
        if r.url.path=='/robots.txt':return httpx.Response(404)
        return httpx.Response(status,text=page('body'),headers={'content-type':'text/html'})
    session,calls=network(monkeypatch,tmp_path,handler);ctx=context(tmp_path)
    version=inventory.TrainingPageText.version
    monkeypatch.setattr(inventory.TrainingPageText,'version','readable_body_v1')
    if status==200:key=collect(ctx,source(),session)
    else:
        with pytest.raises(ValueError,match='no usable pages'):collect(ctx,source(),session)
    old=next((tmp_path/'curriculum/web').glob('*.json'));before=old.read_bytes();requests=list(calls)
    monkeypatch.setattr(inventory.TrainingPageText,'version',version)
    if status==200:assert collect(ctx,source(),session)==key
    else:
        with pytest.raises(ValueError,match='exhausted'):collect(ctx,source(),session)
    assert calls==requests and old.read_bytes()==before
    assert len(list((tmp_path/'curriculum/web').glob('*.json')))==1


def test_revised_extraction_failure_is_still_bounded(tmp_path,monkeypatch):
    from nekaise_loop import web_inventory as inventory
    def handler(r):
        if r.url.path=='/robots.txt':return httpx.Response(404)
        return httpx.Response(200,text='<main>short</main>',headers={'content-type':'text/html'})
    session,calls=network(monkeypatch,tmp_path,handler);ctx=context(tmp_path)
    version=inventory.TrainingPageText.version
    monkeypatch.setattr(inventory.TrainingPageText,'version','readable_body_v1')
    with pytest.raises(ValueError,match='no usable pages'):collect(ctx,source(),session)
    monkeypatch.setattr(inventory.TrainingPageText,'version',version)
    with pytest.raises(ValueError,match='no usable pages'):collect(ctx,source(),session)
    requests=list(calls)
    with pytest.raises(ValueError,match='exhausted'):collect(ctx,source(),session)
    assert calls==requests and len(list((tmp_path/'curriculum/web').glob('*.json')))==2
