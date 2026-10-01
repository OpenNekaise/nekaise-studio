"""Teacher-selected persistent source catalogs and bounded, reusable web inventory.

Acquisition runs only in the worker's preparation thread, independent of the GPU
consumer. It does not create training windows, advance coverage, or choose a mix.
"""
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
from urllib.parse import urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree

import httpx

from .artifacts import atomic_write, canonical, digest
from .curriculum_types import WebTrainingSelection
from .curriculum_research import TrainingPageText
from .processes import Cancelled
from .storage import now
from .web_http import WebSession, extract_page, normalize_url, retry_delay

_JOURNAL_LOCKS = {}
_JOURNAL_GUARD = threading.Lock()
ERRORS = (ValueError, OSError, httpx.HTTPError)
wall_time = time.time


def enabled(ctx):
    loop=ctx.config.curriculum_loop
    return bool(loop and loop.web_training and loop.web_crawl_policy=='inventory_v1')


def catalog_root(ctx):
    return ctx.engine.settings.workspace/'curriculum/source-catalog'/ctx.config.curriculum_loop.namespace


def scope_url(url, seed, prefix=''):
    """Map a same-host HTTP link to selected HTTPS before verifying actual bytes."""
    candidate=normalize_url(url)
    u,p=urlsplit(candidate),urlsplit(prefix or seed)
    if u.netloc==p.netloc and u.scheme=='http' and p.scheme=='https':
        candidate=urlunsplit(('https',u.netloc,u.path,u.query,''));u=urlsplit(candidate)
    if (u.scheme,u.netloc)!=(p.scheme,p.netloc):
        raise ValueError('URL is outside the Teacher-selected origin')
    if prefix:
        path=p.path.rstrip('/')
        if u.query or not (u.path==path or u.path.startswith(path+'/')):
            raise ValueError('URL is outside the Teacher-selected path or uses unselected query parameters')
    return candidate


def admitted_location(location, seed, prefix):
    try:
        # Redirect targets themselves must match; do not silently downgrade HTTP.
        return scope_url(location,seed,prefix)==normalize_url(location)
    except ValueError:
        return False


def _journal_path(ctx, request):
    identity={'contract':'inventory_v1','request':request,
              'policy':ctx.config.curriculum_loop.web_training_policy}
    root=ctx.engine.settings.workspace/'curriculum/web'
    path=root/f'{digest(identity)}.json'
    if path.exists():
        prior=json.loads(path.read_text())
        failures=prior.get('failures',[])
        if (prior.get('extractor','readable_body_v1') != TrainingPageText.version
                and prior['complete'] and not prior['pages'] and failures
                and all(f.get('error')=='Research page contains insufficient readable text'
                        for f in failures)):
            # Preserve the exhausted extraction receipt. A changed extractor may
            # reconsider its cached bytes once under its own bounded journal;
            # successful collections and HTTP/access failures are not reset.
            path=root/f'{digest({**identity, "extractor":TrainingPageText.version})}.json'
    return path


def collect(ctx, request, session=None, *, deadline=None):
    if ctx.cancelled():
        raise Cancelled('Web collection cancelled')
    if ctx.config.curriculum_loop.web_training_policy!='teacher_selected_v1':
        raise ValueError('Inventory acquisition requires the operator teacher-selected source policy')
    path=_journal_path(ctx,request)
    with _JOURNAL_GUARD:
        lock=_JOURNAL_LOCKS.setdefault(str(path),threading.Lock())
    while not lock.acquire(timeout=.1):
        if ctx.cancelled():
            raise Cancelled('Cancelled waiting for web collection journal')
        if deadline is not None and time.monotonic()>=deadline:
            raise TimeoutError('Collection journal wait exceeded acquisition deadline')
    try:
        return _collect(ctx,request,path,session or WebSession(ctx.engine.settings.workspace,ctx.cancelled),deadline)
    finally:
        lock.release()


def _collect(ctx, request, path, session, deadline):
    selection=WebTrainingSelection.model_validate(request.get('training') or {}).model_dump()
    seed=normalize_url(request['url']);original_seed=seed;prefix=selection['collection_prefix']
    if prefix:
        prefix=normalize_url(prefix)
        if urlsplit(prefix).query:
            raise ValueError('Collection prefix cannot have query parameters')
    if not prefix and (selection['max_pages']!=1 or selection['seed_urls'] or selection['sitemap_urls']):
        raise ValueError('Multi-page/index/sitemap acquisition requires a selected path prefix')
    seed=scope_url(seed,seed,prefix)
    seeds=[scope_url(u,seed,prefix) for u in selection['seed_urls']]
    sitemaps=list(dict.fromkeys(scope_url(u,seed) for u in selection['sitemap_urls']))
    if path.exists():
        journal=json.loads(path.read_text())
    else:
        journal={'format':'web_training_collection_v2','acquisition_policy':'inventory_v1',
            'extractor':TrainingPageText.version,
            'permission':selection,'admission_policy':'teacher_selected_v1','license_checked':False,
            'license_evidence_artifact':None,'pages':[],'pending':list(dict.fromkeys([seed,*seeds])),
            'visited':[],'failures':[],'discovery_pending':sitemaps,'discovery_visited':[],
            'url_mappings':([{'discovered_url':original_seed,'fetch_url':seed,
                'rule':'same_host_https_candidate; destination_and_robots_checked_on_fetch'}] if seed!=original_seed else []),
            'collected_chars':0,'created_at':now(),'complete':False}
        atomic_write(path,canonical(journal))
    if journal['complete']:
        if not journal['pages']:
            raise ValueError('Source acquisition exhausted its bounded retries; repair the source plan; inspect '+str(path))
        return ctx.artifacts.put(journal)
    journal.setdefault('retry_attempts',{})
    journal.setdefault('retry_after',{})
    deadline=min(deadline or float('inf'),time.monotonic()+selection['max_seconds'])
    character_cap=False
    attempted=set()
    def persist():
        atomic_write(path,canonical(journal))
    def ready(phase, pending):
        return next((u for u in pending if phase+':'+u not in attempted
                     and journal['retry_after'].get(phase+':'+u,0)<=wall_time()),None)
    def finish(phase, url):
        pending,visited=('pending','visited') if phase=='page' else ('discovery_pending','discovery_visited')
        journal[pending].remove(url);journal[visited].append(url)
        journal['retry_after'].pop(phase+':'+url,None)
    def failure(phase,url,exc):
        key=phase+':'+url;attempted.add(key)
        status=exc.response.status_code if isinstance(exc,httpx.HTTPStatusError) else None
        transient=isinstance(exc,(TimeoutError,httpx.TransportError)) or status==429 or (status is not None and status>=500)
        receipt={'url':url,'phase':phase,'error':str(exc)[:500],'transient':transient}
        if transient:
            count=journal['retry_attempts'].get(key,0)+1;journal['retry_attempts'][key]=count
            receipt['acquisition_attempts']=count
            if count<3:
                header=exc.response.headers.get('retry-after') if isinstance(exc,httpx.HTTPStatusError) else None
                journal['retry_after'][key]=wall_time()+max(30,retry_delay(header,count))
                receipt['retry_after']=journal['retry_after'][key]
            else:
                receipt['retry_exhausted']=True;finish(phase,url)
        else:
            finish(phase,url)
        journal['failures'].append(receipt)
    def add_page_link(link,base):
        try:
            absolute=normalize_url(urljoin(base,link))
            target=scope_url(absolute,seed,prefix)
        except ValueError:
            return
        if target in set(journal['pending'])|set(journal['visited']):
            return
        if len(journal['pending'])+len(journal['visited'])>=selection['max_pages']:
            return
        if absolute!=target:
            journal['url_mappings'].append({'discovered_url':absolute,'fetch_url':target,
                'rule':'same_host_https_candidate; destination_and_robots_checked_on_fetch'})
        journal['pending'].append(target)
    try:
        # Up to eight explicitly rooted sitemap documents, including nested indexes.
        while journal['discovery_pending'] and len(journal['discovery_visited'])<8 and time.monotonic()<deadline:
            if ctx.cancelled():
                raise Cancelled('Sitemap acquisition cancelled')
            url=ready('sitemap',journal['discovery_pending'])
            if url is None:break
            try:
                raw,_=session.fetch(url,lambda u:admitted_location(u,seed,''),deadline=deadline)
                xml=base64.b64decode(raw['body_base64'])
                if b'<!DOCTYPE' in xml.upper() or b'<!ENTITY' in xml.upper():
                    raise ValueError('Sitemap declarations/entities are unsupported')
                root=ElementTree.fromstring(xml)
                index=root.tag.rsplit('}',1)[-1]=='sitemapindex'
                if root.tag.rsplit('}',1)[-1] not in {'sitemapindex','urlset'}:
                    raise ValueError('Not an XML sitemap/index')
                for entry in root:
                    loc=next((n.text for n in entry if n.tag.rsplit('}',1)[-1]=='loc'),None)
                    if not loc:
                        continue
                    if index:
                        try:target=scope_url(loc.strip(),seed)
                        except ValueError:continue
                        if (target not in journal['discovery_pending']+journal['discovery_visited']
                                and len(journal['discovery_pending'])+len(journal['discovery_visited'])<8):
                            journal['discovery_pending'].append(target)
                    else:
                        add_page_link(loc.strip(),raw['url'])
                finish('sitemap',url)
            except ElementTree.ParseError as exc:
                failure('sitemap',url,ValueError(str(exc)))
            except ERRORS as exc:
                failure('sitemap',url,exc)
            persist()
        while (journal['pending'] and len(journal['visited'])<selection['max_pages']
               and journal['collected_chars']<64_000_000 and time.monotonic()<deadline):
            if ctx.cancelled():
                raise Cancelled('Web collection cancelled')
            page=ready('page',journal['pending'])
            if page is None:break
            try:
                raw,robots=session.fetch(page,lambda u:admitted_location(u,seed,prefix),deadline=deadline)
                source=extract_page(raw,{**request,'url':page})
                links=source.pop('links')
                robots_key=ctx.artifacts.put(robots)
                raw_key=ctx.artifacts.put(raw)
                source.update(training_eligible=True,training_permission=selection,
                    admission_policy='teacher_selected_v1',license_checked=False,license_evidence_artifact=None,
                    robots_artifact=robots_key,raw_response_artifact=raw_key,selection_reason=request['purpose'])
                key=ctx.artifacts.put(source)
                if source['text_sha256'] not in {p['text_sha256'] for p in journal['pages']}:
                    if journal['collected_chars']+len(source['text'])>64_000_000:
                        character_cap=True
                        break
                    journal['pages'].append({'artifact':key,'text_sha256':source['text_sha256'],
                        'url':source['url'],'chars':len(source['text'])})
                    journal['collected_chars']+=len(source['text'])
                finish('page',page)
                if prefix:
                    for link in links:
                        add_page_link(link,source['url'])
            except Cancelled:
                raise
            except ERRORS as exc:
                failure('page',page,exc)
            persist()
    finally:
        persist()
    exhausted=not journal['pending'] and (not journal['discovery_pending'] or len(journal['discovery_visited'])>=8)
    capped=character_cap or len(journal['visited'])>=selection['max_pages'] or journal['collected_chars']>=64_000_000
    next_retry=min(journal['retry_after'].values(),default=None)
    journal.update(complete=exhausted or capped,updated_at=now(),next_retry_at=next_retry,
        bounded_by='frontier_exhausted' if exhausted else 'page_or_character_limit' if capped else 'cooldown' if next_retry and next_retry>wall_time() else 'time_slice',
        basis='Collected inventory only; durable trainer saves alone advance coverage')
    persist()
    if not journal['pages']:
        raise ValueError('Selected collection has no usable pages; inspect '+str(path))
    return ctx.artifacts.put(journal)


def _source_id(request):
    return digest({'contract':'source_catalog_v1','request':request})


def register(ctx, request, collection, unit_id):
    sid=_source_id(request)
    path=catalog_root(ctx)/(sid+'.json')
    with _JOURNAL_GUARD:
        lock=_JOURNAL_LOCKS.setdefault(str(path),threading.Lock())
    with lock:
        prior=json.loads(path.read_text()) if path.exists() else {}
        entry={'id':sid,'request':request,'collection_artifact':collection,
            'unit_id':prior.get('unit_id',unit_id),'unit_ids':sorted(set(prior.get('unit_ids',[]))|{unit_id}),
            'namespace':ctx.config.curriculum_loop.namespace,'created_at':prior.get('created_at',now()),'updated_at':now(),
            'authority':'Teacher-nominated source; later reuse requires explicit source ID'}
        atomic_write(path,canonical(entry))
    return sid


def fresh_supply(ctx, references, positions=None):
    positions=positions or {}; seen=set();result=[]
    for key in references.get('training_collections',[]):
        c=ctx.artifacts.get(key);fresh=0;distinct=0
        for page in c['pages']:
            sha=page['text_sha256']
            if sha in seen:continue
            seen.add(sha);distinct+=1
            fresh+=max(0,page['chars']-positions.get(sha,0))
        result.append({'artifact':key,'pages':len(c['pages']),'unique_pages_in_unit':distinct,
            'source_chars':c['collected_chars'],'fresh_chars_after_frontier':fresh,
            'collection_complete':c.get('complete',True),'pending_pages':len(c.get('pending',[])),
            'acquisition_failures':len(c.get('failures',[])),'next_retry_at':c.get('next_retry_at'),
            'admission_policy':c.get('admission_policy'),'license_checked':c.get('license_checked',True),
            'basis':'Exact characters after supplied coverage frontier; not tokenizer counts or completed training. Shared pages across units may overlap.'})
    return result


def catalog(ctx, unit_ids, positions=None):
    entries=[]
    for path in sorted(catalog_root(ctx).glob('*.json')):
        entry=json.loads(path.read_text())
        supply=fresh_supply(ctx,{'training_collections':[entry['collection_artifact']]},positions)
        entries.append({'id':entry['id'],'title':entry['request']['title'],'url':entry['request']['url'],
            'purpose':entry['request']['purpose'],'original_unit_id':entry['unit_id'],
            'selected_for_unit_ids':entry.get('unit_ids',[entry['unit_id']]),
            'fresh_chars_after_frontier':sum(s['fresh_chars_after_frontier'] for s in supply),
            'complete':all(s['collection_complete'] for s in supply),'entry_artifact':ctx.artifacts.put(entry)})
    # Bounded inline context, complete catalog always accessible through its artifact.
    relevant=[e for e in entries if set(e['selected_for_unit_ids']) & set(unit_ids)]
    return {'total_sources':len(entries),'artifact':ctx.artifacts.put({'sources':entries}),
        'matching_units':relevant[:16], 'instruction':'All registered sources remain accessible in the catalog artifact. Select reuse_source_ids explicitly; no automatic topic substitution.'}


def research_sources(ctx, plan, unit_id):
    session=WebSession(ctx.engine.settings.workspace,ctx.cancelled)
    deadline=time.monotonic()+300  # Shared across sources and declared alternatives.
    requests=list(plan.get('sources',[]))
    for sid in plan.get('reuse_source_ids',[]):
        if len(sid)!=64 or any(c not in '0123456789abcdef' for c in sid):
            raise ValueError('Invalid source catalog ID')
        path=catalog_root(ctx)/(sid+'.json')
        if not path.exists():raise ValueError('Teacher selected an unknown source catalog ID: '+sid)
        requests.append(json.loads(path.read_text())['request'])
    def acquire(request):
        failures=[]
        locations=[{k:v for k,v in request.items() if k!='alternatives'},
                   *[{**a,'purpose':request['purpose']} for a in request.get('alternatives',[])]]
        for location in locations:
            try:
                key=collect(ctx,location,session,deadline=deadline)
                c=ctx.artifacts.get(key)
                source=ctx.artifacts.get(c['pages'][0]['artifact'])
                reference={**source,'text':source['text'][:12000],'span_length':min(12000,len(source['text'])),
                    'excerpt_truncated':len(source['text'])>12000,'license':'research_reference_only',
                    'training_eligible':False,'reference_from_collection':key}
                sid=register(ctx,request,key,unit_id)
                return {'source':reference,'collection':key,'source_id':sid,'failures':failures,
                    'selection':{'requested_url':request['url'],'used_url':location['url'],
                                 'used_alternative':location['url']!=request['url']}}
            except Cancelled:raise
            except ERRORS as exc:
                failures.append({'url':location['url'],'error':str(exc)[:1000]})
        return {'failures':failures}
    # Four collections across hosts; WebSession additionally caps all HTTP requests at four.
    with ThreadPoolExecutor(max_workers=4,thread_name_prefix='source-inventory') as pool:
        acquired=list(pool.map(acquire,requests))
    return {'plan':plan,'unit_id':unit_id,'sources':[a['source'] for a in acquired if 'source' in a],
        'training_collections':list(dict.fromkeys(a['collection'] for a in acquired if 'collection' in a)),
        'source_catalog_ids':[a['source_id'] for a in acquired if 'source_id' in a],
        'source_selections':[a['selection'] for a in acquired if 'selection' in a],
        'fetch_failures':[e for a in acquired for e in a['failures']],
        'training_admission_failures':[], 'acquisition_policy':'inventory_v1',
        'web_training_policy':'teacher_selected_v1',
        'basis':'Teacher-selected reusable source inventory; reference excerpts and training prose share fetched bytes'}
