import json
import hashlib
import copy
from datetime import datetime, timezone, timedelta

from fastapi.testclient import TestClient
from nekaise_loop.api import create_app
from nekaise_loop.benchmark import read_benchmark, read_benchmark_history


def projection(cid):
    stamp=datetime.now(timezone.utc).isoformat()
    point={'model_id':'a'*64,'root_id':'b'*64,'round_id':'r1','round_number':1,'retained_tokens':10,
        'checkpoint_at':stamp,'evaluated_at':stamp,'run_id':'fixture-only','protocol_id':'c'*64,
        'protocol':'chat-1','release_id':'d'*64,'release_name':'fixture-only','n':4,'correct':1,
        'score':.25,'invalid':1,'budget_exhausted':1,'numerical':.5,'choice':0.,'delta':None,
        'delta_ci95':None,'clusters':None,'interval_reliable':False,'gold':'PRIVATE'}
    return {'schema_version':1,'campaign_id':cid,'generated_at':stamp,'heartbeat_at':stamp,'status':'ok',
        'latest':point,'points':[point],'issues':[],'baseline_ready':True,'target_model_id':'a'*64,
        'target_round':1,'lag_tokens':0,'stale':False,'interval_seconds':900,'questions':['PRIVATE']}


def write_projection(root,cid,value):
    root.mkdir(exist_ok=True)
    (root/(cid+'.json')).write_text(json.dumps(value))
    (root.parent/'heartbeat.json').write_text(json.dumps({'at':value['generated_at']}))


def test_projection_whitelist_and_zero_is_distinct_from_missing(tmp_path):
    root=tmp_path/'projection';value=projection('c1');write_projection(root,'c1',value)
    got=read_benchmark('c1',root)
    assert got['latest']['score']==.25 and not got['service_stale']
    assert 'PRIVATE' not in json.dumps(got)
    assert read_benchmark('absent',root)['latest'] is None
    value['latest']['score']=0.;value['latest']['correct']=0
    write_projection(root,'c1',value)
    assert read_benchmark('c1',root)['latest']['score']==0


def test_reject_mismatch_corrupt_oversize_and_symlink_escape(tmp_path):
    root=tmp_path/'projection';value=projection('other');write_projection(root,'c1',value)
    assert read_benchmark('c1',root)['status']=='unavailable'
    (root/'c1.json').write_text('x'*(1024*1024+1))
    assert read_benchmark('c1',root)['latest'] is None
    (root/'c1.json').unlink();(root/'c1.json').symlink_to(tmp_path/'heartbeat.json')
    assert read_benchmark('c1',root)['status']=='unavailable'
    assert read_benchmark('../heartbeat',root)['latest'] is None


def test_heartbeat_respects_observer_poll_interval(tmp_path):
    root = tmp_path/'projection'
    write_projection(root, 'c1', projection('c1'))
    old = (datetime.now(timezone.utc)-timedelta(seconds=600)).isoformat()
    pulse = root.parent/'heartbeat.json'
    pulse.write_text(json.dumps({'at': old, 'poll_seconds': 300}))
    assert not read_benchmark('c1', root)['service_stale']
    pulse.write_text(json.dumps({'at': old, 'poll_seconds': 60}))
    assert read_benchmark('c1', root)['service_stale']
    pulse.write_text(json.dumps({'at': old, 'poll_seconds': True}))
    assert read_benchmark('c1', root)['status'] == 'unavailable'


def test_projection_endpoint_never_enqueues_work_or_enters_history(setup_loop,tmp_path,monkeypatch):
    settings,service,campaign,engine=setup_loop
    root=tmp_path/'projection';write_projection(root,campaign['id'],projection(campaign['id']))
    monkeypatch.setenv('NEKAISE_BENCH_PROJECTION_DIR',str(root))
    monkeypatch.setattr('nekaise_loop.service.Service.ensure_worker',lambda *_:None)
    before={t:service.store.query(f'SELECT * FROM {t}') for t in ('events','records','metrics','actions')}
    with TestClient(create_app(settings)) as client:
        response=client.get(f"/api/campaigns/{campaign['id']}/benchmark")
        assert response.status_code==200 and response.headers['cache-control']=='no-store'
        assert response.json()['latest']['correct']==1
        assert 'PRIVATE' not in response.text
        snap=client.get(f"/api/campaigns/{campaign['id']}").json()
        assert 'benchmark' not in snap
    assert before=={t:service.store.query(f'SELECT * FROM {t}') for t in before}


def seal(root, category, data):
    raw = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()+b'\n'
    key = hashlib.sha256(raw).hexdigest()
    path = root/'history'/category/(key+'.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return key


def history(root, cid='c1', count=241):
    value = projection(cid)
    points = []
    for i in range(count):
        points.append({**value['latest'], 'model_id': f'{i+1:064x}', 'retained_tokens': i,
                       'round_number': i, 'observation_kind': 'baseline' if i == 0 else 'sample'})
    identity = {k: points[0][k] for k in ('root_id', 'protocol_id', 'release_id')}
    cohort = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    pages = [seal(root, 'pages', {'schema_version': 2, 'cohort_id': cohort,
             'offset': offset, 'points': points[offset:offset+100], 'answers': 'PRIVATE'})
             for offset in range(0, count, 100)]
    manifest = {'schema_version': 2, 'campaign_id': cid, 'cohort_id': cohort, **identity,
                'count': count, 'page_size': 100, 'pages': pages, 'questions': 'PRIVATE'}
    key = seal(root, 'manifests', manifest)
    return key, manifest, points


def test_complete_history_is_sealed_paginated_scoped_and_aggregate_only(tmp_path):
    root = tmp_path/'projection'
    key, manifest, points = history(root)
    pages = [read_benchmark_history('c1', key, i, root) for i in range(3)]
    assert [len(p['points']) for p in pages] == [100, 100, 41]
    assert [p['model_id'] for page in pages for p in page['points']] == [p['model_id'] for p in points]
    assert all('PRIVATE' not in json.dumps(p) for p in pages)
    # A new observation cannot invalidate a user's pinned browsing snapshot.
    next_key, _, _ = history(root, count=242)
    assert next_key != key
    assert read_benchmark_history('c1', key, 2, root)['count'] == 241
    for cid, requested_key, page in [('other', key, 0), ('../c1', key, 0), ('c1', '../secret', 0),
                                     ('c1', key, -1), ('c1', key, 3), ('c1', key, True)]:
        assert read_benchmark_history(cid, requested_key, page, root)['status'] == 'unavailable'
    (root/'history/pages'/f"{manifest['pages'][0]}.json").write_text('{}')
    assert read_benchmark_history('c1', key, 0, root)['status'] == 'unavailable'


def test_history_rejects_sealed_wrong_cohort_offset_counts_and_symlink_escape(tmp_path):
    root = tmp_path/'projection'
    key, manifest, points = history(root, count=2)
    original = {'schema_version': 2, 'cohort_id': manifest['cohort_id'], 'offset': 0, 'points': points}
    bad_pages = [dict(original, offset=100), dict(original, points=points[:1]),
                 dict(original, cohort_id='f'*64), dict(original, points=list(reversed(points)))]
    other = copy.deepcopy(original)
    other['points'][0]['release_id'] = 'f'*64
    bad_pages.append(other)
    for bad in bad_pages:
        bad_key = seal(root, 'manifests', {**manifest, 'pages': [seal(root, 'pages', bad)]})
        assert read_benchmark_history('c1', bad_key, 0, root)['status'] == 'unavailable'
    path = root/'history/pages'/f"{manifest['pages'][0]}.json"
    external = tmp_path/'external.json'
    external.write_bytes(path.read_bytes())
    path.unlink();path.symlink_to(external)
    assert read_benchmark_history('c1', key, 0, root)['status'] == 'unavailable'


def test_v2_comparison_counts_and_private_fields_are_validated(tmp_path):
    root = tmp_path/'projection'
    value = projection('c1')
    pair = {'kind': 'baseline', 'reference_model_id': 'b'*64, 'reference_round_id': None,
            'reference_tokens': 0, 'reference_score': 0., 'n': 4, 'delta': .25,
            'delta_ci95': [0., .5], 'gained': 1, 'lost': 0, 'invalid_to_correct': 1,
            'correct_to_invalid': 0, 'clusters': 1, 'interval_reliable': False,
            'interpretation': 'descriptive', 'gold': 'PRIVATE'}
    value.update(schema_version=2, comparisons={'baseline': pair, 'milestone': None},
                 comparison_reasons={'baseline': None, 'milestone': 'no_earlier_milestone'})
    value['latest'].update(numerical_n=2, numerical_correct=1, choice_n=2, choice_correct=0)
    write_projection(root, 'c1', value)
    result = read_benchmark('c1', root)
    assert result['comparisons']['baseline']['gained'] == 1
    assert 'PRIVATE' not in json.dumps(result)
    pair['gained'] = 2
    write_projection(root, 'c1', value)
    assert read_benchmark('c1', root)['status'] == 'unavailable'


def test_history_endpoint_does_not_mutate_training_or_enter_report(setup_loop, tmp_path, monkeypatch):
    settings, service, campaign, engine = setup_loop
    root = tmp_path/'projection'
    key, _, _ = history(root, campaign['id'])
    monkeypatch.setenv('NEKAISE_BENCH_PROJECTION_DIR', str(root))
    monkeypatch.setattr('nekaise_loop.service.Service.ensure_worker', lambda *_: None)
    tables = ('events', 'records', 'metrics', 'actions', 'recoveries')
    before = {t: service.store.query(f'SELECT * FROM {t}') for t in tables}
    with TestClient(create_app(settings)) as client:
        response = client.get(f"/api/campaigns/{campaign['id']}/benchmark/history", params={'history': key, 'page': 2})
        assert response.status_code == 200
        assert response.headers['cache-control'] == 'no-store'
        assert len(response.json()['points']) == 41 and 'PRIVATE' not in response.text
        assert client.get(f"/api/campaigns/{campaign['id']}/benchmark/history", params={'history': '../x', 'page': 0}).status_code == 422
        assert 'comparisons' not in client.get('/api/reports').text
    assert before == {t: service.store.query(f'SELECT * FROM {t}') for t in tables}
