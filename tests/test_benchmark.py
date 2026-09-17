import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from nekaise_loop.api import create_app
from nekaise_loop.benchmark import read_benchmark


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
