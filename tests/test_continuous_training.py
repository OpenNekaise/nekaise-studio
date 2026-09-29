"""Continuous scheduling tests use explicit fake teachers, authors and ML workers."""
import threading
import time

import pytest

from test_teaching_cycles import setup_cycle, CycleTeacher, CycleModel
from nekaise_loop.cycle_store import cycle_for


def setup_continuous(setup_loop, monkeypatch, **updates):
    return setup_cycle(setup_loop, monkeypatch,
        teaching_cycle={'policy':'continuous_v1', 'initial_blocks_per_cycle':1, 'blocks_per_cycle':1},
        **updates)


def test_cross_cycle_preparation_and_teacher_review_overlap_training(setup_loop, monkeypatch):
    _, service, campaign, engine, _ = setup_continuous(setup_loop, monkeypatch, rounds=3)
    second_trained, reviewed = threading.Event(), threading.Event()
    review = CycleTeacher.cycle_review
    def slow_review(self, brief):
        if not reviewed.is_set():
            assert second_trained.wait(10), 'Second window must train while first review waits'
            reviewed.set()
        return review(self, brief)
    monkeypatch.setattr(CycleTeacher, 'cycle_review', slow_review)
    def hook(model, parent):
        # Fake model directory belongs to the actual bound train stage.
        n = service.store.one('SELECT COUNT(*) AS n FROM curriculum_receipts')['n']
        if n == 0:
            deadline = time.monotonic()+10
            while time.monotonic() < deadline:
                ready = service.store.one("SELECT id FROM rounds WHERE number=2 AND status='prepared'")
                if ready:
                    break
                time.sleep(.01)
            assert ready, 'Next cycle must prepare before first cycle training finishes'
            assert not service.store.query('SELECT * FROM curriculum_receipts')
        elif n == 1:
            from nekaise_loop.checkpoint_lineage import latest_completed_snapshot
            snapshot=latest_completed_snapshot(service,campaign['id'])
            assert snapshot and snapshot[1]['number']==1
            assert service.store.one('SELECT status FROM rounds WHERE id=?',(snapshot[1]['id'],))['status']!='complete'
            second_trained.set()
    CycleModel.hook = hook
    engine.run(campaign['id'])
    result = service.store.campaign(campaign['id'])
    assert result['status'] == 'complete', result['error']
    assert second_trained.is_set() and reviewed.is_set()
    assert len(service.store.query('SELECT * FROM curriculum_receipts')) == 3
    assert len(service.store.query("SELECT * FROM teaching_cycles WHERE status='complete'")) == 3


def test_continuous_failure_drains_prepared_windows(setup_loop, monkeypatch):
    _, service, campaign, engine, _ = setup_continuous(setup_loop, monkeypatch, rounds=3, auto_recover=True)
    import nekaise_loop.cycle_stages as stages
    original = stages.select
    def fail(ctx):
        if ctx.round['number'] == 3:
            raise ValueError('Fixture producer incident')
        return original(ctx)
    # Routes retains function references.
    import nekaise_loop.cycle_engine as runner
    monkeypatch.setitem(runner.ROUTES, 'select', fail)
    engine.run(campaign['id'])
    result = service.store.campaign(campaign['id'])
    assert result['status'] == 'recovering', result['error']
    assert len(service.store.query('SELECT * FROM curriculum_receipts')) == 2
    recovery = service.store.one('SELECT * FROM recoveries')
    failed = service.store.one('SELECT * FROM stage_runs WHERE id=?', (recovery['stage_id'],))
    assert failed['stage'] == 'select' and 'Fixture producer' in failed['error']


def test_continuous_stop_does_not_credit_prepared_future(setup_loop, monkeypatch):
    _, service, campaign, engine, _ = setup_continuous(setup_loop, monkeypatch, rounds=3)
    stop = threading.Event()
    CycleModel.hook = lambda *args: stop.set()
    engine.run(campaign['id'], controls=stop.is_set)
    assert service.store.campaign(campaign['id'])['status'] == 'stopped'
    # Fake train finishes before cancellation is checked; exactly its saved exposure.
    assert len(service.store.query('SELECT * FROM curriculum_receipts')) == 1
    CycleModel.hook = None
    engine.run(campaign['id'])
    result = service.store.campaign(campaign['id'])
    assert result['status'] == 'complete', result['error']
    assert len(service.store.query('SELECT * FROM curriculum_receipts')) == 3


def test_continuous_review_failure_does_not_double_credit_on_retry(setup_loop, monkeypatch):
    _, service, campaign, engine, calls = setup_continuous(setup_loop, monkeypatch, rounds=2)
    CycleTeacher.review_failure = True
    engine.run(campaign['id'])
    assert service.store.campaign(campaign['id'])['status'] == 'failed'
    saved = len(service.store.query('SELECT * FROM curriculum_receipts'))
    assert saved >= 1
    CycleTeacher.review_failure = False
    engine.run(campaign['id'])
    result = service.store.campaign(campaign['id'])
    assert result['status'] == 'complete', result['error']
    assert len(service.store.query('SELECT * FROM curriculum_receipts')) == 2
    assert len(calls) == 6


def test_adam_bridge_rejects_changed_numerical_core(monkeypatch):
    from nekaise_loop import training_runtime as runtime
    from nekaise_loop.config import CampaignConfig
    from nekaise_loop.training import recipe_hash
    config = CampaignConfig(training_execution='resident_v1').model_dump()
    previous = {'training_code':runtime.BATCHED_PREDECESSOR, 'recipe_hash':recipe_hash(config)}
    assert runtime.optimizer_transition(previous, config)['policy'] == 'batched_to_continuous_v1'
    monkeypatch.setattr(runtime, 'numerical_core_hash', lambda:'unreviewed')
    with pytest.raises(ValueError, match='Unreviewed'):
        runtime.optimizer_transition(previous, config)


def test_diagnostic_windows_review_before_reopening_unchanged_frontier(setup_loop, monkeypatch):
    _, service, campaign, engine, calls = setup_continuous(setup_loop, monkeypatch, rounds=2)
    monkeypatch.setattr(CycleTeacher, 'diagnostic', True)
    engine.run(campaign['id'])
    current=service.store.campaign(campaign['id'])
    assert current['status']=='complete', current['error']
    assert not service.store.query('SELECT * FROM curriculum_receipts')
    assert not calls
    assert CycleTeacher.calls==['research','plan','review','plan','review']


def test_resident_protocol_waits_for_ack_and_discards_foreign_events(setup_loop, monkeypatch):
    import json
    from pathlib import Path
    from nekaise_loop.artifacts import atomic_write, canonical
    from nekaise_loop.config import CampaignConfig
    from nekaise_loop.processes import ProcessRunner
    from nekaise_loop.providers.local import LocalModel
    from nekaise_loop.resident_training import ResidentTrainer
    settings, service, campaign, engine, _ = setup_continuous(setup_loop, monkeypatch, rounds=1)
    def transport(runner, command, *, cwd, on_message, **kwargs):
        seen=None
        while not runner.cancelled():
            path=cwd/'command.json'
            if not path.exists():
                time.sleep(.005);continue
            request=json.loads(path.read_text())
            if request['id']==seen:
                time.sleep(.005);continue
            seen=request['id']
            atomic_write(cwd/(seen+'.result.json'),canonical({'request_id':seen,'result':{'checkpoint':'fixture-saved'}}))
            on_message({'type':'resident_event','data':{'request_id':'previous-request','event':{'type':'metric','data':{'loss':999}}}})
            time.sleep(.1)
            on_message({'type':'resident_event','data':{'request_id':seen,'event':{'type':'metric','data':{'loss':1}}}})
            on_message({'type':'resident_complete','data':{'request_id':seen}})
        return ''
    monkeypatch.setattr(ProcessRunner,'run',transport)
    trainer=ResidentTrainer(engine,campaign['id'])
    local=LocalModel(CampaignConfig.model_validate(campaign['config']),settings,ProcessRunner(service.store,None),settings.workspace/'request')
    received=[]
    try:
        for _ in range(2):
            start=time.monotonic()
            trainer.request(local,'train',{},received.append)
            assert time.monotonic()-start>=.09
    finally:
        trainer.close()
    assert [r['data']['loss'] for r in received]==[1,1]
    assert service.store.one('SELECT status FROM trainer_sessions')['status']=='closed'


def test_reset_is_consumed_by_save_even_while_review_is_pending(setup_loop,monkeypatch):
    _,service,campaign,engine,_=setup_continuous(setup_loop,monkeypatch,rounds=2,inherit_optimizer=False)
    second=threading.Event()
    review=CycleTeacher.cycle_review
    def slow(self,brief):
        assert second.wait(10)
        return review(self,brief)
    monkeypatch.setattr(CycleTeacher,'cycle_review',slow)
    seen=[]
    def hook(model,parent):
        seen.append(model.config.inherit_optimizer)
        if len(seen)==2:
            second.set()
    CycleModel.hook=hook
    engine.run(campaign['id'])
    result=service.store.campaign(campaign['id'])
    assert result['status']=='complete',result['error']
    assert seen==[False,True]


def test_multiblock_continuous_cycles_bind_last_predecessor(setup_loop,monkeypatch):
    _,service,campaign,engine,_=setup_cycle(setup_loop,monkeypatch,rounds=4,
        teaching_cycle={'policy':'continuous_v1','initial_blocks_per_cycle':2,'blocks_per_cycle':2})
    engine.run(campaign['id'])
    result=service.store.campaign(campaign['id'])
    assert result['status']=='complete',result['error']
    rows=service.store.query('SELECT * FROM rounds WHERE campaign_id=? ORDER BY number',(campaign['id'],))
    assert len(rows)==4
    for previous,current in zip(rows,rows[1:]):
        assert current['model_before']==previous['checkpoint']
    assert service.store.one('SELECT predecessor_round_id FROM teaching_blocks WHERE round_id=?',(rows[2]['id'],))['predecessor_round_id']==rows[1]['id']
    assert len(service.store.query('SELECT * FROM curriculum_receipts'))==4


def test_due_review_does_not_open_a_cycle_after_waiting_for_slot(setup_loop,monkeypatch):
    import nekaise_loop.history as history
    _,service,campaign,engine,calls=setup_continuous(setup_loop,monkeypatch,rounds=5,auto_recover=True,manage_history=True)
    due=threading.Event()
    monkeypatch.setattr(history,'review_due',lambda store:(due.set() or True))
    original=CycleTeacher.cycle_review
    def review(self,brief):
        assert due.wait(10)
        time.sleep(.1)
        return original(self,brief)
    monkeypatch.setattr(CycleTeacher,'cycle_review',review)
    def hook(model,parent):
        if service.store.one('SELECT COUNT(*) AS n FROM curriculum_receipts')['n']:
            return
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            if service.store.one("SELECT id FROM rounds WHERE number=2 AND status='prepared'"):
                time.sleep(.1)
                return
            time.sleep(.01)
        raise AssertionError('Second cycle not prepared')
    CycleModel.hook=hook
    engine.run(campaign['id'])
    assert len(service.store.query('SELECT * FROM teaching_cycles'))==2
    assert len(calls)==6
    assert service.store.one('SELECT kind FROM recoveries')['kind']=='history_review'


def test_concurrent_teacher_calls_cannot_overspend_last_allowance(setup_loop,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from nekaise_loop.providers.teacher import CliTeacher
    from nekaise_loop.cycle_types import CycleResearch
    from nekaise_loop.config import CampaignConfig
    from nekaise_loop.storage import now
    from nekaise_loop.failures import TeacherUnavailable
    settings,service,campaign,engine,_=setup_continuous(setup_loop,monkeypatch,rounds=2)
    config=CampaignConfig.model_validate({**campaign['config'],'max_teacher_calls':1})
    class NoNetwork:
        def run(self,*a,**kw):
            raise RuntimeError('Fixture transport deliberately stops before network')
    barrier=threading.Barrier(2)
    def call(i):
        rid=f'quota-fixture-{i}'
        service.store.execute("INSERT INTO rounds(id,campaign_id,number,status,model_before,created_at,updated_at) VALUES(?,?,?,'ready',?,?,?)",
            (rid,campaign['id'],i+1,campaign['config']['student_model'],now(),now()))
        teacher=CliTeacher(config,settings,service.store,campaign['id'],rid,NoNetwork(),settings.workspace/rid)
        barrier.wait()
        try:
            teacher.request('cycle_research',{'units':[]},CycleResearch)
        except Exception as exc:
            return exc
    with ThreadPoolExecutor(max_workers=2) as pool:
        failures=list(pool.map(call,range(2)))
    assert sum(isinstance(e,TeacherUnavailable) and e.kind=='budget' for e in failures)==1
    assert service.store.one('SELECT COUNT(*) AS n FROM teacher_calls')['n']==1
