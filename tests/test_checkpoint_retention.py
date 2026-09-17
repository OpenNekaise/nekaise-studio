import json
from pathlib import Path
from unittest.mock import patch

import pytest

from nekaise_loop.artifacts import verify_checkpoint
from nekaise_loop.checkpoint_retention import apply, inventory, sha, storage_status
from nekaise_loop.storage import now


@pytest.fixture
def checkpoints(setup_loop):
    settings, service, campaign, engine = setup_loop
    engine.run(campaign['id'])
    rows = service.store.query("SELECT s.id,s.artifact FROM stage_runs s JOIN rounds r ON r.id=s.round_id WHERE r.campaign_id=? AND s.stage='train' ORDER BY r.number", (campaign['id'],))
    artifacts = []
    for row in rows:
        result = service.artifacts.get(row['artifact']);root = Path(result['checkpoint'])
        state = root/'training_state.pt';state.write_bytes(b'FIXTURE OPTIMIZER STATE')
        result['manifest']['files']['training_state.pt'] = sha(state)
        (root/'checkpoint.json').write_text(json.dumps(result['manifest']))
        key = service.artifacts.put(result)
        service.store.execute('UPDATE stage_runs SET artifact=? WHERE id=?', (key,row['id']))
        artifacts.append(result)
    recovery = service.store.recover(campaign['id'], 'storage_pressure', 'Fixture capacity review')
    return settings, service, campaign, engine, artifacts, recovery


def choose(item, disposition):
    return {'path':item['path'], 'manifest_sha256':item['manifest_sha256'], 'disposition':disposition,
            'summary':'Retain fixture training outcome and immutable provenance.', 'reason':'Explicit fixture orchestrator retention decision.'}


def test_optimizer_retirement_preserves_inference_and_rejects_resume(checkpoints):
    settings, service, _, _, artifacts, recovery = checkpoints
    first = inventory(service)['checkpoints'][0];root = Path(artifacts[0]['checkpoint'])
    original = (root/'checkpoint.json').read_bytes()
    result = apply(service, recovery, [choose(first, 'weights')])
    assert result[0]['bytes_released'] > 0
    assert not (root/'training_state.pt').exists()
    assert (root/'checkpoint.json').read_bytes() == original
    verify_checkpoint(artifacts[0], require_optimizer=False)
    with pytest.raises(ValueError, match='intentionally retired'):
        verify_checkpoint(artifacts[0])
    assert apply(service, recovery, [choose(first,'weights')]) == result


def test_summary_preserves_metadata_but_cannot_infer(checkpoints):
    _, service, _, _, artifacts, recovery = checkpoints
    first = inventory(service)['checkpoints'][0]
    apply(service, recovery, [choose(first, 'summary')])
    root = Path(artifacts[0]['checkpoint'])
    assert (root/'checkpoint.json').exists()
    assert not (root/'model.safetensors').exists()
    with pytest.raises(ValueError, match='intentionally retired'):
        verify_checkpoint(artifacts[0], require_optimizer=False)
    assert inventory(service)['checkpoints'][0]['retention']['disposition'] == 'summary'


def test_all_choices_validated_before_deletion_and_tip_protected(checkpoints):
    _, service, _, _, artifacts, recovery = checkpoints
    items = inventory(service)['checkpoints']
    with pytest.raises(ValueError, match='needed for resumption'):
        apply(service, recovery, [choose(items[0],'weights'),choose(items[-1],'summary')])
    assert (Path(artifacts[0]['checkpoint'])/'training_state.pt').exists()


def test_corruption_and_symlink_rejected(checkpoints):
    _, service, _, _, artifacts, recovery = checkpoints
    item = inventory(service)['checkpoints'][0];p=Path(artifacts[0]['checkpoint'])/'training_state.pt'
    p.write_bytes(b'CORRUPT')
    with pytest.raises(ValueError, match='integrity'):
        apply(service,recovery,[choose(item,'weights')])
    p.unlink();p.symlink_to('/etc/hosts')
    with pytest.raises(ValueError, match='Unsafe'):
        apply(service,recovery,[choose(item,'weights')])


def test_crash_after_unlink_replays_recorded_plan(checkpoints, monkeypatch):
    _, service, _, _, artifacts, recovery = checkpoints
    item=inventory(service)['checkpoints'][0];root=Path(artifacts[0]['checkpoint'])
    from nekaise_loop import checkpoint_retention as cr
    original=cr.atomic_write
    def crash(path,data):
        if path == root/'.retention' and json.loads(data)['status']=='complete':
            raise OSError('Simulated crash after unlink')
        original(path,data)
    monkeypatch.setattr(cr,'atomic_write',crash)
    with pytest.raises(OSError):apply(service,recovery,[choose(item,'weights')])
    verify_checkpoint(artifacts[0],require_optimizer=False)
    monkeypatch.setattr(cr,'atomic_write',original)
    assert apply(service,recovery,[choose(item,'weights')])[0]['status']=='complete'


def test_low_space_requests_review_before_training(checkpoints, monkeypatch):
    settings, service, campaign, engine, _, _ = checkpoints
    child=service.continue_campaign(campaign['id'],start=False)
    monkeypatch.setattr('nekaise_loop.checkpoint_retention.storage_status',lambda *args: {'pressure':True,'free_bytes':1,'required_bytes':100})
    engine.run(child['id'])
    assert service.store.campaign(child['id'])['status']=='recovering'
    assert service.store.one('SELECT kind FROM recoveries ORDER BY id DESC')['kind']=='storage_pressure'
    assert not service.store.query('SELECT s.id FROM stage_runs s JOIN rounds r ON r.id=s.round_id WHERE r.campaign_id=?',(child['id'],))


def test_pending_reader_protects_weights_and_lock_excludes_cleanup(checkpoints, tmp_path):
    settings, service, _, _, artifacts, recovery = checkpoints
    import fcntl
    settings.root = tmp_path/'studio'
    leases = tmp_path/'nekaise-bench/workspace/monitor/protected-checkpoints.json'
    leases.parent.mkdir(parents=True);leases.write_text(json.dumps({'paths':[artifacts[0]['checkpoint']]}))
    item=inventory(service)['checkpoints'][0]
    with pytest.raises(ValueError,match='needed for inference'):
        apply(service,recovery,[choose(item,'summary')])
    with (settings.workspace/'checkpoint-retention.lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_SH)
        with pytest.raises(BlockingIOError):
            apply(service,recovery,[choose(item,'weights')])
    apply(service,recovery,[choose(item,'weights')])


def test_missing_optimizer_without_receipt_is_corruption(checkpoints):
    _, _, _, _, artifacts, _ = checkpoints
    (Path(artifacts[0]['checkpoint'])/'training_state.pt').unlink()
    with pytest.raises(FileNotFoundError):
        verify_checkpoint(artifacts[0],require_optimizer=False)
