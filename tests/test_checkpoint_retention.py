import json
from pathlib import Path
from unittest.mock import patch

import pytest

from nekaise_loop.artifacts import verify_checkpoint
from nekaise_loop.checkpoint_retention import apply, inventory, sha, storage_status
from nekaise_loop.storage import now


def successor(checkpoints, **updates):
    settings, service, campaign, engine, artifacts, recovery = checkpoints
    child = service.continue_campaign(campaign['id'], {'rounds': 2, **updates}, start=False)
    engine.run(child['id'])
    return child


def test_superseded_campaign_tip_and_origin_can_be_retired_without_losing_history(checkpoints):
    _, service, parent, _, artifacts, recovery = checkpoints
    child = successor(checkpoints)
    items = inventory(service)['checkpoints']
    old = next(x for x in items if x['path'] == Path(artifacts[-1]['checkpoint']).relative_to(service.settings.workspace).as_posix())
    current = next(x for x in reversed(items) if x['campaign_id'] == child['id'])
    assert old['historical_references'] and not old['protected_resumable']
    assert current['protected_resumable']
    manifests = [(Path(a['checkpoint'])/'checkpoint.json').read_bytes() for a in artifacts]
    parent_config = service.store.campaign(parent['id'])['config']
    rounds_before = service.store.query('SELECT * FROM rounds WHERE campaign_id=?', (parent['id'],))
    apply(service, recovery + 1, [choose(old, 'summary')])
    assert service.store.query('SELECT * FROM rounds WHERE campaign_id=?', (parent['id'],)) == rounds_before
    assert service.store.campaign(parent['id'])['config'] == parent_config
    assert [(Path(a['checkpoint'])/'checkpoint.json').read_bytes() for a in artifacts] == manifests
    assert service.artifacts.get(old['artifact']) == artifacts[-1]
    count = service.store.one('SELECT COUNT(*) AS n FROM campaigns')['n']
    with pytest.raises(ValueError, match='intentionally retired'):
        service.continue_campaign(parent['id'], start=False)
    assert service.store.one('SELECT COUNT(*) AS n FROM campaigns')['n'] == count


def test_current_paused_lineage_survives_a_new_unstarted_draft(checkpoints):
    from nekaise_loop.config import CampaignConfig
    from nekaise_loop.checkpoint_lineage import current_campaign_id
    _, service, _, _, _, _ = checkpoints
    child = successor(checkpoints)
    service.store.set_status(child['id'], 'paused')
    service.create('Unstarted draft', CampaignConfig.model_validate(child['config']))
    assert current_campaign_id(service.store) == child['id']
    items = inventory(service)['checkpoints']
    assert next(x for x in reversed(items) if x['campaign_id'] == child['id'])['protected_resumable']


def test_queued_draft_protection_is_recomputed_at_apply(checkpoints):
    from nekaise_loop.config import CampaignConfig
    _, service, _, _, artifacts, recovery = checkpoints
    child = successor(checkpoints)
    old = next(x for x in inventory(service)['checkpoints'] if x['round_id'] == Path(artifacts[0]['checkpoint']).parts[-3])
    assert not old['protected_resumable']
    draft = service.create('Queued historical input', CampaignConfig.model_validate({**child['config'], 'student_model': artifacts[0]['checkpoint']}))
    service.action(draft['id'], 'start', spawn=False)
    with pytest.raises(ValueError, match='needed for resumption'):
        apply(service, recovery + 1, [choose(old, 'summary')])
    assert (Path(artifacts[0]['checkpoint'])/'training_state.pt').exists()


def test_chat_snapshot_and_frozen_diagnostic_remain_protected_during_unfinished_round(checkpoints):
    from nekaise_loop.model_chat import latest_snapshot
    _, service, _, _, artifacts, recovery = checkpoints
    child = successor(checkpoints, student_format='chat_template')
    current_round = service.store.one('SELECT id FROM rounds WHERE campaign_id=? ORDER BY number DESC LIMIT 1', (child['id'],))['id']
    service.store.set_status(child['id'], 'paused')
    service.store.execute("UPDATE rounds SET status='paused' WHERE id=?", (current_round,))
    selected = service.store.one("SELECT id,artifact FROM stage_runs WHERE round_id=? AND stage='select'", (current_round,))
    original = service.artifacts.get(selected['artifact'])
    frozen = service.artifacts.put({**original, 'diagnostic_checkpoint': artifacts[0]['checkpoint']})
    service.store.execute('UPDATE stage_runs SET artifact=? WHERE id=?', (frozen, selected['id']))
    metadata, snapshot = latest_snapshot(service)
    assert metadata['campaign_id'] == child['id']
    items = inventory(service)['checkpoints']
    chat = next(x for x in items if str(service.settings.workspace/x['path']) == snapshot['checkpoint'])
    old = next(x for x in items if str(service.settings.workspace/x['path']) == artifacts[0]['checkpoint'])
    assert 'Current Model chat snapshot' in chat['protected_weights']
    assert old['protected_weights'] and not old['protected_resumable']
    with pytest.raises(ValueError, match='needed for inference'):
        apply(service, recovery + 1, [choose(old, 'summary')])
    service.store.execute("UPDATE rounds SET status='complete' WHERE id=?", (current_round,))
    assert not next(x for x in inventory(service)['checkpoints'] if x['path'] == old['path'])['protected_weights']


def test_unsuperseded_explicit_pause_keeps_its_resume_state(checkpoints):
    from nekaise_loop.config import CampaignConfig
    _, service, campaign, engine, artifacts, _ = checkpoints
    # A separate new run does not supersede an explicitly paused branch.
    service.store.execute("UPDATE campaigns SET status='paused',operator_hold='pause' WHERE id=?", (campaign['id'],))
    service.store.execute("UPDATE recoveries SET status='resolved' WHERE campaign_id=?", (campaign['id'],))
    other = service.create('Separate current run', CampaignConfig.model_validate(campaign['config']))
    engine.run(other['id'])
    old = next(x for x in inventory(service)['checkpoints'] if str(service.settings.workspace/x['path']) == artifacts[-1]['checkpoint'])
    assert old['protected_resumable']


def test_v2_reader_lease_protects_only_inference_bytes(checkpoints, tmp_path):
    settings, service, _, _, artifacts, recovery = checkpoints
    successor(checkpoints)
    settings.root = tmp_path/'studio'
    leases = tmp_path/'nekaise-bench/workspace/monitor-v2/protected-checkpoints.json'
    leases.parent.mkdir(parents=True)
    leases.write_text(json.dumps({'paths': [artifacts[0]['checkpoint']]}))
    old = next(x for x in inventory(service)['checkpoints'] if str(settings.workspace/x['path']) == artifacts[0]['checkpoint'])
    assert old['protected_weights'] and not old['protected_resumable']
    with pytest.raises(ValueError, match='needed for inference'):
        apply(service, recovery + 1, [choose(old, 'summary')])
    apply(service, recovery + 1, [choose(old, 'weights')])


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


def test_keep_only_recovery_applies_during_independent_checkpoint_read(checkpoints):
    import fcntl
    from nekaise_loop.history import apply_history
    from nekaise_loop.ownership import source_lock
    settings, service, campaign, _, artifacts, recovery = checkpoints
    items = inventory(service)['checkpoints']
    decision = {'run_retention': [], 'log_removals': [],
                'checkpoint_retention': [choose(items[-1], 'keep')], 'history_review_after_rounds': 1}
    originals = {str(Path(a['checkpoint'])/name): (Path(a['checkpoint'])/name).read_bytes()
                 for a in artifacts for name in [*a['manifest']['files'], 'checkpoint.json']}
    with (settings.workspace/'checkpoint-retention-writer.lock').open('a+') as other_writer:
        fcntl.flock(other_writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            apply(service, recovery, decision['checkpoint_retention'])
    with (settings.workspace/'checkpoint-retention.lock').open('a+') as reader:
        fcntl.flock(reader, fcntl.LOCK_SH | fcntl.LOCK_NB)
        with source_lock(exclusive=True):
            result = apply_history(service, recovery, decision)
            assert result == apply_history(service, recovery, decision)
            assert service.store.one('SELECT result FROM history_reviews WHERE recovery_id=?', (recovery,))
            for choices in ([choose(items[0], 'weights')],
                            [choose(items[-1], 'keep'), choose(items[0], 'summary')]):
                with pytest.raises(BlockingIOError):
                    apply(service, recovery + 1, choices)
    assert all(Path(path).read_bytes() == content for path, content in originals.items())
    assert not (Path(artifacts[0]['checkpoint'])/'.retention').exists()
    assert json.loads((Path(artifacts[-1]['checkpoint'])/'.retention').read_text())['status'] == 'complete'


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
