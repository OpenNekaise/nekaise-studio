"""Measured update-loop duty and saved exposure, including preparation/recovery gaps."""
from datetime import datetime, timezone
import json


def snapshot(store, artifacts, campaign):
    if (campaign['config'].get('teaching_cycle') or {}).get('policy') != 'continuous_v1':
        return None
    lineage, row = [], campaign
    while row and (row['config'].get('teaching_cycle') or {}).get('policy') == 'continuous_v1':
        lineage.append(row)
        row = store.campaign(row['parent_campaign_id']) if row.get('parent_campaign_id') else None
    ids = [c['id'] for c in lineage]
    marks = ','.join('?' for _ in ids)
    beginning = min(c['created_at'] for c in lineage)
    current = datetime.now(timezone.utc)
    seconds = max(0, (current-datetime.fromisoformat(beginning)).total_seconds())
    # One actual last metric per attempt. Interrupted work is visible, but never
    # conflated with durable trained targets or unique source exposure.
    metrics = store.query(f'''SELECT m.round_id,m.attempt,m.data FROM metrics m
        JOIN rounds r ON r.id=m.round_id WHERE r.campaign_id IN ({marks})
        AND m.step=(SELECT MAX(x.step) FROM metrics x WHERE x.round_id=m.round_id AND x.attempt=m.attempt)''', ids)
    observed, saved_loop, tokens = 0., 0., 0
    shares = {'general_chat':0, 'general_prose':0, 'domain':0, 'unspecified':0}
    saved = store.query(f"SELECT s.round_id,s.attempt,s.artifact FROM stage_runs s JOIN rounds r ON r.id=s.round_id WHERE r.campaign_id IN ({marks}) AND s.stage='train' AND s.status='complete'", ids)
    saved_keys = {(r['round_id'],r['attempt']) for r in saved}
    for metric in metrics:
        data = json.loads(metric['data'])
        duration = data.get('elapsed_seconds', 0)
        observed += duration
        if (metric['round_id'],metric['attempt']) in saved_keys:
            saved_loop += duration
    for stage in saved:
        data = artifacts.get(stage['artifact'])
        receipt = data.get('material_portfolio', {})
        if receipt.get('status') == 'verified':
            tokens += receipt['tokens']
            for scope, count in receipt['by_scope'].items():
                shares[scope] = shares.get(scope, 0)+count
    prepared = store.query("SELECT r.id,r.number,s.artifact FROM rounds r JOIN stage_runs s ON s.round_id=r.id "
        "WHERE r.campaign_id=? AND s.stage='freeze' AND s.status='complete' "
        "AND NOT EXISTS(SELECT 1 FROM stage_runs t WHERE t.round_id=r.id AND t.stage='train' AND t.status='complete')", (campaign['id'],))
    from .preparation_summary import read
    queue_tokens = sum(read(store, artifacts, r['artifact'])['ledger']['total_tokens'] for r in prepared)
    return {'since':beginning, 'wall_seconds':seconds, 'observed_training_seconds':observed,
        'saved_training_seconds':saved_loop, 'unsaved_training_seconds':max(0,observed-saved_loop),
        'training_time_share':observed/seconds if seconds else None,
        'durable_targets':tokens, 'durable_targets_per_second':tokens/seconds if seconds else None,
        'durable_scope_targets':shares, 'prepared_windows':len(prepared), 'prepared_targets':queue_tokens,
        'basis':'Measured optimizer-loop time / all elapsed wall time since continuous deployment. Includes startup, teaching, saving, recovery and operator holds in the denominator. Unsaved work may be in flight or rolled back; it is not verified exposure. This is not CUDA utilization or learning quality.'}
