"""Operator source replacement: new corpus frontier, preserved GPC and saved Adam."""
import copy
import json
from pathlib import Path

from .artifacts import atomic_write, canonical, digest, verify_checkpoint
from .curriculum_progress import initial_state, require_active_namespace
from .domain_corpus import DomainSource
from .storage import encode, now
from .training_runtime import optimizer_transition


def prepare(service, parent, config, reason):
    old = parent['config'].get('curriculum_loop'); new = config.curriculum_loop
    if not old or not new or new.namespace == old['namespace']:
        raise ValueError('Corpus replacement requires a distinct forward namespace')
    if new.projection_artifact != old['projection_artifact']:
        raise ValueError('Corpus replacement must preserve the GPC projection')
    old_root = (service.settings.root / parent['config']['corpus_path']).resolve()
    if Path(config.corpus_path).resolve() == old_root:
        raise ValueError('Corpus replacement needs a different source binding')
    if config.source_prefix:
        raise ValueError('Domain replacement requires an empty source_prefix')
    # This operator operation intentionally supports explicit domain bindings only.
    backend = DomainSource(Path(config.corpus_path))
    if not config.inherit_optimizer:
        raise ValueError('Corpus replacement must preserve the optimizer')
    if service.store.one('SELECT namespace FROM curriculum_progress WHERE namespace=?', (new.namespace,)):
        raise ValueError('Replacement namespace is already in use')
    previous = service.store.one('SELECT * FROM curriculum_progress WHERE namespace=?', (old['namespace'],))
    if not previous: raise ValueError('Committed parent curriculum progress is missing')
    contract = {'projection_artifact': old['projection_artifact'], 'corpus_path': str(old_root)}
    if json.loads(previous['contract']) != contract: raise ValueError('Parent source contract changed')
    before = json.loads(previous['state'])
    if before.get('checkpoint') and Path(before['checkpoint']).resolve() != Path(config.student_model).resolve():
        raise ValueError('Source replacement checkpoint differs from committed frontier')
    checkpoint = Path(config.student_model) / 'checkpoint.json'
    bridge = None
    if checkpoint.is_file():
        manifest = json.loads(checkpoint.read_text())
        if before.get('checkpoint') and 'training_state.pt' not in manifest.get('files', {}):
            raise ValueError('Committed checkpoint has no saved optimizer state')
        verify_checkpoint({'checkpoint': config.student_model, 'manifest': manifest})
        bridge = optimizer_transition(manifest, config.model_dump())
    elif before.get('checkpoint'):
        raise ValueError('Committed checkpoint manifest is missing')
    # Fail visibly when the state contract grows until migration semantics are explicit.
    after = copy.deepcopy(before); empty = initial_state()
    reset = ('corpus_cycle', 'inventory', 'position', 'document_artifact', 'offset', 'documents_completed', 'chars_trained')
    preserved = {'gpc_completed', 'completed_rounds', 'checkpoint', 'web_chars_trained', 'web_coverage_artifact'}
    unknown = set(before) - set(reset) - preserved
    if unknown or set(empty) - set(before):
        raise ValueError('Unknown or incomplete curriculum state during corpus replacement: ' + ', '.join(sorted(unknown)))
    for key in reset: after[key] = empty[key]
    after['checkpoint'] = config.student_model
    pending = service.store.query("SELECT id,status FROM rounds WHERE campaign_id=? AND status!='complete' ORDER BY number", (parent['id'],))
    catalog = service.settings.workspace / 'curriculum/source-catalog'
    copied = []
    for source in sorted((catalog / old['namespace']).glob('*.json')):
        original = json.loads(source.read_text()); entry = {**original, 'namespace': new.namespace,
            'source_migration': {'from_namespace': old['namespace'], 'original_entry_artifact': service.artifacts.put(original)}}
        target = catalog / new.namespace / source.name
        if target.exists() and target.read_bytes() != canonical(entry):
            raise ValueError('Conflicting migrated web source catalog')
        atomic_write(target, canonical(entry)); copied.append({'name': source.name, 'sha256': digest(entry), 'original_sha256': digest(original)})
    projection = service.artifacts.get(new.projection_artifact)
    unit = projection['units'][after['gpc_completed'] % len(projection['units'])]['id']
    record = {'kind': 'operator_corpus_replacement_v1', 'reason': reason, 'parent_campaign_id': parent['id'],
              'from_namespace': old['namespace'], 'to_namespace': new.namespace, 'previous': previous,
              'before': before, 'after': after, 'sequence': previous['sequence'],
              'contract': {'projection_artifact': new.projection_artifact, 'corpus_path': str(Path(config.corpus_path).resolve())},
              'domain_source_hash': backend.identity, 'domain_binding': backend.descriptor,
              'gpc_next_unit': unit, 'catalog_root': str(catalog), 'copied_web_catalog': copied, 'optimizer_transition': bridge,
              'superseded_preparation_for_this_continuation': pending,
              'prior_budget_since': parent.get('teacher_budget_since') or parent['created_at'],
              'coverage_meaning': 'Corpus coverage restarts within the new source contract, not novel-to-model knowledge. GPC/web exposure and all past spending remain recorded.'}
    return record


def commit(db, record):
    """Called in the same transaction that creates the continuation and start action."""
    require_active_namespace(db, record['from_namespace'])
    row = db.execute('SELECT * FROM curriculum_progress WHERE namespace=?', (record['from_namespace'],)).fetchone()
    if row is None or dict(row) != record['previous']:
        raise ValueError('Committed frontier changed during source replacement')
    if db.execute('SELECT namespace FROM curriculum_progress WHERE namespace=?', (record['to_namespace'],)).fetchone():
        raise ValueError('Replacement namespace is already in use')
    catalog = Path(record['catalog_root'])
    for entry in record['copied_web_catalog']:
        for namespace, key in [(record['from_namespace'], 'original_sha256'), (record['to_namespace'], 'sha256')]:
            if digest(json.loads((catalog / namespace / entry['name']).read_text())) != entry[key]:
                raise ValueError('Web source catalog changed during corpus replacement')
    db.execute('INSERT INTO curriculum_progress VALUES (?,?,?,?,?)',
               (record['to_namespace'], encode(record['contract']), record['sequence'], encode(record['after']), now()))
    db.execute('INSERT INTO curriculum_source_replacements VALUES (?,?,?,?)',
               (record['from_namespace'], record['to_namespace'], record['parent_campaign_id'], now()))
