"""Pinned domain inputs and operator replacement, without real models or teaching calls."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nekaise_loop.artifacts import canonical, digest
from nekaise_loop.config import CampaignConfig
from nekaise_loop.corpus import pinned_source_eligible, read_source, search_sources
from nekaise_loop.curriculum_inventory import build_inventory, next_document, read_inventory_source
from nekaise_loop.curriculum_progress import assignment, initial_state
from nekaise_loop.domain_corpus import DomainSource, bind_source, source_id, inside
from nekaise_loop.general_curriculum import import_curriculum
from nekaise_loop.storage import encode, now


def domain_fixture(tmp_path):
    root = tmp_path/'publisher'; root.mkdir()
    texts = [('git:org/models@abc:Wall.mo', 'model Wall\n Real T; equation der(T)=-T;\nend Wall;\n'*12, 'core'),
             ('upload:manual:Thermal.pdf', 'A PDF thermal reference with incomplete math glyphs. '*20, 'reference_pdf')]
    members=[]
    for sid,text,role in texts:
        payload=text.encode();h=hashlib.sha256(payload).hexdigest();p=root/'objects'/h;p.parent.mkdir(exist_ok=True);p.write_bytes(payload)
        members.append({'id':sid,'title':sid,'source':'physical-library','split':role,'categories':['modelica'],
                        'license':'not verified; operator source selection','quality_note':'Fixture reference' if role!='core' else '',
                        'duplicate_key':hashlib.sha256(text.strip().encode()).hexdigest(),
                        'text_artifact':{'stage':'text','path':str(p.relative_to(root)),'bytes':len(payload),'sha256':h}})
    identity={'schema':1,'definition':{'id':'physical-modeling'},'inventories':[],'deduplication':'exact prepared body hash','builder':'fixture'}
    sid=digest(identity);snapshot=root/'snapshots'/sid;snapshot.mkdir(parents=True)
    (snapshot/'members.jsonl').write_bytes(b'\n'.join(canonical(r) for r in members)+b'\n')
    alias={**members[0],'id':'git:old/mirror@abc:Wall.mo'}
    (snapshot/'aliases.jsonl').write_bytes(canonical(alias)+b'\n')
    report={**identity,'id':sid,'counts':{'core':1,'reference_pdf':1},
            'members_sha256':hashlib.sha256((snapshot/'members.jsonl').read_bytes()).hexdigest(),
            'aliases_sha256':hashlib.sha256((snapshot/'aliases.jsonl').read_bytes()).hexdigest()}
    (snapshot/'snapshot.json').write_bytes(canonical(report))
    return root,snapshot,members,alias


def test_binding_routes_reference_search_and_forward_inventory_separately(tmp_path):
    root,snapshot,members,alias=domain_fixture(tmp_path);workspace=tmp_path/'workspace'
    bound=bind_source(workspace,root,snapshot);path=Path(bound['corpus_path'])
    assert bind_source(workspace,root,snapshot)['corpus_path']==str(path)
    results=search_sources(path,limit=20)['rows'];assert len(results)==2
    assert sum(r['forward_eligible'] for r in results)==1
    document=read_source(path,source_id(alias['id']));assert document['id']==source_id(members[0]['id'])
    assert document['source_sha256']==members[0]['text_artifact']['sha256']==alias['text_artifact']['sha256']
    assert document['original_document_id']==members[0]['id']
    assert document['license'].startswith('not verified') and pinned_source_eligible(path,document)
    pdf=read_source(path,source_id(members[1]['id']));assert not pinned_source_eligible(path,pdf)
    inventory=build_inventory(path,workspace);assert inventory['documents']==1
    entry=next_document(workspace,inventory,'');assert entry['document_id']==document['id']
    assert read_inventory_source(path,inventory,entry)['text']==document['text']
    assert next_document(workspace,inventory,entry['position']) is None
    other=bind_source(workspace,root,snapshot,roles=['core','reference_pdf'])
    assert other['corpus_path']!=str(path)
    with pytest.raises(ValueError,match='different binding'):
        read_inventory_source(Path(other['corpus_path']),inventory,entry)


def test_changed_payload_metadata_and_escaping_references_fail(tmp_path):
    root,snapshot,members,_=domain_fixture(tmp_path);workspace=tmp_path/'workspace'
    binding=bind_source(workspace,root,snapshot);path=Path(binding['corpus_path'])
    with pytest.raises(ValueError,match='contained'):inside(root,'../outside.txt')
    outside=tmp_path/'outside.txt';outside.write_text('not corpus')
    (root/'escape').symlink_to(outside)
    with pytest.raises(ValueError,match='escapes'):inside(root,'escape')
    (root/members[0]['text_artifact']['path']).write_bytes(b'tampered')
    with pytest.raises(ValueError,match='payload hash'):read_source(path,source_id(members[0]['id']))
    with pytest.raises(ValueError,match='payload hash'):bind_source(workspace,root,snapshot)
    (path/'index.sqlite3').write_bytes(b'bad')
    with pytest.raises(ValueError,match='metadata hash'):search_sources(path)


def test_binding_descriptor_cannot_change_in_place(tmp_path):
    root,snapshot,_,_=domain_fixture(tmp_path);binding=bind_source(tmp_path/'workspace',root,snapshot)
    path=Path(binding['corpus_path']);desc=json.loads((path/'domain-source.json').read_text())
    desc['forward_roles']=['reference_pdf'];(path/'domain-source.json').write_bytes(canonical(desc))
    with pytest.raises(ValueError,match='identity mismatch'):DomainSource(path)


def test_forward_assignment_contains_only_selected_role_and_retains_source_identity(setup_loop,tmp_path):
    settings,service,parent,_=setup_loop
    root,snapshot,members,_=domain_fixture(tmp_path)
    bound=bind_source(settings.workspace,root,snapshot)
    loop=import_curriculum(settings.workspace,Path(__file__).parents[1]/'curricula/general_purpose_curriculum.json',namespace='domain-test',span_chars=128,corpus_window_chars=1024)
    config=CampaignConfig.model_validate({**parent['config'],'corpus_path':bound['corpus_path'],'source_prefix':'','curriculum_loop':loop.model_dump(),'train_steps':0})
    context=SimpleNamespace(config=config,engine=SimpleNamespace(settings=settings),store=service.store,artifacts=service.artifacts,cancelled=lambda:False,round={'model_before':config.student_model})
    work=assignment(context)
    assert work['spans'] and all(s['domain_role']=='core' for s in work['spans'])
    assert all(s['original_document_id']==members[0]['id'] for s in work['spans'])
    assert ''.join(s['text'] for s in work['spans'])==read_source(Path(bound['corpus_path']),source_id(members[0]['id']))['text']


def migration_parent(setup_loop,tmp_path):
    settings,service,parent,_=setup_loop
    loop=import_curriculum(settings.workspace,Path(__file__).parents[1]/'curricula/general_purpose_curriculum.json',namespace='original')
    config=CampaignConfig.model_validate({**parent['config'],'curriculum_loop':loop.model_dump(),'train_steps':0})
    campaign=service.create('Source replacement fixture',config)
    before=initial_state();before.update(gpc_completed=17,completed_rounds=17,chars_trained=999,documents_completed=4,web_chars_trained=123,web_coverage_artifact=service.artifacts.put({'positions':{'fixture-page':123}}))
    contract={'projection_artifact':loop.projection_artifact,'corpus_path':str(Path(config.corpus_path).resolve())}
    service.store.execute('INSERT INTO curriculum_progress VALUES (?,?,?,?,?)',('original',encode(contract),17,encode(before),now()))
    service.store.execute("UPDATE campaigns SET status='paused',operator_hold='pause',teacher_budget_since='2026-09-01T00:00:00Z' WHERE id=?",(campaign['id'],))
    cat=settings.workspace/'curriculum/source-catalog/original';cat.mkdir(parents=True)
    (cat/'source.json').write_bytes(canonical({'namespace':'original','unit_id':'unit','id':'source','request':{},'collection_artifact':'a'*64}))
    root,snapshot,_,_=domain_fixture(tmp_path)
    binding=bind_source(settings.workspace,root,snapshot)
    updates={'replace_corpus_source':True,'corpus_path':binding['corpus_path'],'source_prefix':'','curriculum_loop':{**loop.model_dump(),'namespace':'physical-new'}}
    return settings,service,campaign,before,updates


def test_operator_fork_preserves_gpc_web_and_budget_and_resets_only_corpus(setup_loop,tmp_path):
    settings,service,parent,before,updates=migration_parent(setup_loop,tmp_path)
    child=service.continue_campaign(parent['id'],updates,start=True,reason='Use physical modeling corpus')
    new=service.store.one("SELECT * FROM curriculum_progress WHERE namespace='physical-new'");after=json.loads(new['state'])
    assert after['gpc_completed']==17 and after['web_chars_trained']==123 and after['web_coverage_artifact']==before['web_coverage_artifact']
    assert after['chars_trained']==0 and after['documents_completed']==0 and after['inventory'] is None
    assert new['sequence']==17 and after['completed_rounds']==17
    assert child['teacher_budget_since']=='2026-09-01T00:00:00Z' and child['config']['inherit_optimizer']
    assert child['status']=='queued' and child['operator_hold'] is None
    assert json.loads(service.store.one("SELECT state FROM curriculum_progress WHERE namespace='original'")['state'])==before
    assert service.store.one('SELECT kind FROM actions WHERE campaign_id=?',(child['id'],))['kind']=='start'
    copied=json.loads((settings.workspace/'curriculum/source-catalog/physical-new/source.json').read_text())
    assert copied['source_migration']['from_namespace']=='original'
    assert service.artifacts.get(child['context_artifact'])['corpus_source_migration']
    with pytest.raises(ValueError,match='superseded'):service.continue_campaign(parent['id'])
    with pytest.raises(ValueError,match='superseded'):service.action(parent['id'],'resume',spawn=False)
    with pytest.raises(ValueError,match='superseded'):service.action(parent['id'],'review',spawn=False)


def test_source_replacement_is_explicit_and_cannot_reset_gpc_or_adam(setup_loop,tmp_path):
    _,service,parent,_,updates=migration_parent(setup_loop,tmp_path)
    with pytest.raises(ValueError,match='operator'):service.continue_campaign(parent['id'],updates,actor='orchestrator')
    with pytest.raises(Exception,match='replace_corpus_source'):service.continue_campaign(parent['id'],{k:v for k,v in updates.items() if k!='replace_corpus_source'})
    with pytest.raises(ValueError,match='optimizer'):service.continue_campaign(parent['id'],{**updates,'inherit_optimizer':False})
    with pytest.raises(ValueError,match='GPC projection'):service.continue_campaign(parent['id'],{**updates,'curriculum_loop':{**updates['curriculum_loop'],'projection_artifact':'b'*64}})
    with pytest.raises(ValueError,match='source_prefix'):service.continue_campaign(parent['id'],{**updates,'source_prefix':'old-hvac-'})
    assert not service.store.one("SELECT * FROM curriculum_progress WHERE namespace='physical-new'")


def test_stale_frontier_rolls_back_child_namespace_and_start_action(setup_loop,tmp_path,monkeypatch):
    _,service,parent,before,updates=migration_parent(setup_loop,tmp_path)
    from nekaise_loop import corpus_migration
    original=corpus_migration.prepare
    def raced(*args):
        record=original(*args)
        service.store.execute("UPDATE curriculum_progress SET sequence=18 WHERE namespace='original'")
        return record
    monkeypatch.setattr(corpus_migration,'prepare',raced)
    with pytest.raises(ValueError,match='frontier changed'):service.continue_campaign(parent['id'],updates,start=True)
    assert not service.store.one("SELECT namespace FROM curriculum_progress WHERE namespace='physical-new'")
    assert not service.store.one('SELECT id FROM campaigns WHERE parent_campaign_id=?',(parent['id'],))


def test_committed_checkpoint_without_adam_cannot_be_replaced(setup_loop,tmp_path):
    _,service,parent,before,updates=migration_parent(setup_loop,tmp_path)
    checkpoint=Path(parent['config']['student_model'])
    before['checkpoint']=str(checkpoint)
    service.store.execute("UPDATE curriculum_progress SET state=? WHERE namespace='original'",(encode(before),))
    (checkpoint/'checkpoint.json').write_text(json.dumps({'files':{'model.safetensors':hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest()}}))
    with pytest.raises(ValueError,match='optimizer state'):service.continue_campaign(parent['id'],updates,start=True)


def test_unknown_cursor_fields_fail_before_migration(setup_loop,tmp_path):
    _,service,parent,before,updates=migration_parent(setup_loop,tmp_path)
    before['future_corpus_offset']=42
    service.store.execute("UPDATE curriculum_progress SET state=? WHERE namespace='original'",(encode(before),))
    with pytest.raises(ValueError,match='Unknown.*future_corpus_offset'):
        service.continue_campaign(parent['id'],updates,start=True)


def test_aborted_catalog_copy_is_inactive_and_retryable(setup_loop,tmp_path,monkeypatch):
    settings,service,parent,_,updates=migration_parent(setup_loop,tmp_path)
    from nekaise_loop import corpus_migration
    original=corpus_migration.prepare
    def corrupted(*args):
        record=original(*args)
        (settings.workspace/'curriculum/source-catalog/physical-new/source.json').write_text('{}')
        return record
    monkeypatch.setattr(corpus_migration,'prepare',corrupted)
    with pytest.raises(ValueError,match='catalog changed'):service.continue_campaign(parent['id'],updates,start=True)
    assert not service.store.one("SELECT * FROM curriculum_source_replacements")
    assert not service.store.one("SELECT * FROM curriculum_progress WHERE namespace='physical-new'")
    monkeypatch.setattr(corpus_migration,'prepare',original)
    updated={**updates,'curriculum_loop':{**updates['curriculum_loop'],'namespace':'physical-retry'}}
    child=service.continue_campaign(parent['id'],updated,start=True)
    assert child['status']=='queued'


def test_teacher_can_explicitly_read_reference_role_outside_forward_inventory(setup_loop,tmp_path):
    settings,service,parent,_=setup_loop
    root,snapshot,members,_=domain_fixture(tmp_path)
    bound=bind_source(settings.workspace,root,snapshot)
    from nekaise_loop.stages import _sources
    config=CampaignConfig.model_validate({**parent['config'],'corpus_path':bound['corpus_path'],'curriculum_loop':None})
    ctx=SimpleNamespace(config=config,engine=SimpleNamespace(settings=settings))
    selected=_sources(ctx,[{'document_id':source_id(members[1]['id']),'start':0,'length':50,'material_scope':'domain'}])
    assert selected[0]['domain_role']=='reference_pdf' and not selected[0]['forward_eligible']
    assert selected[0]['span_length']==50 and selected[0]['quality_note']
