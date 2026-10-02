"""Offline preparation protects original model code, source integrity and scope."""
import importlib.util
import json
from pathlib import Path
import sys
import io
import tarfile
from types import SimpleNamespace

import pytest


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prep = load_script('prepare_energy_modeling')
final = load_script('finalize_energy_modeling')
web = load_script('acquire_energy_modeling_web')


def test_body_mentions_and_patents_do_not_become_core():
    assert prep.classify({'source': 'google_patents', 'title': 'A battery connector'}, True)[0] == 'mention_candidate'
    assert prep.classify({'source': 'google_patents', 'title': 'Thermal simulation of a building'})[0] == 'patent_supplement'
    assert prep.classify({'source': 'ibpsa', 'title': 'A building simulation study'})[0] == 'research'
    assert prep.classify({'source': 'gh_ideas', 'title': 'GPQA task bank'})[0] == 'excluded'


def test_untrusted_archive_paths_and_benchmark_banks_are_not_selected():
    assert prep.include_repo_file('Buildings/Fluid/Examples/Pump.mo')
    assert prep.include_repo_file('idd/Energy+.idd.in')
    for path in ('../../escaped.mo', '/absolute.mo', 'vendor/foo.mo', 'tasks/GPQA.md', 'ReferenceResults/run.mo'):
        assert not prep.include_repo_file(path)
    assert not prep.include_repo_file('code/foo.py', ['docs/'])


def test_audit_restores_exact_original_modelica_and_never_mutates_source(tmp_path):
    root, out = tmp_path / 'corpus', tmp_path / 'prepared'
    for name in ('manifest', 'corpus', 'raw', 'registry'):
        (root / name).mkdir(parents=True)
    prep.dump(root / 'registry/eligibility.json', {'version': 2, 'restrictions': {}})
    original = b'model Wall\n  Real T;\nequation\n  der(T) = -T;\nannotation (Documentation(info="wall"));\nend Wall;\n'
    damaged = b'# Modelica Wall\nannotation (\nend Wall;'
    (root / 'raw/wall.mo').write_bytes(original)
    (root / 'corpus/wall.md').write_bytes(damaged)
    row = {'id': 'wall', 'title': 'Modelica Wall', 'source': 'gh_modelica-buildings', 'status': 'ok',
        'license': 'open', 'url': 'https://raw.githubusercontent.com/lbl-srg/modelica-buildings/master/Wall.mo',
        'raw_path': 'raw/wall.mo', 'corpus_path': 'corpus/wall.md', 'sha256': prep.sha(original), 'corpus_sha256': prep.sha(damaged)}
    (root / 'manifest/models.jsonl').write_text(json.dumps(row) + '\n')
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    prep.audit(root, out)
    record = next(final.records(out / 'local-documents.jsonl'))
    assert (out / record['object_path']).read_bytes() == original
    assert record['representation'] == 'verbatim_original_code_or_document'
    assert before == {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    (root / 'corpus/wall.md').write_text('unexpected source mutation')
    prep.audit(root, tmp_path / 'quarantine')
    report = json.loads((tmp_path / 'quarantine/local-summary.json').read_text())
    assert 'hash differs' in report['errors'][0]['error']
    assert not (tmp_path / 'quarantine/local-documents.jsonl').read_text()


def make_bundle(tmp_path):
    prep.dump(tmp_path / 'local-summary.json', {'counts': {}, 'fulltext_match_documents': 0, 'verified_by_tier': {}, 'errors': []})
    spec = tmp_path / 'sources.json'
    prep.dump(spec, {'repositories': [], 'web_sources': [], 'unresolved': []})
    text = 'model Wall\n  Real T;\nend Wall;\n'
    digest, path = prep.pin_text(tmp_path, text.encode())
    row = {'id': 'wall', 'url': 'https://example.org/Wall.mo', 'title': 'Wall', 'source': 'fixture-modelica',
        'origin': 'existing_corpus', 'tier': 'core', 'object_path': path, 'text_sha256': digest,
        'duplicate_key': prep.sha(text.strip().encode()), 'chars': len(text), 'categories': ['modelica']}
    with (tmp_path / 'local-documents.jsonl').open('w') as handle:
        prep.emit(handle, row)
        prep.emit(handle, {**row, 'id': 'duplicate'})
    return spec, row, text


def test_finalize_exact_dedup_and_verbatim_export(tmp_path):
    spec, _, text = make_bundle(tmp_path)
    result = final.finalize(tmp_path, spec)
    assert result['counts']['core'] == result['counts']['exact_duplicates'] == 1
    assert next(final.records(tmp_path / 'dataset/core.text.jsonl'))['text'] == text
    assert not result['training_started'] and not result['curriculum_activated']


def test_corrupt_object_cannot_be_claimed_ready(tmp_path):
    spec, row, _ = make_bundle(tmp_path)
    (tmp_path / row['object_path']).write_text('tampered')
    with pytest.raises(RuntimeError, match='integrity'):
        final.finalize(tmp_path, spec)
    assert json.loads((tmp_path / 'readiness.json').read_text())['readiness'] == 'integrity_failure'


def test_versions_and_bundled_sources_remain_separate():
    row = {'source': 'NREL/EnergyPlus', 'origin': 'new_repository_snapshot', 'path': 'idd/versions/V9/Energy+.idd', 'tier': 'core'}
    assert final.partition(row, 'IDD', set())[0] == 'reference_assets'
    row.update(source='NatLabRockies/Spawn', path='energyplus/src/EnergyPlus/Zone.cc')
    assert final.partition(row, 'code', set())[0] == 'reference_assets'
    row.update(source='santoshphilip/eppy', path='eppy/iddv940.py')
    assert final.partition(row, 'schema', set())[0] == 'reference_assets'
    row['path'] = 'eppy/resources/iddfiles/Energy+V9_2_0.idd'
    assert final.partition(row, 'schema', set())[0] == 'reference_assets'
    row = {'origin': 'existing_corpus', 'url': 'https://raw.githubusercontent.com/modelica/ModelicaStandardLibrary/master/Modelica/Fluid.mo', 'tier': 'core'}
    assert final.partition(row, 'code', {('modelica/modelicastandardlibrary', 'Modelica/Fluid.mo')})[0] == 'reference_history'


def test_source_and_object_paths_cannot_escape(tmp_path):
    with pytest.raises(ValueError, match='escapes'):
        prep.safe_path(tmp_path, '../outside')


def test_empty_repository_scope_is_not_complete_and_corrected_scope_retries(tmp_path, monkeypatch):
    commit = 'a' * 40
    directory = tmp_path / 'repositories/example__models'
    directory.mkdir(parents=True)
    body = b'model Wall\nend Wall;\n'
    with tarfile.open(directory / (commit + '.tar.gz'), 'w:gz') as archive:
        member = tarfile.TarInfo('root/Models/Wall.mo')
        member.size = len(body)
        archive.addfile(member, io.BytesIO(body))
    monkeypatch.setattr(prep.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout=commit + '\tHEAD\n', stderr=''))
    spec = {'repo': 'example/models', 'prefixes': ['wrong/'], 'categories': ['modelica']}
    assert prep.acquire_repo(spec, tmp_path)['status'] == 'empty'
    spec['prefixes'] = ['Models/']
    result = prep.acquire_repo(spec, tmp_path)
    assert result['status'] == 'complete'
    assert result['counts']['documents'] == 1
    assert list((directory / 'attempts').glob('*.json'))


@pytest.mark.parametrize('title, expected', [
    ('IDAICE building simulation', 'ida_ice'),
    ('IDA-ICE workflow', 'ida_ice'),
    ('IDA Klimat och Energi', 'ida_ice'),
    ('住宅建筑能效标识方法', 'building_energy_rating'),
    ('Rakennuksen energialuokka', 'building_energy_rating'),
    ('Building energy class A', 'energy_class_contextual'),
    ('EPC engineering procurement construction', None),
    ('ICE railway timetable', None),
    ('Energy class A refrigerator', None),
    ('NMF matrix factorization and IDA loans', None),
])
def test_multilingual_aliases_keep_ambiguous_abbreviations_out(title, expected):
    config = json.loads((Path(__file__).parents[1] / 'curricula/energy_modeling_keywords.json').read_text())
    result = prep.keyword_matches(title, config)
    assert expected in result if expected else not result


def test_addition_deduplicates_verified_base_without_modifying_it(tmp_path):
    base, addition = tmp_path / 'base', tmp_path / 'addition'
    base.mkdir(); addition.mkdir()
    spec, _, _ = make_bundle(base)
    final.finalize(base, spec)
    before = {str(p): p.read_bytes() for p in base.rglob('*') if p.is_file()}
    spec, row, _ = make_bundle(addition)
    text = b'model NewWall\nend NewWall;\n'
    digest, path = prep.pin_text(addition, text)
    with (addition / 'local-documents.jsonl').open('a') as handle:
        prep.emit(handle, {**row, 'id': 'new-wall', 'object_path': path, 'text_sha256': digest,
                           'duplicate_key': prep.sha(text.strip()), 'chars': len(text)})
    report = final.finalize(addition, spec, base)
    assert report['counts']['core'] == 1
    assert report['counts']['base_duplicates'] == 2
    assert before == {str(p): p.read_bytes() for p in base.rglob('*') if p.is_file()}
    (base / 'dataset/core.manifest.jsonl').write_text('tampered')
    with pytest.raises(ValueError, match='Base manifest hash mismatch'):
        final.finalize(addition, spec, base)


def test_addition_cannot_overwrite_its_base(tmp_path):
    spec, _, _ = make_bundle(tmp_path)
    before = (tmp_path / 'local-summary.json').read_bytes()
    with pytest.raises(ValueError, match='must differ'):
        final.finalize(tmp_path, spec, tmp_path)
    assert (tmp_path / 'local-summary.json').read_bytes() == before


@pytest.mark.parametrize('role', ['reference_assets', 'reference_history', 'reference_pdf'])
def test_archive_preparation_role_keeps_references_out_of_core(role):
    split, reason = final.partition({'preparation_role': role}, 'model Example end Example;', set())
    assert split == role and 'archive preparation role' in reason


def test_declared_main_region_recovers_form_wrapped_handbook_not_cookie_banner():
    html = '''<form><nav>Site navigation</nav><div role="main"><nav>Breadcrumb</nav>
    <div><h1>Building energy calculation</h1><p>Heating load equals transmission
    plus ventilation losses. Outdoor &lt; indoor temperature.</p>
    <table><tr><td>U-value</td><td>0.18 W/(m² K)</td></tr></table><br/>
    <form>Search control</form></div></div></form><p>Cookie banner</p>'''
    text = web.role_main_text(html)
    assert 'Heating load' in text and 'Outdoor < indoor' in text
    assert '0.18 W/(m² K)' in text
    assert all(value not in text for value in ('Breadcrumb', 'Cookie banner', 'Search control', 'Site navigation'))
    with pytest.raises(ValueError, match='missing or unclosed'):
        web.role_main_text('<p>Cookie banner</p>')


def test_web_region_omission_keeps_readable_pages_and_source_metadata(tmp_path, monkeypatch):
    import base64

    def collect(ctx, request):
        pages = []
        for index, html in enumerate([
            '<form><div role="main"><p>Building heat loss and ventilation calculation with '
            'reference outdoor and indoor temperatures, heat transfer coefficients and areas.</p></div></form>',
            '<div role="main">Index</div>',
        ]):
            raw = html.encode()
            raw_key = ctx.artifacts.put({'body_base64': base64.b64encode(raw).decode(), 'encoding': 'utf-8'})
            page_key = ctx.artifacts.put({'id': str(index), 'url': request['url'] + str(index),
                'text': 'Cookie banner', 'extractor': 'readable_body_v2', 'text_sha256': prep.sha(b'Cookie banner'),
                'raw_response_artifact': raw_key, 'source_sha256': prep.sha(raw),
                'robots_artifact': 'test-robots', 'retrieved_at': '2026-10-02'})
            pages.append({'artifact': page_key})
        return ctx.artifacts.put({'complete': True, 'pages': pages, 'pending': [],
            'failures': [], 'bounded_by': 'frontier_exhausted'})

    monkeypatch.setattr(web, 'collect', collect)
    receipt = web.acquire({'id': 'handbook', 'url': 'https://example.org/', 'prefix': 'https://example.org/',
        'categories': ['energy_rating'], 'html_region': 'role_main',
        'language_hint': 'da', 'jurisdiction': 'DK', 'version_note': '2023'}, tmp_path)
    assert receipt['status'] == 'partial' and receipt['pages'] == 1
    assert len(receipt['extraction_omissions']) == 1
    row, = final.records(tmp_path / 'web-sources/handbook/documents.jsonl')
    assert row['language_hint'] == 'da' and row['language_hint_is_detection'] is False
    assert row['jurisdiction'] == 'DK' and row['version_note'] == '2023'
    assert row['prior_extraction_sha256'] == prep.sha(b'Cookie banner')
    assert 'Building heat loss' in (tmp_path / row['object_path']).read_text()
