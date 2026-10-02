"""Offline, operator-selected domain preparation. Never starts or alters training.

Corpus is read-only. All manifests, bytes, scans and acquisition receipts are written
under the explicitly supplied ignored output directory. Downloaded code is data and
is never imported, built or executed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile
import time

import httpx

from nekaise_loop.corpus import eligible, policy_at

TOOLS = r"modelica|openmodelica|dymola|energyplus|energy\s*plus|openstudio|trnsys|esp-r|espres|ida\s*ice|designbuilder|boptest|besmod|aixlib|coolprop|tespy|fmus?|co-simulation|cosimulation|radiance|contam|wufi|delphin|champs|thermosyspro|thermopower|thermofluidstream"
PHYSICS = r"building|thermal|thermo|heat|hvac|energy|fluid|airflow|ventilat|hydronic|refriger|psychrom|district|radiat|moisture|enthalpy|建筑|热|暖通|能耗|能源|流体|通风|湿|wärme|gebäude|bâtiment"
MODELING = r"simulat|model(?:ling|ing)?|calibrat|differential|finite.volume|finite.element|equation|numerical|cfd|dae|模型|建模|仿真|模拟|方程|数值|modell|simulation"
FULLTEXT_TERMS = (
    'modelica', 'dymola', 'energyplus', 'energy plus', 'openstudio', 'trnsys',
    'esp-r', 'ida ice', 'designbuilder', 'boptest', 'besmod', 'aixlib', 'coolprop',
    'tespy', 'FMU', 'co-simulation', 'cosimulation', 'radiance', 'contam', 'wufi',
    'delphin', 'champs', 'thermosyspro', 'thermopower', 'thermofluidstream',
    'building simulation', 'building energy model', 'thermal simulation',
    'thermal model', 'thermal modelling', 'thermal modeling', 'energy model',
    'energy simulation', 'HVAC model', 'HVAC simulation', 'airflow model',
    'heat transfer model', 'heat-transfer model', 'hygrothermal',
    '建筑模拟', '建筑仿真', '建筑能耗', '热工模型', '热工模拟', '能耗模型',
    '能耗模拟', '物理建模', '物理模型', '暖通仿真', '流体仿真', '流体模型',
    'Gebäudesimulation', 'Gebäudemodell', 'Wärmemodell',
)
TOOL_RE = re.compile(rf"\b(?:{TOOLS})\b", re.I)
PHYSICS_RE = re.compile(PHYSICS, re.I)
MODEL_RE = re.compile(MODELING, re.I)
EXCLUDED_RE = re.compile(r"gpqa|mmlu|nemotron.cc|fineweb|dclm|dolma|modigen|modbench|xmufst|nekaise.bench", re.I)
CORE_SOURCES = set("""energyplus-api energyplus-docs openmodelica-docs modelica-spec
modelica_buildings buildingspy openstudio-docs openstudio_hpxml_docs resstock_docs
urbanopt_docs eppy boptest_docs mosaik_docs soep gh_aixlib gh_buildingsystems
gh_coolprop gh_ebcpy gh_energym gh_energyplus gh_geomeppy gh_greenhouses-library
gh_helics gh_honeybee-energy gh_ideas gh_modelica-buildings gh_modelica-ibpsa
gh_modelicastandardlibrary gh_ochre gh_openstudio gh_openstudio-hpxml
gh_openstudio-standards gh_project1-boptest gh_radiance gh_resstock gh_comstock
gh_sinergym gh_teaser gh_tespy gh_fmi-standard gh_archetypal gh_cityenergyanalyst
gh_citylearn gh_pvlib-python gh_pysam gh_bifacial-radiance gh_fds gh_cfast""".split())
RESEARCH_SOURCES = {'ibpsa', 'modelica_conf'}
CODE_EXTENSIONS = {'.mo', '.mos', '.py', '.cpp', '.cc', '.c', '.h', '.hh', '.hpp', '.f90', '.f', '.idf', '.idd', '.idd.in', '.cal', '.rad', '.order'}
DOC_EXTENSIONS = {'.md', '.rst', '.tex', '.adoc', '.1'}
OMIT_PARTS = {'.git', '.github', 'node_modules', 'third_party', 'third-party', 'vendor', 'vendors', '_build', 'build', 'dist', 'referenceResults', 'ReferenceResults', 'obsolete', 'Obsolete'}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def emit(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False) + '\n')


def body_text(text):
    # Remove only the corpus publisher's fixed provenance preamble, not code.
    if text.startswith('# ') and '\nsource: ' in text[:2000] and '\n---\n' in text[:4000]:
        return text.split('\n---\n', 1)[1].strip()
    return text.strip()


def classify(row, fulltext_hit=False):
    title = row.get('title', '')
    location = ' '.join(str(row.get(k) or '') for k in ('id', 'title', 'url', 'source'))
    if EXCLUDED_RE.search(location):
        return 'excluded', 'excluded dataset or benchmark source'
    source = row.get('source', '')
    if source in CORE_SOURCES:
        return 'core', 'named modeling software, library or documentation source'
    if source in RESEARCH_SOURCES:
        return 'research', 'building simulation or Modelica conference collection'
    named = bool(TOOL_RE.search(title))
    physical_model = bool(PHYSICS_RE.search(title) and MODEL_RE.search(title))
    if source == 'google_patents':
        if named or physical_model:
            return 'patent_supplement', 'patent title explicitly names modeling software or physical/energy modeling'
        if fulltext_hit:
            return 'mention_candidate', 'patent full-text match; relevance not established'
    elif named or physical_model:
        return 'research', 'title names modeling software or combines physics and modeling terms'
    elif fulltext_hit:
        return 'mention_candidate', 'full-text match only; may be a citation or incidental mention'
    return None, None


def categories(text):
    groups = {
        'modelica': r'modelica|dymola|aixlib|ibpsa|besmod|thermopower|thermosyspro',
        'energyplus_bes': r'energyplus|energy.plus|openstudio|building.simulat|建筑.*(?:模拟|仿真)',
        'other_bes': r'trnsys|esp-r|ida.ice|designbuilder|wufi|delphin|champs',
        'thermal_fluid': r'therm|heat|hvac|fluid|airflow|refriger|psychrom|hydronic|热|暖通|流体',
        'controls_calibration': r'control|calibrat|parameter.estim|optim|boptest|控制|校准|优化',
        'numerics_coupling': r'fmi|fmu|co.simulat|differential|numerical|solver|数值|方程',
        'solar_daylight': r'radiance|daylight|solar|photovoltaic|pvlib|日照|太阳',
    }
    return [k for k, pattern in groups.items() if re.search(pattern, text, re.I)]


def safe_path(root, relative):
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError('Source path escapes corpus')
    return candidate


def pin_text(out, data):
    digest = sha(data)
    target = out / 'objects' / digest[:2] / (digest + '.txt')
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        # Parallel source acquisitions can discover identical bytes. Publish only
        # a complete object so another worker cannot observe a partial write.
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        temporary.replace(target)
    elif sha(target.read_bytes()) != digest:
        raise ValueError('Pinned object hash mismatch')
    return digest, str(target.relative_to(out))


def audit(root, out, keyword_config=None):
    out.mkdir(parents=True, exist_ok=True)
    started = now()
    snapshots = out / 'source-manifests'
    snapshots.mkdir(exist_ok=True)
    manifest_receipts = []
    for source in sorted((root / 'manifest').glob('*.jsonl')):
        target = snapshots / source.name
        for attempt in range(3):
            before = source.stat()
            shutil.copyfile(source, target)
            after = source.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns):
                break
        else:
            raise RuntimeError('Manifest changed during all snapshot attempts: ' + source.name)
        manifest_receipts.append({'name': source.name, 'sha256': sha(target.read_bytes()), 'bytes': after.st_size})
    policy = policy_at(root)
    dump(out / 'publisher-policy.json', policy)
    dump(out / 'manifest-snapshot.json', {'started_at': started, 'finished_at': now(), 'manifests': manifest_receipts})
    scan = out / 'fulltext-matches.txt'
    # Fixed-string discovery avoids expensive Unicode word-boundary regex scans
    # over tens of gigabytes. Broad hits remain candidates, not automatic admission.
    command = ['rg', '-l', '-i', '-F', '--threads', '4', '--glob', '*.md']
    additional = json.loads(keyword_config.read_text()) if keyword_config else {}
    phrases = [phrase for group in additional.get('concepts', []) for phrase in group['aliases']]
    for term in dict.fromkeys((*FULLTEXT_TERMS, *phrases)):
        command.extend(['-e', term])
    command.extend(['--', str(root / 'corpus')])
    with scan.open('w') as hits, (out / 'fulltext-scan.stderr').open('w') as errors:
        result = subprocess.run(command, stdout=hits, stderr=errors)
    dump(out / 'fulltext-scan.json', {'command': command, 'returncode': result.returncode, 'finished_at': now(), 'scope': 'Every readable *.md in the published corpus directory; lexical discovery, not semantic completeness.'})
    if result.returncode not in (0, 1):
        raise RuntimeError('Full-text discovery had errors; inspect fulltext-scan.stderr')
    hit_ids = {Path(line).stem for line in scan.read_text().splitlines()}
    counts, sources, tiers, errors, duplicates = Counter(), Counter(), Counter(), [], []
    unique = {}
    with (out / 'local-candidates.jsonl').open('w') as candidates, (out / 'local-documents.jsonl').open('w') as documents:
        for manifest in sorted(snapshots.glob('*.jsonl')):
            with manifest.open() as rows:
                for line in rows:
                    row = json.loads(line)
                    counts['manifest_rows'] += 1
                    sources[row.get('source', 'unknown')] += 1
                    admitted = eligible(row, policy)
                    counts['publisher_eligible' if admitted else 'outside_publisher_view'] += 1
                    tier, reason = classify(row, row['id'] in hit_ids)
                    matched = keyword_matches(row.get('title') or '', additional)
                    if matched and tier not in ('core', 'research', 'excluded'):
                        tier = 'patent_supplement' if row.get('source') == 'google_patents' else 'research'
                        reason = 'title matches explicit multilingual modeling/building-rating phrase: ' + ', '.join(matched)
                    if tier is None:
                        continue
                    record = {k: row.get(k) for k in ('id', 'title', 'url', 'source', 'license', 'topic', 'format', 'fetched_at', 'sha256', 'corpus_sha256')}
                    record.update(tier=tier, selection_reason=reason, manifest=manifest.name, publisher_eligible=admitted, origin='existing_corpus', matched_concepts=matched)
                    emit(candidates, record)
                    counts['candidates_' + tier] += 1
                    if not admitted or tier in ('excluded', 'mention_candidate'):
                        continue
                    try:
                        corpus = safe_path(root, row.get('corpus_path') or 'corpus/' + row['id'] + '.md')
                        raw = corpus.read_bytes()
                        if sha(raw) != row.get('corpus_sha256'):
                            raise ValueError('Published corpus hash differs from snapshotted manifest')
                        chosen, representation = raw, 'publisher_corpus_text'
                        url_path = row.get('url', '').split('?', 1)[0]
                        extension = PurePosixPath(url_path).suffix.lower()
                        if row.get('source', '').startswith('gh_') and extension in CODE_EXTENSIONS | DOC_EXTENSIONS:
                            original = safe_path(root, row['raw_path']).read_bytes()
                            if sha(original) != row['sha256']:
                                raise ValueError('Original code/document hash differs from manifest')
                            chosen, representation = original, 'verbatim_original_code_or_document'
                            counts['originals_restored'] += 1
                        text = chosen.decode('utf-8-sig')
                        if not text.strip() or '\x00' in text:
                            raise ValueError('Empty or binary-looking text')
                        digest, object_path = pin_text(out, chosen)
                        duplicate_key = sha(body_text(text).encode())
                        record.update(text_sha256=digest, object_path=object_path, chars=len(text), bytes=len(chosen), representation=representation,
                                      categories=categories(row.get('title', '') + ' ' + row.get('source', '') + ' ' + text[:20000]),
                                      duplicate_key=duplicate_key, verified_at=now(), original_version='as recorded in source URL; not necessarily latest')
                        if duplicate_key in unique:
                            record['duplicate_of'] = unique[duplicate_key]
                            duplicates.append({'id': row['id'], 'duplicate_of': unique[duplicate_key]})
                        else:
                            unique[duplicate_key] = row['id']
                        emit(documents, record)
                        counts['verified_documents'] += 1
                        counts['verified_characters'] += len(text)
                        tiers[tier] += 1
                    except (OSError, ValueError, KeyError) as exc:
                        errors.append({'id': row['id'], 'error': str(exc), 'manifest': manifest.name})
            dump(out / 'audit-progress.json', {'finished_manifest': manifest.name, 'counts': counts, 'errors': len(errors), 'at': now()})
    if policy_at(root) != policy:
        raise RuntimeError('Publisher eligibility changed during audit; repeat snapshot and selection')
    dump(out / 'local-summary.json', {'started_at': started, 'finished_at': now(), 'counts': counts, 'by_source': sources, 'verified_by_tier': tiers, 'fulltext_match_documents': len(hit_ids), 'exact_duplicates': len(duplicates), 'unique_document_bodies': len(unique), 'errors': errors, 'training_started': False, 'semantic_recall': 'not measured; full-text-only matches remain candidates'})
    dump(out / 'local-duplicates.json', duplicates)


def keyword_matches(text, config):
    """Explicit phrases only: ambiguous acronyms are query hints, never aliases."""
    folded = text.casefold()
    return [group['id'] for group in config.get('concepts', [])
            if any(phrase.casefold() in folded for phrase in group['aliases'])
            and (not group.get('requires_building_context') or any(
                anchor.casefold() in folded for anchor in config.get('building_context', [])))]


def include_repo_file(path, prefixes=()):
    p = PurePosixPath(path)
    if p.is_absolute() or '..' in p.parts or any(part in OMIT_PARTS for part in p.parts):
        return False
    if EXCLUDED_RE.search(path):
        return False
    if prefixes and not any(path.startswith(prefix) for prefix in prefixes):
        return False
    return (any(p.name.lower().endswith(extension) for extension in CODE_EXTENSIONS | DOC_EXTENSIONS)
            or p.name.lower().startswith(('readme', 'license', 'copying')))


def acquire_repo(spec, out):
    repo = spec['repo']
    if not re.fullmatch(r'[\w.-]+/[\w.-]+', repo):
        raise ValueError('Invalid GitHub repository')
    root = out / 'repositories' / repo.replace('/', '__')
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / 'receipt.json'
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    if (previous.get('status') == 'complete' and previous.get('counts', {}).get('documents', 0) > 0
            and previous.get('requested_ref') == spec.get('ref', 'HEAD')
            and previous.get('selected_prefixes') == spec.get('prefixes', [])
            and previous.get('selected_extensions') == sorted(CODE_EXTENSIONS | DOC_EXTENSIONS)
            and previous.get('omitted_path_components') == sorted(OMIT_PARTS)
            and (root / 'documents.jsonl').is_file()):
        return previous
    if previous:
        dump(root / 'attempts' / (sha(json.dumps(previous, sort_keys=True).encode()) + '.json'), previous)
    receipt = {'repo': repo, 'requested_ref': spec.get('ref', 'HEAD'), 'started_at': now(), 'status': 'fetching', 'training_started': False}
    try:
        ref = spec.get('ref', 'HEAD')
        listing = subprocess.run(['git', '-c', 'credential.helper=', 'ls-remote', 'https://github.com/' + repo + '.git', ref, ref + '^{}'], capture_output=True, text=True, timeout=90, env={**__import__('os').environ, 'GIT_TERMINAL_PROMPT': '0'})
        if listing.returncode or not listing.stdout.strip():
            raise ValueError('Cannot resolve requested public ref: ' + listing.stderr[-400:])
        lines = listing.stdout.strip().splitlines()
        commit = next((line.split()[0] for line in lines if line.endswith('^{}')), lines[0].split()[0])
        if not re.fullmatch(r'[0-9a-f]{40}', commit):
            raise ValueError('Invalid resolved commit')
        url = f'https://codeload.github.com/{repo}/tar.gz/{commit}'
        archive = root / (commit + '.tar.gz')
        if not archive.exists():
            temporary = archive.with_suffix('.partial')
            deadline = time.monotonic() + 900
            with httpx.Client(timeout=60, follow_redirects=False, trust_env=False) as client, client.stream('GET', url) as response, temporary.open('wb') as handle:
                response.raise_for_status()
                count = 0
                for block in response.iter_bytes(1024 * 1024):
                    count += len(block)
                    if count > spec.get('max_archive_bytes', 800_000_000) or time.monotonic() > deadline:
                        raise ValueError('Archive acquisition exceeded explicit byte/time limit')
                    handle.write(block)
            temporary.replace(archive)
        counters = Counter()
        with tarfile.open(archive, 'r:gz') as tar, (root / 'documents.jsonl').open('w') as documents:
            for member in tar:
                counters['archive_members'] += 1
                parts = PurePosixPath(member.name).parts
                if len(parts) < 2 or not member.isfile():
                    continue
                path = '/'.join(parts[1:])
                if not include_repo_file(path, spec.get('prefixes', [])):
                    counters['outside_selected_file_types_or_scope'] += 1
                    continue
                if member.size > 20_000_000:
                    counters['oversized_text_files'] += 1
                    continue
                raw = tar.extractfile(member).read()
                try:
                    text = raw.decode('utf-8-sig')
                except UnicodeDecodeError:
                    counters['non_utf8_files'] += 1
                    continue
                if not text.strip() or '\x00' in text:
                    counters['empty_or_binary'] += 1
                    continue
                digest, object_path = pin_text(out, raw)
                record = {'id': f'git:{repo}@{commit}:{path}', 'title': repo + ': ' + path, 'url': f'https://github.com/{repo}/blob/{commit}/{path}',
                          'source': repo, 'origin': 'new_repository_snapshot', 'tier': 'core', 'commit': commit, 'requested_ref': ref, 'path': path,
                          'license': 'preserved in repository snapshot; not used as an admission gate', 'text_sha256': digest, 'object_path': object_path,
                          'chars': len(text), 'bytes': len(raw), 'duplicate_key': sha(body_text(text).encode()), 'categories': spec['categories'],
                          'representation': 'verbatim_repository_file', 'retrieved_at': now(), 'validated_as_executable': False}
                emit(documents, record)
                counters['documents'] += 1
                counters['characters'] += len(text)
        receipt.update(status='complete' if counters['documents'] else 'empty', commit=commit, archive_url=url, archive_bytes=archive.stat().st_size, archive_sha256=sha(archive.read_bytes()),
                       counts=counters, selected_extensions=sorted(CODE_EXTENSIONS | DOC_EXTENSIONS), selected_prefixes=spec.get('prefixes', []),
                       omitted_path_components=sorted(OMIT_PARTS), scope='All selected text/code files in one fixed public repository snapshot; no outputs/binaries or execution')
    except Exception as exc:
        receipt.update(status='failed', error=repr(exc))
    receipt['finished_at'] = now()
    dump(receipt_path, receipt)
    print(json.dumps({'repo': repo, 'status': receipt['status'], 'counts': receipt.get('counts'), 'error': receipt.get('error')}, ensure_ascii=False), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit', 'repositories'])
    parser.add_argument('--corpus', type=Path, default=Path('../nekaise-corpus'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sources', type=Path)
    parser.add_argument('--keywords', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.action == 'audit':
        audit(args.corpus.resolve(), args.out.resolve(), args.keywords)
    else:
        specs = json.loads(args.sources.read_text())['repositories']
        with ThreadPoolExecutor(max_workers=2) as pool:
            receipts = list(pool.map(lambda spec: acquire_repo(spec, args.out.resolve()), specs))
        dump(args.out / 'repository-summary.json', receipts)


if __name__ == '__main__':
    main()
