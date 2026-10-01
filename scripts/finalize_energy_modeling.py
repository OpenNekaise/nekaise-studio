"""Verify and export an offline domain bundle. No training/teacher/benchmark access.

Objects and source catalogs remain immutable. Selection is a reviewable preparation
proposal, not an activated curriculum or a claim of semantic completeness.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
from urllib.parse import urlsplit

from prepare_energy_modeling import body_text, dump, emit, now, safe_path, sha


def git_location(row):
    if row.get('origin') == 'new_repository_snapshot':
        return row['source'].lower(), row['path']
    url = urlsplit(row.get('url', ''))
    parts = url.path.strip('/').split('/')
    if url.hostname == 'raw.githubusercontent.com' and len(parts) >= 4:
        return '/'.join(parts[:2]).lower(), '/'.join(parts[3:])
    if url.hostname == 'github.com' and len(parts) >= 5 and parts[2] == 'blob':
        return '/'.join(parts[:2]).lower(), '/'.join(parts[4:])
    return None


def partition(row, text, new_paths):
    path = row.get('path', '')
    location = git_location(row)
    if location:
        path = location[1]
    normalized = '/' + path.lower()
    if path.endswith('.order'):
        return 'reference_assets', 'Modelica package ordering metadata; retained alongside source'
    if '/idd/versions/' in normalized or '/obsolete/' in normalized:
        return 'reference_assets', 'historical schema or explicitly obsolete component'
    if row.get('source') == 'NatLabRockies/Spawn' and path.startswith('energyplus/'):
        return 'reference_assets', 'bundled EnergyPlus; canonical EnergyPlus snapshot is collected separately'
    if 'precipitationschedules' in normalized:
        return 'reference_assets', 'large numerical weather schedule; retained as simulator input'
    if len(text) > 100000 and sum(c.isdigit() for c in text) / len(text) > 0.45:
        return 'reference_assets', 'numerical table dominates text; retained for modeling use'
    if row['origin'] == 'existing_corpus' and location in new_paths:
        return 'reference_history', 'same repository path acquired in the new pinned snapshot'
    if row['tier'] == 'patent_supplement':
        return 'patent_supplement', 'title-selected patent; separate from primary modeling material'
    if row['tier'] == 'research':
        return 'research_candidates', 'conference/source/title relevance; document-level quality not certified'
    if row['origin'] == 'new_web_pdf':
        return 'reference_pdf', 'original PDF and extracted text retained; equations and figures need targeted review'
    return 'core', 'named modeling source with verified readable bytes'


def records(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def finalize(out, sources):
    dump(out / 'readiness.json', {'readiness': 'building', 'training_started': False, 'started_at': now()})
    summary = json.loads((out / 'local-summary.json').read_text())
    spec = json.loads(sources.read_text())
    catalogs, receipts = [], []
    for family, entries, key in [('repositories', spec['repositories'], 'repo'), ('web-sources', spec['web_sources'], 'id')]:
        for entry in entries:
            directory = out / family / entry[key].replace('/', '__')
            receipt_path = directory / 'receipt.json'
            receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {'status': 'missing'}
            receipts.append({'source': entry[key], **receipt})
            if receipt.get('status') in ('complete', 'partial') and (directory / 'documents.jsonl').exists():
                catalogs.append(directory / 'documents.jsonl')
    # New pinned snapshots win exact duplicates; local versions remain attributable.
    catalogs.append(out / 'local-documents.jsonl')
    new_paths = {git_location(row) for path in catalogs if path.parent.parent.name == 'repositories' for row in records(path)}
    new_paths.discard(None)
    destination = out / 'dataset'
    destination.mkdir(exist_ok=True)
    split_names = ('core', 'research_candidates', 'patent_supplement', 'reference_assets', 'reference_history', 'reference_pdf')
    counts, chars, categories, sources_count, reasons = Counter(), Counter(), Counter(), Counter(), Counter()
    unique, failures = {}, []
    with ExitStack() as stack:
        handles = {split: stack.enter_context((destination / (split + '.manifest.jsonl')).open('w')) for split in split_names}
        full_text = stack.enter_context((destination / 'core.text.jsonl').open('w'))
        duplicates = stack.enter_context((destination / 'duplicates.jsonl').open('w'))
        for catalog in catalogs:
            for row in records(catalog):
                try:
                    raw = safe_path(out, row['object_path']).read_bytes()
                    if sha(raw) != row['text_sha256']:
                        raise ValueError('object hash mismatch')
                    text = raw.decode('utf-8-sig')
                    if not text.strip() or '\x00' in text:
                        raise ValueError('empty or NUL-containing text')
                    # Ignore publisher provenance only for body comparison. Export
                    # original text (including code whitespace) without rewriting.
                    key = sha(body_text(text).encode())
                    if key != row['duplicate_key']:
                        raise ValueError('body hash mismatch')
                    split, reason = partition(row, text, new_paths)
                    if key in unique:
                        emit(duplicates, {'id': row['id'], 'duplicate_of': unique[key], 'url': row['url'], 'text_sha256': row['text_sha256'], 'would_be_split': split})
                        counts['exact_duplicates'] += 1
                        continue
                    unique[key] = row['id']
                    entry = {**row, 'split': split, 'preparation_reason': reason, 'input_catalog': str(catalog.relative_to(out))}
                    entry.pop('duplicate_of', None)
                    emit(handles[split], entry)
                    counts[split] += 1
                    chars[split] += len(text)
                    reasons[reason] += 1
                    if split == 'core':
                        emit(full_text, {**entry, 'text': text})
                        categories.update(row.get('categories', []))
                        sources_count[row['source']] += 1
                except (OSError, KeyError, ValueError) as exc:
                    failures.append({'id': row.get('id'), 'catalog': str(catalog), 'error': str(exc)})
    exported = {path.name: {'sha256': sha(path.read_bytes()), 'bytes': path.stat().st_size} for path in destination.glob('*.jsonl')}
    report = {'finished_at': now(), 'training_started': False, 'curriculum_activated': False,
        'readiness': 'integrity_verified_core_available' if counts['core'] and not failures else 'integrity_failure',
        'meaning': 'Core export is verified source material, not simulated/compiled models or teacher-approved teaching targets.',
        'counts': counts, 'characters': chars, 'core_by_category_overlapping': categories, 'core_by_source': sources_count,
        'selection_reasons': reasons, 'verification_failures': failures, 'source_receipts': receipts,
        'local_scan': {k: summary[k] for k in ('counts', 'fulltext_match_documents', 'verified_by_tier')},
        'local_quarantined': summary['errors'], 'unresolved_names': spec['unresolved'], 'exports': exported,
        'limitations': ['Lexical/title discovery has unmeasured recall and precision; body-only matches stay in local-candidates.jsonl.',
            'Exact normalized-body dedup only; namespace-renamed libraries and translated/near-duplicate docs can overlap.',
            'Repository text/code snapshots do not execute code or fetch Git submodules/LFS bodies.',
            'PDF/OCR equations, figures, extracted older papers and non-UTF8 files require focused review.',
            'HEAD is a pinned development snapshot, not a claim of release compatibility.',
            'Web frontier exhaustion includes failed/unsupported links; inspect every source receipt.',
            'No private benchmark evidence was read; source exclusions are not a content-overlap guarantee.',
            'Dataset preparation does not change the paused campaign, source cursor, mixture or training format.']}
    dump(out / 'readiness.json', report)
    print(json.dumps({'readiness': report['readiness'], 'counts': counts, 'characters': chars, 'failures': len(failures)}), flush=True)
    if failures or not counts['core']:
        raise RuntimeError('Do not use bundle: integrity verification failed; inspect readiness.json')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    args = parser.parse_args()
    finalize(args.out.resolve(), args.sources)
