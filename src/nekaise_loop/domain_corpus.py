"""Explicit, pinned domain-source backend. Publisher corpus policy stays separate.

Metadata is copied into an immutable Studio index; original payloads stay in the
corpus-owned content store and are verified on read. No ML or sibling-repo imports.
"""
from contextlib import closing
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile

from .artifacts import atomic_write, canonical, digest
from .storage import now

ROLES = frozenset({'core', 'reference_pdf', 'research_candidates', 'patent_supplement', 'reference_history', 'reference_assets'})
FORMAT = 'domain_source_v1'
EXCLUDED = re.compile(r'gpqa|mmlu|nemotron.cc|fineweb|dclm|dolma|modigen|modbench|xmufst|nekaise.bench', re.I)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def inside(root, relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Domain artifact path must be relative and contained')
    result = (root / p).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError('Domain artifact escapes corpus root')
    return result


@lru_cache(maxsize=32)
def _verified(path, expected, inode, size, mtime, ctime):
    if file_hash(Path(path)) != expected:
        raise ValueError('Domain metadata hash mismatch: ' + path)


def verify_metadata(path, expected):
    s = path.stat()
    _verified(str(path), expected, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def is_domain(root):
    return (Path(root) / 'domain-source.json').is_file()


def source_id(original):
    return 'domain-' + hashlib.sha256(original.encode()).hexdigest()


def bind_source(workspace, corpus_root, snapshot, roles=('core',)):
    """Validate and bind a frozen publication; never follow latest at runtime."""
    corpus_root, snapshot = Path(corpus_root).resolve(), Path(snapshot).resolve()
    if not roles or len(set(roles)) != len(roles) or not set(roles) <= ROLES:
        raise ValueError('Choose nonempty, distinct known domain roles')
    if not snapshot.is_relative_to(corpus_root):
        raise ValueError('Domain snapshot is outside corpus root')
    raw = (snapshot / 'snapshot.json').read_bytes(); report = json.loads(raw)
    identity = {k: report[k] for k in ('schema', 'definition', 'inventories', 'deduplication', 'builder')}
    if digest(identity) != report['id'] or snapshot.name != report['id']:
        raise ValueError('Domain snapshot identity mismatch')
    for name, key in [('members.jsonl', 'members_sha256'), ('aliases.jsonl', 'aliases_sha256')]:
        verify_metadata(snapshot / name, report[key])
    directory = Path(workspace) / 'curriculum/domain-sources'; directory.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix='.bind-', dir=directory)); counts = {}; canonical_ids = {}
    try:
        with closing(sqlite3.connect(pending / 'index.sqlite3')) as db:
            db.execute('CREATE TABLE documents (id TEXT PRIMARY KEY, position TEXT UNIQUE NOT NULL, body_hash TEXT UNIQUE NOT NULL, forward INTEGER NOT NULL, title TEXT, original_id TEXT, metadata TEXT NOT NULL)')
            db.execute('CREATE TABLE aliases (alias_id TEXT PRIMARY KEY, canonical_id TEXT NOT NULL)')
            with (snapshot / 'members.jsonl').open() as f:
                for line in f:
                    row = json.loads(line); role = row['split']
                    if role not in ROLES: raise ValueError('Unknown domain role')
                    if EXCLUDED.search(' '.join(str(row.get(k, '')) for k in ('id', 'source', 'path', 'url', 'title', 'categories', 'text_artifact'))):
                        raise ValueError('Excluded dataset/source in domain publication: ' + row['id'])
                    ref = row['text_artifact']; payload = inside(corpus_root, ref['path'])
                    if ref.get('stage') != 'text' or not re.fullmatch('[a-f0-9]{64}', ref['sha256']):
                        raise ValueError('Invalid domain text artifact')
                    # Full validation before activation; references also remain verifiable.
                    content = payload.read_bytes()
                    if len(content) != ref['bytes'] or hashlib.sha256(content).hexdigest() != ref['sha256']:
                        raise ValueError('Domain payload hash mismatch: ' + row['id'])
                    text = content.decode('utf-8-sig')
                    if not text.strip() or '\x00' in text: raise ValueError('Empty/binary domain text')
                    sid = source_id(row['id'])
                    if row['duplicate_key'] in canonical_ids:
                        raise ValueError('Duplicate canonical domain body: ' + row['id'])
                    canonical_ids[row['duplicate_key']] = sid
                    try:
                        db.execute('INSERT INTO documents VALUES (?,?,?,?,?,?,?)', (sid, hashlib.sha256(sid.encode()).hexdigest(), row['duplicate_key'], int(role in roles), row.get('title', row['id']), row['id'], json.dumps(row)))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError('Duplicate canonical domain ID: ' + row['id']) from exc
                    counts[role] = counts.get(role, 0) + 1
            if counts != report['counts']: raise ValueError('Domain manifest counts mismatch')
            with (snapshot / 'aliases.jsonl').open() as f:
                for line in f:
                    alias = json.loads(line); sid = source_id(alias['id']); target = canonical_ids.get(alias['duplicate_key'])
                    if target is None: raise ValueError('Alias lacks a canonical body: ' + alias['id'])
                    if sid == target: continue
                    existing = db.execute('SELECT id FROM documents WHERE id=?', (sid,)).fetchone()
                    if existing: raise ValueError('Ambiguous domain source ID')
                    previous = db.execute('SELECT canonical_id FROM aliases WHERE alias_id=?', (sid,)).fetchone()
                    if previous and previous[0] != target: raise ValueError('Ambiguous alias ID')
                    db.execute('INSERT OR IGNORE INTO aliases VALUES (?,?)', (sid, target))
            if not sum(n for r, n in counts.items() if r in roles): raise ValueError('No forward domain documents')
            db.execute('CREATE INDEX forward_positions ON documents(forward,position)'); db.commit()
        # Bind exact snapshot, role selection, reader/ID contract and derived index bytes.
        descriptor = {'format': FORMAT, 'corpus_root': str(corpus_root), 'snapshot_path': str(snapshot.relative_to(corpus_root)),
                      'snapshot_sha256': hashlib.sha256(raw).hexdigest(), 'snapshot_id': report['id'],
                      'members_sha256': report['members_sha256'], 'aliases_sha256': report['aliases_sha256'],
                      'forward_roles': sorted(roles), 'counts_by_role': counts, 'index_sha256': file_hash(pending / 'index.sqlite3'),
                      'id_mapping': 'sha256-full-original-source-id-v1', 'admission_basis': 'operator_source_selection_v1'}
        target = directory / digest(descriptor)
        atomic_write(pending / 'domain-source.json', canonical(descriptor))
        if target.exists():
            DomainSource(target); shutil.rmtree(pending)
        else:
            try: pending.rename(target)
            except OSError:
                if not target.exists(): raise
                DomainSource(target)  # A concurrent identical bind must verify normally.
        return {'corpus_path': str(target), 'binding': descriptor}
    finally:
        if pending.exists(): shutil.rmtree(pending)


class DomainSource:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.descriptor = d = json.loads((self.root / 'domain-source.json').read_text())
        self.identity = digest(d)
        if d.get('format') != FORMAT or self.root.name != self.identity:
            raise ValueError('Domain source binding identity mismatch')
        self.corpus_root = Path(d['corpus_root']).resolve()
        self.snapshot = inside(self.corpus_root, d['snapshot_path'])
        verify_metadata(self.snapshot / 'snapshot.json', d['snapshot_sha256'])
        verify_metadata(self.snapshot / 'members.jsonl', d['members_sha256'])
        verify_metadata(self.snapshot / 'aliases.jsonl', d['aliases_sha256'])
        self.index = self.root / 'index.sqlite3'
        verify_metadata(self.index, d['index_sha256'])

    def connect(self):
        return sqlite3.connect(f'file:{self.index}?mode=ro&immutable=1', uri=True)

    def search(self, query='', prefix='', offset=0, limit=20):
        if offset < 0 or not 1 <= limit <= 200: raise ValueError('Invalid domain search bounds')
        clauses, args = ['id>=?', 'id<?'], [prefix, prefix+'\uffff']
        for word in query.split():
            clauses.append('(title LIKE ? OR original_id LIKE ?)'); args.extend(['%'+word+'%']*2)
        with closing(self.connect()) as db:
            found = db.execute("SELECT id,forward,metadata FROM documents WHERE "+' AND '.join(clauses)+' ORDER BY id LIMIT ? OFFSET ?', [*args, limit+1, offset]).fetchall()
        rows = []
        for sid, forward, raw in found[:limit]:
            r = json.loads(raw)
            rows.append({'id': sid, 'title': r.get('title', r['id']), 'source': r['source'], 'topic': ', '.join(r.get('categories', [])),
                         'license': r.get('license', ''), 'domain_role': r['split'], 'forward_eligible': bool(forward),
                         'original_document_id': r['id'], 'quality_note': r.get('quality_note', ''), 'eligible_candidate': True})
        return {'rows': rows, 'next_offset': offset+limit if len(found)>limit else None,
                'domain_snapshot': self.descriptor['snapshot_id'], 'forward_roles': self.descriptor['forward_roles']}

    def read(self, document_id, start=0, length=0, *, forward=False):
        if not re.fullmatch(r'domain-[0-9a-f]{64}', document_id) or start < 0 or length < 0:
            raise ValueError('Invalid domain source reference')
        with closing(self.connect()) as db:
            row = db.execute('SELECT id,forward,metadata FROM documents WHERE id=?', (document_id,)).fetchone()
            if row is None:
                row = db.execute('SELECT d.id,d.forward,d.metadata FROM documents d JOIN aliases a ON a.canonical_id=d.id WHERE a.alias_id=?', (document_id,)).fetchone()
        if row is None or forward and not row[1]: raise ValueError('Missing/unselected forward domain source')
        sid, selected, raw = row; r = json.loads(raw); ref = r['text_artifact']
        payload = inside(self.corpus_root, ref['path']).read_bytes()
        if len(payload) != ref['bytes'] or hashlib.sha256(payload).hexdigest() != ref['sha256']:
            raise ValueError('Domain payload hash mismatch: '+sid)
        text = payload.decode('utf-8-sig'); excerpt = text[start:start+length] if length else text[start:]
        if not excerpt.strip(): raise ValueError('Selected domain source span is empty')
        return {'id': sid, 'title': r.get('title', r['id']), 'url': r.get('url', ''), 'source': r['source'],
                'license': r.get('license', ''), 'topic': ', '.join(r.get('categories', [])), 'source_sha256': ref['sha256'],
                'manifest_sha256': self.descriptor['members_sha256'], 'text': excerpt, 'span_start': start,
                'span_length': len(excerpt), 'document_chars': len(text), 'selection_reason': 'Explicit domain source selection', 'replay': False,
                'domain_source_hash': self.identity, 'domain_snapshot': self.descriptor['snapshot_id'],
                'domain_role': r['split'], 'forward_eligible': bool(selected), 'original_document_id': r['id'],
                'source_provenance': r, 'quality_note': r.get('quality_note', ''), 'admission_basis': self.descriptor['admission_basis']}

    def admits_pinned(self, document):
        return (document.get('domain_source_hash') == self.identity and document.get('forward_eligible') is True
                and document.get('domain_role') in self.descriptor['forward_roles'])

    def inventory(self, workspace, cancelled):
        directory = Path(workspace) / 'curriculum/inventories'; directory.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=directory, suffix='.sqlite3'); os.close(fd); path = Path(temporary)
        count = 0
        try:
            with closing(sqlite3.connect(path)) as out, closing(self.connect()) as source:
                out.execute('CREATE TABLE documents(position TEXT NOT NULL, document_id TEXT NOT NULL, metadata TEXT NOT NULL)')
                out.execute('CREATE TABLE manifests(name TEXT PRIMARY KEY, stamp TEXT NOT NULL)')
                out.execute('INSERT INTO manifests VALUES (?,?)', ('domain-source.json', json.dumps({'domain_source_hash': self.identity})))
                for position, sid in source.execute('SELECT position,id FROM documents WHERE forward=1 ORDER BY position'):
                    if count % 1000 == 0 and cancelled():
                        from .processes import Cancelled
                        raise Cancelled('Domain inventory cancelled')
                    out.execute('INSERT INTO documents VALUES (?,?,?)', (position, sid, json.dumps({'id': sid, 'manifest_name': 'domain-source.json', 'domain_source_hash': self.identity}))); count += 1
                out.execute('CREATE UNIQUE INDEX positions ON documents(position)'); out.execute('CREATE UNIQUE INDEX identifiers ON documents(document_id)'); out.commit()
            checksum = file_hash(path); os.replace(path, directory/(checksum+'.sqlite3'))
            return {'sha256': checksum, 'documents': count, 'excluded_documents': sum(self.descriptor['counts_by_role'].values())-count,
                    'domain_source_hash': self.identity, 'domain_snapshot': self.descriptor['snapshot_id'],
                    'forward_roles': self.descriptor['forward_roles'], 'manifest_count': 1, 'created_at': now(),
                    'revision_policy': 'Frozen explicit domain snapshot; later additions require an operator continuation, never auto-follow latest'}
        finally: path.unlink(missing_ok=True)
