"""Teacher-authorized web prose: immutable sources, bounded collection, exact coverage.

Teacher-selected prose follows the campaign's recorded admission policy. License
checks apply only to historical license_evidence_v1. No benchmark feed is consulted.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import re
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from .artifacts import atomic_write, canonical, digest
from .curriculum_research import fetch_source
from .curriculum_types import WebTrainingPermission, WebTrainingSelection
from .processes import Cancelled
from .storage import now
from .web_access import AGENT, permitted_source_url, robots, wait_turn


class SourceAdmissionError(ValueError):
    pass


def verify_permission(permission, evidence, seed_url):
    if urlsplit(permission['license_url']).netloc != urlsplit(seed_url).netloc or urlsplit(evidence['url']).netloc != urlsplit(seed_url).netloc:
        raise SourceAdmissionError('License evidence must come from the source publisher, not an unrelated license deed')
    quote = ' '.join(permission['evidence_quote'].split()).casefold()
    text = ' '.join(evidence['text'].split()).casefold()
    offset = text.find(quote)
    if offset < 0:
        raise SourceAdmissionError('Declared license quote is absent from retrieved evidence; source remains reference-only')
    # A generic prefix of an NC/ND license is not evidence for a permissive label.
    context = text[max(0,offset-40):offset+len(quote)+100]
    if re.search(r'non[\s-]?commercial|no[\s-]?derivatives|by[\s-]+(?:nc|nd)\b', context):
        raise SourceAdmissionError('License evidence includes NC/ND restrictions; source remains reference-only')
    code = permission['license']
    normalized_quote = re.sub(r'[^a-z0-9]', '', quote)
    if code.startswith('CC-BY'):
        version = code.rsplit('-',1)[1].replace('.','')
        sa = '-SA-' in code
        forms = ['ccby'+('sa' if sa else '')+version,
                 'creativecommonsattribution'+('sharealike' if sa else '')+version]
    else:
        forms = {'CC0-1.0':['cc0','creativecommonszero'], 'public-domain':['publicdomain'],
                 'MIT':['mitlicense'], 'Apache-2.0':['apachelicenseversion20','apachelicense20'],
                 'PSF-2.0':['pythonsoftwarefoundationlicenseversion2','psflicenseversion2']}[code]
    if not any(form in normalized_quote for form in forms):
        raise SourceAdmissionError('The quoted publisher evidence does not identify the declared license')


def normalized(url):
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc, p.path or '/', p.query, ''))


def in_collection(url, prefix):
    u, p = urlsplit(url), urlsplit(prefix)
    path = p.path.rstrip('/')
    return (u.scheme, u.netloc) == (p.scheme, p.netloc) and (u.path == path or u.path.startswith(path+'/')) and not u.query


def collect(ctx, request):
    """One immutable bounded collection, with paid/fetched work reused after failure."""
    policy = ctx.config.curriculum_loop.web_training_policy
    licensed = policy == 'license_evidence_v1'
    selection_type = WebTrainingPermission if licensed else WebTrainingSelection
    permission = selection_type.model_validate(request.get('training') or {}).model_dump()
    url, prefix = normalized(request['url']), permission['collection_prefix']
    permitted_source_url(url)
    if prefix and not in_collection(url, prefix):
        raise ValueError('Training collection prefix must contain its same-origin seed URL')
    if not prefix and permission['max_pages'] != 1:
        raise ValueError('Multi-page training collection requires an explicit selected path prefix')
    identity = digest({'url':url, 'permission':permission, 'contract':'web_training_v1',
                       **({} if licensed else {'admission_policy':policy})})
    path = ctx.engine.settings.workspace/'curriculum'/'web'/f'{identity}.json'
    if path.exists():
        import json
        journal = json.loads(path.read_text())
    else:
        evidence_key = None
        if licensed:
            evidence = fetch_source({'url': permission['license_url'], 'title': 'Source license evidence',
                'purpose': permission['scope_reason']}, cancelled=ctx.cancelled, full_text=True)
            evidence.pop('links', None)
            evidence_key = ctx.artifacts.put(evidence)
            try:
                verify_permission(permission, evidence, url)
            except SourceAdmissionError as exc:
                raise SourceAdmissionError(str(exc)+'; evidence artifact '+evidence_key) from exc
        journal = {'format': 'web_training_collection_v1', 'permission': permission,
            'admission_policy': policy, 'license_checked': licensed,
            'license_evidence_artifact': evidence_key, 'pages': [],
            'pending': [url], 'visited': [], 'failures': [], 'collected_chars': 0,
            'created_at': now(), 'complete': False}
        atomic_write(path, canonical(journal))
    if journal['complete']:
        return ctx.artifacts.put(journal)
    if not journal['pages'] and not journal['pending']:
        failed = {r['url'] for r in journal['failures']}
        journal['pending'] = [u for u in journal['visited'] if u in failed]
        journal['visited'] = [u for u in journal['visited'] if u not in failed]
    robot, delay, robot_evidence = robots(url)
    journal['robots_artifact'] = ctx.artifacts.put(robot_evidence)

    def retrieve(page):
        try:
            permitted_source_url(page)
            if not robot.can_fetch(AGENT, page):
                raise ValueError('Publisher robots.txt excludes this page')
            wait_turn(page, delay, ctx.cancelled)
            return fetch_source({'url': page, 'title': request['title'], 'purpose': request['purpose']},
                                cancelled=ctx.cancelled, full_text=True, readable=True), None
        except Cancelled:
            raise
        except (httpx.HTTPError, ValueError, OSError) as exc:
            return None, str(exc)[:500]

    # Four bounded HTTP fetches, independent of LLM concurrency and budgets.
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix='gpc-web') as pool:
        while journal['pending'] and len(journal['visited']) < permission['max_pages']:
            if ctx.cancelled():
                raise Cancelled('Web training collection cancelled')
            batch = journal['pending'][:min(4, permission['max_pages']-len(journal['visited']))]
            results = list(pool.map(retrieve, batch))
            # Commit a whole batch; interrupted fetches can be repeated, never counted as training.
            journal['pending'] = journal['pending'][len(batch):]
            for page, (source, error) in zip(batch, results):
                journal['visited'].append(page)
                if error:
                    journal['failures'].append({'url': page, 'error': error})
                    continue
                if urlsplit(source['url']).netloc != urlsplit(url).netloc or (prefix and not in_collection(source['url'], prefix)):
                    journal['failures'].append({'url': page, 'error': 'Redirect left the authorized collection'})
                    continue
                if not robot.can_fetch(AGENT, source['url']):
                    journal['failures'].append({'url':page, 'error':'Redirect target excluded by robots.txt'})
                    continue
                permitted_source_url(source['url'])
                links = source.pop('links', [])
                source.update(license=permission['license'] if licensed else 'not_assessed', training_permission=permission,
                    admission_policy=policy, license_checked=licensed,
                    license_evidence_artifact=journal['license_evidence_artifact'],
                    extractor='readable_body_v1', robots_artifact=journal['robots_artifact'],
                    selection_reason=request['purpose'], training_eligible=True)
                key = ctx.artifacts.put(source)
                if source['text_sha256'] not in {p['text_sha256'] for p in journal['pages']}:
                    journal['pages'].append({'artifact': key, 'text_sha256': source['text_sha256'],
                        'url': source['url'], 'chars': len(source['text'])})
                    journal['collected_chars'] += len(source['text'])
                if prefix:
                    seen = set(journal['visited']) | set(journal['pending'])
                    for link in links:
                        candidate = normalized(urljoin(source['url'], link))
                        if in_collection(candidate, prefix) and candidate not in seen:
                            journal['pending'].append(candidate)
                            seen.add(candidate)
                            if len(journal['pending']) >= permission['max_pages']:
                                break
            if journal['collected_chars'] >= 64_000_000:
                journal['bounded_by'] = '64M extracted characters per collection'
                break
            atomic_write(path, canonical(journal))
    if not journal['pages']:
        atomic_write(path, canonical(journal))
        raise ValueError('Authorized web collection has no usable pages; inspect '+str(path))
    journal.update(complete=True, finished_at=now(),
        basis='Bounded retrieved collection, not a claim that the website was exhaustively crawled')
    atomic_write(path, canonical(journal))
    return ctx.artifacts.put(journal)


def add_collections(ctx, research):
    """Original research snapshots remain reference-only; attach distinct authorization."""
    if not ctx.config.curriculum_loop.web_training:
        return research
    policy = ctx.config.curriculum_loop.web_training_policy
    requests = [r for r in research['plan']['sources']
                if policy == 'teacher_selected_v1' or r.get('training')]
    collections, failures = [], []
    for request in requests:
        try:
            collections.append(collect(ctx, request))
        except Cancelled:
            raise
        except (ValueError, httpx.HTTPError, OSError) as exc:
            # This happens BEFORE the teaching plan. The Teacher receives the
            # explicit admission/supply failure and chooses its actual recipe;
            # the host never backfills, changes a planned ratio or invents text.
            failures.append({'url':request['url'], 'error':str(exc)[:1500],
                             'status':'reference_only; source acquisition failed' if policy == 'teacher_selected_v1'
                                      else 'reference_only; no direct training permission admitted'})
    return {**research, 'training_collections':collections, 'training_admission_failures':failures,
            'web_training_policy':policy}


def training_rows(ctx, work, references):
    """Expose unread source suffixes. Token preparation selects whole span prefixes."""
    if not ctx.config.curriculum_loop.web_training:
        return []
    prior = work['before'].get('web_coverage_artifact')
    positions = ctx.artifacts.get(prior)['positions'] if prior else {}
    seen, rows = set(), []
    for collection_key in references.get('training_collections', []):
        collection = ctx.artifacts.get(collection_key)
        for entry in collection['pages']:
            sha = entry['text_sha256']
            if sha in seen:
                continue
            seen.add(sha)
            source = ctx.artifacts.get(entry['artifact'])
            if (not source.get('training_eligible') or source['text_sha256'] != sha
                    or hashlib.sha256(source['text'].encode()).hexdigest() != sha):
                raise ValueError('Web training source integrity or eligibility mismatch')
            for start in range(positions.get(sha, 0), len(source['text']), ctx.config.curriculum_loop.span_chars):
                text = source['text'][start:start+ctx.config.curriculum_loop.span_chars]
                rows.append({'id': 'web-'+digest([sha,start,len(text)])[:24], 'stream': 'corpus',
                    'text': text, 'learning_track': 'gpc', 'material_scope': 'general_prose',
                    'curriculum_unit_id': work['unit']['id'], 'web_span': True,
                    'document_id': source['id'], 'source_sha256': source['source_sha256'],
                    'text_sha256': sha, 'span_start': start, 'span_length': len(text),
                    'source_artifact': entry['artifact'], 'collection_artifact': collection_key,
                    'license': source['license'], 'url': source['url']})
    return rows


def coverage(ctx, work, rows):
    prior = work['before'].get('web_coverage_artifact')
    positions = dict(ctx.artifacts.get(prior)['positions']) if prior else {}
    sources, chars = {}, 0
    for row in rows:
        if not row.get('web_span'):
            continue
        key = row['source_artifact']
        if key not in sources:
            sources[key] = ctx.artifacts.get(key)
        source = sources[key]
        sha, start, length = row['text_sha256'], row['span_start'], row['span_length']
        if (not source.get('training_eligible') or source['text_sha256'] != sha
                or source['text'][start:start+length] != row['text'] or len(row['text']) != length
                or start != positions.get(sha, 0)):
            raise ValueError('Web coverage must consume exact fresh contiguous source spans')
        positions[sha] = start + length
        chars += length
    if not chars:
        return prior, 0
    return ctx.artifacts.put({'format': 'web_coverage_v1', 'positions': positions}), chars
