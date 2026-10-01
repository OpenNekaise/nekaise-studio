"""Acquire declared domain documentation into an isolated offline preparation area."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

from nekaise_loop.artifacts import Artifacts
from nekaise_loop.web_http import WebSession
from nekaise_loop.web_inventory import collect
from prepare_energy_modeling import dump, emit, now, pin_text, sha


def acquire(spec, out):
    workspace = out / 'web-acquisition'
    artifacts = Artifacts(workspace)
    ctx = SimpleNamespace(engine=SimpleNamespace(settings=SimpleNamespace(workspace=workspace)),
        artifacts=artifacts, cancelled=lambda: False,
        config=SimpleNamespace(curriculum_loop=SimpleNamespace(web_training_policy='teacher_selected_v1')))
    receipt = {'id': spec['id'], 'url': spec['url'], 'started_at': now(), 'training_started': False}
    destination = out / 'web-sources' / spec['id']
    destination.mkdir(parents=True, exist_ok=True)
    try:
        if spec.get('format') == 'pdf':
            host = urlsplit(spec['url']).netloc
            response, robots = WebSession(workspace).fetch(spec['url'], lambda url: urlsplit(url).netloc == host,
                max_bytes=80_000_000, deadline=time.monotonic()+120)
            data = base64.b64decode(response['body_base64'])
            if not data.startswith(b'%PDF-'):
                raise ValueError('Nominated PDF did not return PDF bytes')
            original = destination / 'original.pdf'
            original.write_bytes(data)
            text_path = destination / 'extracted.txt'
            subprocess.run(['pdftotext', '-layout', str(original), str(text_path)], check=True, timeout=120, capture_output=True)
            text = text_path.read_text()
            if len(text.strip()) < 200:
                raise ValueError('PDF has insufficient extractable text; OCR/visual review required')
            digest, object_path = pin_text(out, text.encode())
            rows = [{'id': 'pdf:'+response['sha256'], 'title': spec['id'], 'url': response['url'], 'source': spec['id'], 'origin': 'new_web_pdf',
                'tier': 'core', 'categories': spec['categories'], 'text_sha256': digest, 'duplicate_key': sha(text.strip().encode()),
                'object_path': object_path, 'chars': len(text), 'bytes': len(text.encode()), 'source_sha256': response['sha256'],
                'representation': 'pdftotext_layout', 'original_pdf': str(original.relative_to(out)), 'retrieved_at': response['retrieved_at'],
                'version_note': spec.get('version_note', 'See original PDF'), 'quality_note': 'Original PDF retained; equations/images and column order require visual checks.'}]
            dump(destination/'http-receipt.json', {k:v for k,v in response.items() if k != 'body_base64'})
            dump(destination/'robots.json', robots)
            receipt.update(status='complete', pages=None, characters=len(text), extraction='pdftotext -layout')
        else:
            request={'url': spec['url'], 'title': spec['id'], 'purpose': 'Operator-selected energy/physical modeling domain preparation; training remains paused',
                'training': {'collection_prefix': spec['prefix'], 'max_pages': spec.get('max_pages',256), 'max_seconds':300,
                             'seed_urls':spec.get('seed_urls',[]), 'sitemap_urls':spec.get('sitemap_urls',[])}}
            # Bounded, resumable acquisition; completed frontier and capped frontier
            # are not represented as complete coverage of a publisher's site.
            for _ in range(3):
                key=collect(ctx,request)
                collection=artifacts.get(key)
                if collection['complete']:
                    break
                if collection.get('next_retry_at',0) and collection['next_retry_at']>time.time():
                    break
            rows=[]
            for entry in collection['pages']:
                page=artifacts.get(entry['artifact']);text=page['text']
                digest,object_path=pin_text(out,text.encode())
                rows.append({'id':page['id'],'title':spec['id']+': '+page['url'],'url':page['url'],'source':spec['id'],
                    'origin':'new_web_page','tier':'core','categories':spec['categories'],'text_sha256':digest,
                    'duplicate_key':sha(text.strip().encode()),'object_path':object_path,'chars':len(text),'bytes':len(text.encode()),
                    'source_sha256':page['source_sha256'],'representation':page['extractor'],'retrieved_at':page['retrieved_at'],
                    'raw_response_artifact':page['raw_response_artifact'],'robots_artifact':page['robots_artifact'],
                    'license':'recorded source-use policy; no license check','quality_note':'HTML extraction; original bytes preserved for equation/code review'})
            receipt.update(status='complete' if collection['complete'] else 'partial',pages=len(rows),characters=sum(x['chars'] for x in rows),
                collection_artifact=key,bounded_by=collection.get('bounded_by'),remaining_frontier=len(collection['pending']),failures=collection['failures'],
                coverage_limit=request['training']['max_pages'],scope=spec['prefix'])
        with (destination/'documents.jsonl').open('w') as handle:
            for row in rows:emit(handle,row)
    except Exception as exc:
        receipt.update(status='failed',error=repr(exc))
    receipt['finished_at']=now()
    dump(destination/'receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    specs=json.loads(args.sources.read_text())['web_sources']
    with ThreadPoolExecutor(max_workers=3) as pool:
        receipts=list(pool.map(lambda spec:acquire(spec,args.out.resolve()),specs))
    dump(args.out/'web-summary.json',receipts)
