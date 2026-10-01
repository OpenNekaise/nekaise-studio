"""Bounded, cancellable HTTP acquisition shared by references and training prose.

Responses are pinned on first successful retrieval in workspace storage. The cache
retains actual response bytes; extraction never requires a second network download.
"""
import base64
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import threading
import time
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

from .artifacts import atomic_write, canonical, digest
from .processes import Cancelled
from .storage import now
from .web_access import AGENT, permitted_source_url, wait_turn

_HTTP_SLOTS = threading.BoundedSemaphore(4)
_CACHE_LOCKS = {}
_CACHE_GUARD = threading.Lock()


def normalize_url(url):
    """Preserve meaningful query parameters and punctuation; reject malformed URLs."""
    p = urlsplit(url)
    if (p.scheme not in {'http', 'https'} or not p.hostname or p.username or p.password
            or p.port not in {None, 80, 443} or any(c.isspace() for c in url)
            or '://' in p.path or '\\' in url):
        raise ValueError('Malformed public source URL; repair the nominated address')
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path or '/', p.query, ''))


def pause(seconds, cancelled, deadline):
    end = min(time.monotonic()+seconds, deadline)
    while time.monotonic() < end:
        if cancelled():
            raise Cancelled('Web acquisition cancelled')
        time.sleep(min(.1, max(0, end-time.monotonic())))
    if time.monotonic() >= deadline:
        raise TimeoutError('Web acquisition deadline reached')


def retry_delay(value, attempt):
    if value:
        try:
            return max(0, float(value))
        except ValueError:
            try:
                return max(0, (parsedate_to_datetime(value)-datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError, OverflowError):
                pass
    return 2**attempt


def response_bytes(workspace, url, *, cancelled=lambda:False, validate=None,
                   client_factory=httpx.Client, max_bytes=2_000_000,
                   allowed_status=(), cache_seconds=None, deadline=None):
    requested = normalize_url(url)
    deadline = min(deadline or float('inf'), time.monotonic()+45)
    key = digest({'url':requested, 'format':'raw_http_v1'})
    path = Path(workspace)/'curriculum/http'/f'{key}.json'
    with _CACHE_GUARD:
        lock = _CACHE_LOCKS.setdefault(str(path), threading.RLock())
    while not lock.acquire(timeout=.1):
        if cancelled():
            raise Cancelled('Cancelled waiting for source cache')
        if time.monotonic() >= deadline:
            raise TimeoutError('Source cache wait exceeded acquisition deadline')
    try:
        if path.exists():
            cached = json.loads(path.read_text())
            age = time.time()-datetime.fromisoformat(cached['retrieved_at']).timestamp()
            if cache_seconds is None or age < cache_seconds:
                # Revalidate the complete redirect chain for the caller's scope.
                for location in cached['redirect_chain']:
                    permitted_source_url(location)
                    if validate:
                        validate(location, False)
                raw = base64.b64decode(cached['body_base64'], validate=True)
                if hashlib.sha256(raw).hexdigest() != cached['sha256'] or len(raw)>max_bytes:
                    raise ValueError('Cached HTTP response integrity/size mismatch')
                if cached['status'] != 200 and cached['status'] not in allowed_status:
                    raise ValueError('Cached HTTP status is not admitted for this request')
                return cached
        current, chain = requested, []
        with client_factory(timeout=15, follow_redirects=False, trust_env=False,
                            headers={'User-Agent':AGENT+'/1.0'}) as client:
            for _ in range(5):
                chain.append(current)
                for attempt in range(3):
                    if cancelled():
                        raise Cancelled('Web acquisition cancelled')
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Web acquisition deadline reached')
                    permitted_source_url(current)
                    if validate:
                        validate(current, True)
                    while not _HTTP_SLOTS.acquire(timeout=.1):
                        if cancelled():
                            raise Cancelled('Cancelled waiting for HTTP slot')
                        if time.monotonic() >= deadline:
                            raise TimeoutError('HTTP slot deadline reached')
                    delay = None
                    try:
                        with client.stream('GET', current) as response:
                            if response.is_redirect:
                                current = normalize_url(urljoin(current, response.headers['location']))
                                break
                            if response.status_code == 429 or 500 <= response.status_code < 600:
                                delay = retry_delay(response.headers.get('retry-after'), attempt)
                                if attempt == 2 or delay > 15 or time.monotonic()+delay >= deadline:
                                    response.raise_for_status()
                            else:
                                if response.status_code not in allowed_status:
                                    response.raise_for_status()
                                raw = bytearray()
                                for part in response.iter_bytes():
                                    if cancelled():
                                        raise Cancelled('Web acquisition cancelled')
                                    raw.extend(part)
                                    if len(raw)>max_bytes:
                                        raise ValueError('Source exceeded byte budget')
                                    if time.monotonic()>=deadline:
                                        raise TimeoutError('Source exceeded acquisition deadline')
                                result = {'format':'raw_http_v1', 'requested_url':requested, 'url':current,
                                    'redirect_chain':list(chain), 'status':response.status_code,
                                    'content_type':response.headers.get('content-type','').split(';')[0].lower(),
                                    'encoding':response.encoding or 'utf-8', 'retrieved_at':now(),
                                    'sha256':hashlib.sha256(raw).hexdigest(),
                                    'body_base64':base64.b64encode(raw).decode(), 'bytes':len(raw)}
                                atomic_write(path, canonical(result))
                                return result
                    except (httpx.TimeoutException, httpx.NetworkError):
                        if attempt == 2:
                            raise
                        delay = 2**attempt
                    finally:
                        _HTTP_SLOTS.release()
                    if delay is not None:
                        pause(delay, cancelled, deadline)
                else:
                    raise RuntimeError('HTTP acquisition exhausted retries')
        raise ValueError('Source redirect limit exceeded')
    finally:
        lock.release()


class WebSession:
    """One worker-owned acquisition batch; per-host robots and cached raw responses."""
    def __init__(self, workspace, cancelled=lambda:False, client_factory=httpx.Client):
        self.workspace, self.cancelled, self.client_factory = workspace, cancelled, client_factory
        self._robots, self._lock = {}, threading.Lock()

    def robots(self, url, deadline=None):
        p = urlsplit(url)
        origin = urlunsplit((p.scheme,p.netloc,'','',''))
        with self._lock:
            if origin in self._robots:
                return self._robots[origin]
        def same_host(location, network):
            q=urlsplit(location)
            if q.netloc != p.netloc or (p.scheme=='https' and q.scheme!='https'):
                raise ValueError('robots redirect left its host or downgraded HTTPS')
        record=response_bytes(self.workspace, origin+'/robots.txt', cancelled=self.cancelled,
            validate=same_host, client_factory=self.client_factory, max_bytes=1_000_000,
            allowed_status=(404,), cache_seconds=1800, deadline=deadline)
        text=base64.b64decode(record['body_base64']).decode('utf-8',errors='replace')
        parser=RobotFileParser(record['url'])
        parser.parse(text.splitlines() if record['status']==200 else [])
        delay=max(1,parser.crawl_delay(AGENT) or parser.crawl_delay('*') or 1)
        rate=parser.request_rate(AGENT) or parser.request_rate('*')
        if rate and rate.requests:
            delay=max(delay,rate.seconds/rate.requests)
        result=(parser,delay,{k:v for k,v in record.items() if k!='body_base64'} | {'text':text})
        with self._lock:
            self._robots[origin]=result
        return result

    def fetch(self, url, scope, *, deadline=None, max_bytes=2_000_000):
        deadline=min(deadline or float('inf'),time.monotonic()+45)
        evidence={}
        def validate(location, network):
            if not scope(location):
                raise ValueError('Redirect or source left the Teacher-selected acquisition scope')
            robot,delay,record=self.robots(location,deadline)
            if not robot.can_fetch(AGENT,location):
                raise ValueError('Publisher robots.txt excludes this page')
            evidence[urlsplit(location).netloc]=record
            if network:
                wait_turn(location,delay,self.cancelled,deadline=deadline)
        result=response_bytes(self.workspace,url,cancelled=self.cancelled,validate=validate,
            client_factory=self.client_factory,deadline=deadline,max_bytes=max_bytes)
        return result, list(evidence.values())


def extract_page(record, request):
    from .curriculum_research import TrainingPageText
    if record['content_type'] not in {'text/html','application/xhtml+xml','text/plain','text/markdown','application/json'}:
        raise ValueError('Unsupported research content type: '+record['content_type'])
    raw=base64.b64decode(record['body_base64'])
    decoded=raw.decode(record['encoding'],errors='replace')
    links=[]
    if record['content_type'] in {'text/html','application/xhtml+xml'}:
        parser=TrainingPageText(); parser.feed(decoded)
        decoded=''.join(parser.main_parts or parser.parts); links=parser.links
    text='\n'.join(line.rstrip() for line in decoded.splitlines() if line.strip())
    if len(text)<80:
        raise ValueError('Research page contains insufficient readable text')
    return {'id':'research-'+digest([record['url'],record['sha256']])[:24], 'url':record['url'],
        'requested_url':request['url'], 'title':request['title'],'purpose':request['purpose'],
        'text':text,'links':links,'bytes':len(raw),'document_chars':len(text),
        'span_start':0,'span_length':len(text),'excerpt_truncated':False,
        'retrieved_at':record['retrieved_at'],'content_type':record['content_type'],
        'source_sha256':record['sha256'],'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'extractor':TrainingPageText.version,'topic':'general curriculum','replay':False,
        'redirect_chain':record['redirect_chain'], 'license':'not_assessed'}
