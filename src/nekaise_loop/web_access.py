"""Small public-web crawler boundary: robots evidence, per-host pacing and exclusions."""
import hashlib
import threading
import time
from urllib.parse import urlsplit, urlunsplit, unquote
from urllib.robotparser import RobotFileParser

import httpx

from .curriculum_research import public_url
from .processes import Cancelled
from .storage import now

AGENT = 'Nekaise-Studio-Curriculum'
_LOCK = threading.Lock()
_NEXT = {}


def permitted_source_url(url):
    value = unquote(url).casefold()
    excluded = ('nemotron-cc', 'fineweb', 'dclm', 'dolma', 'gpqa', 'mmlu-pro', 'mmlu_pro',
                'commoncrawl.org', 'data.commoncrawl.org')
    if any(name in value for name in excluded):
        raise ValueError('Source belongs to an excluded general dataset or benchmark collection')
    public_url(url)


def robots(url):
    p = urlsplit(url)
    path = urlunsplit((p.scheme, p.netloc, '/robots.txt', '', ''))
    public_url(path)
    with httpx.Client(timeout=15, follow_redirects=False, trust_env=False,
                      headers={'User-Agent':AGENT+'/1.0'}) as client:
        with client.stream('GET', path) as response:
            code = response.status_code
            raw = bytearray()
            for part in response.iter_bytes():
                raw.extend(part)
                if len(raw) > 1_000_000:
                    raise ValueError('robots.txt exceeds the crawler evidence byte limit')
    parser = RobotFileParser(path)
    if code == 404:
        parser.parse([])
    elif code == 200:
        parser.parse(bytes(raw).decode('utf-8', errors='replace').splitlines())
    else:
        raise ValueError(f'Cannot establish crawler access from robots.txt: HTTP {code}')
    delay = max(1, parser.crawl_delay(AGENT) or parser.crawl_delay('*') or 1)
    rate = parser.request_rate(AGENT) or parser.request_rate('*')
    if rate and rate.requests:
        delay = max(delay, rate.seconds/rate.requests)
    return parser, delay, {'url':path, 'status':code, 'sha256':hashlib.sha256(raw).hexdigest(),
                          'text':bytes(raw).decode('utf-8', errors='replace'), 'retrieved_at':now()}


def wait_turn(url, delay, cancelled, *, deadline=None):
    host = urlsplit(url).netloc
    with _LOCK:
        turn = max(time.monotonic(), _NEXT.get(host, 0))
        if deadline is not None and turn >= deadline:
            raise TimeoutError('Host pacing exceeds the acquisition deadline')
        _NEXT[host] = turn+delay
    while time.monotonic() < turn:
        if cancelled():
            raise Cancelled('Web collection cancelled during host pacing')
        time.sleep(max(0, min(.1, turn-time.monotonic())))
