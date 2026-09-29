"""Teacher-directed web research with real, bounded source snapshots for Authors."""
import hashlib
from html.parser import HTMLParser
import ipaddress
import socket
import time
from urllib.parse import urljoin, urlsplit

import httpx

from .artifacts import digest
from .curriculum_types import ResearchPlan
from .storage import now


class PageText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)
        if tag in {"script", "style", "noscript"}:
            self.hidden += 1
        if tag in {"p", "br", "div", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, text):
        if not self.hidden:
            self.parts.append(text)


class TrainingPageText(PageText):
    """Conservative readable-body extraction; preserve explicit preformatted text."""
    excluded = {'script', 'style', 'noscript', 'nav', 'footer', 'header', 'aside', 'form'}

    def __init__(self):
        super().__init__()
        self.stack = []
        self.main_parts = []
        self.main_depth = 0

    def handle_starttag(self, tag, attrs):
        super().handle_starttag(tag, attrs)
        if tag in self.excluded and tag not in {'script', 'style', 'noscript'}:
            self.hidden += 1
        if tag in {'main', 'article'}:
            self.main_depth += 1
        if self.main_depth and tag in {'p', 'br', 'div', 'li', 'h1', 'h2', 'h3', 'tr', 'pre'}:
            self.main_parts.append('\n')

    def handle_endtag(self, tag):
        super().handle_endtag(tag)
        if tag in self.excluded and tag not in {'script', 'style', 'noscript'}:
            self.hidden = max(0, self.hidden-1)
        if tag in {'main', 'article'}:
            self.main_depth = max(0, self.main_depth-1)

    def handle_data(self, text):
        super().handle_data(text)
        if self.main_depth and not self.hidden:
            self.main_parts.append(text)


def public_url(url):
    parsed = urlsplit(url)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.port not in {None, 80, 443}):
        raise ValueError("Research URL must be public HTTP(S) without credentials or nonstandard ports")
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ValueError("Research URL resolved to a non-public address")


def fetch_source(request, *, cancelled=lambda: False, client_factory=httpx.Client, full_text=False, readable=False):
    url, deadline = request["url"], time.monotonic() + 45
    with client_factory(timeout=httpx.Timeout(15), follow_redirects=False, trust_env=False,
                        headers={"User-Agent": "Nekaise-Studio-Curriculum/1.0", "Accept": "text/html,text/plain,application/json"}) as client:
        for _ in range(5):
            if cancelled():
                from .processes import Cancelled
                raise Cancelled("Curriculum research cancelled")
            public_url(url)
            with client.stream("GET", url) as response:
                if response.is_redirect:
                    url = urljoin(url, response.headers["location"])
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";")[0].lower()
                if content_type not in {"text/html", "text/plain", "application/json", "application/xhtml+xml", "text/markdown"}:
                    raise ValueError("Unsupported research content type: " + content_type)
                parts, size = [], 0
                for block in response.iter_bytes():
                    if cancelled():
                        from .processes import Cancelled
                        raise Cancelled("Curriculum research cancelled")
                    size += len(block)
                    if size > 2_000_000 or time.monotonic() > deadline:
                        raise ValueError("Research source exceeded its byte or time budget")
                    parts.append(block)
                raw = b"".join(parts)
                decoded = raw.decode(response.encoding or "utf-8", errors="replace")
                links = []
                if content_type in {"text/html", "application/xhtml+xml"}:
                    parser = TrainingPageText() if readable else PageText()
                    parser.feed(decoded)
                    decoded = "".join((parser.main_parts or parser.parts) if readable else parser.parts)
                    links = parser.links
                text = "\n".join((line.rstrip() if readable else " ".join(line.split())) for line in decoded.splitlines() if line.strip())
                if len(text) < 80:
                    raise ValueError("Research page contains insufficient readable text")
                excerpt = text if full_text else text[:12000]
                return {"id": "research-" + digest([url, hashlib.sha256(raw).hexdigest()])[:24],
                        "title": request["title"], "url": url, "requested_url": request["url"],
                        "purpose": request["purpose"], "retrieved_at": now(), "content_type": content_type,
                        "source_sha256": hashlib.sha256(raw).hexdigest(), "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "bytes": size, "document_chars": len(text), "text": excerpt,
                        "span_start": 0, "span_length": len(excerpt), "excerpt_truncated": len(text) > len(excerpt),
                        "license": "research_reference_only", "topic": "general curriculum",
                        "selection_reason": "Teacher web research; reference for original synthetic material", "replay": False,
                        **({"links": links} if full_text else {})}
        raise ValueError("Research redirect limit exceeded")


def research(ctx, work):
    keys = (work["namespace"], work["sequence"])
    saved = ctx.store.one("SELECT research_artifact FROM curriculum_assignments WHERE namespace=? AND sequence=?", keys)
    if saved["research_artifact"]:
        return ctx.artifacts.get(saved["research_artifact"])
    # Save the research plan before fetching, so an HTTP failure does not waste
    # another Teacher call on an unchanged-stage retry.
    plan_key = "research-plan-" + digest(keys)
    path = ctx.engine.settings.workspace / "curriculum" / (plan_key + ".json")
    from .artifacts import atomic_write, canonical
    if path.exists():
        plan = ResearchPlan.model_validate_json(path.read_text()).model_dump()
    else:
        plan = ResearchPlan.model_validate(ctx.teacher.research({"unit": work["unit"],
            "gpc_cycle": work["gpc_cycle"], "instruction": "Search the live web for this unit; choose a few accessible primary HTML/text pages. Generate no benchmark items or benchmark feedback."})).model_dump()
        atomic_write(path, canonical(plan))
    sources, failures = [], []
    from .processes import Cancelled
    for request in plan["sources"]:
        try:
            sources.append(fetch_source(request, cancelled=ctx.cancelled))
        except Cancelled:
            raise
        except (httpx.HTTPError, ValueError, OSError) as exc:
            failures.append({"url": request["url"], "error": str(exc)[:500]})
    result = {"plan": plan, "sources": sources, "fetch_failures": failures,
              "unit_id": work["unit"]["id"], "basis": "Retrieved reference excerpts, not corpus downloads or benchmark results"}
    artifact = ctx.artifacts.put(result)
    if not sources:
        raise ValueError("No research page could be retrieved; inspect artifact " + artifact + "; retry these URLs or explicitly repair the saved research plan")
    ctx.store.execute("UPDATE curriculum_assignments SET research_artifact=? WHERE namespace=? AND sequence=?",
                      (artifact, *keys))
    return result
