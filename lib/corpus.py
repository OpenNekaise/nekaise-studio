"""building_corpus — the open-book context: a building's data IN HAND, as text.

The SAME input is given to the teacher and to the small model (fairness), and the ontology is
just ONE source among tag/point lists, alarm exports, notes and trends — not the center. PDFs
and images need OCR/captioning (not yet wired) so they are reported as `skipped`, never silently
dropped. The text is capped so it fits the small model's context window; when a building's data
is larger than that window, the principled path is *retrieval* (select the relevant slice per
question), not a bigger dump.
"""
from __future__ import annotations

import re
from pathlib import Path

# One shared cap so the teacher and the student see the SAME amount of context (fair comparison).
MAX_CHARS = 48000
_TEXT = (".ttl", ".txt", ".md", ".csv")
# densest building facts first, so a cap drops the least-dense data last
_ORDER = {".ttl": 0, ".txt": 1, ".csv": 2, ".md": 3, ".xlsx": 4}
_SKIP = (".pdf", ".png", ".jpg", ".jpeg", ".pptx", ".docx")


def _xlsx_text(p: Path) -> str:
    import openpyxl
    wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        out.append(f"## sheet: {ws.title}")
        for row in ws.iter_rows(values_only=True):
            out.append("\t".join("" if c is None else str(c) for c in row))
    return "\n".join(out)


def building_corpus(bdir, max_chars: int = MAX_CHARS) -> tuple[str, list[str]]:
    """Return (text, skipped): the building's text data-in-hand (capped), and skipped binary files."""
    bdir = Path(bdir)
    files = sorted((p for p in bdir.rglob("*") if p.is_file()),
                   key=lambda p: (_ORDER.get(p.suffix.lower(), 9), str(p)))
    parts, skipped = [], []
    for p in files:
        s = p.suffix.lower()
        try:
            if s in _TEXT:
                body = p.read_text(errors="replace")
            elif s == ".xlsx":
                body = _xlsx_text(p)
            elif s in _SKIP:
                skipped.append(p.name)
                continue
            else:
                continue
            parts.append(f"# FILE: {p.relative_to(bdir)}\n{body}")
        except Exception:
            continue
    return ("\n\n".join(parts))[:max_chars], skipped


# --- retrieval over RAW corpus text (keep open-book context short for a small model) ----------

_STOP = frozenset((
    "what whats which where when whos whose how does did the and for its with from are there has "
    "have much many about you your using use into over why check would should this that these those "
    "give get pull grab show tell list name walk through between value values data file files sensor "
    "building room number what's give me run loop each per all any both not but its his her their"
).split())


def _chunks(text: str, target: int = 1600) -> list[str]:
    """Split corpus into retrievable chunks: blank-line paragraphs, with big ones (e.g. a TTL file
    with no blank lines) sub-split into ~target-sized line windows (small overlap). Works for prose,
    markdown, tables and Turtle alike."""
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        if not para.strip():
            continue
        if len(para) <= target * 1.7:
            out.append(para)
            continue
        lines = para.split("\n")
        i = 0
        while i < len(lines):
            j, size = i, 0
            while j < len(lines) and size < target:
                size += len(lines[j]) + 1
                j += 1
            out.append("\n".join(lines[i:j]))
            i = max(j - 2, i + 1)  # small overlap so a fact split across the boundary survives
    return out


def retrieve(corpus_text: str, question: str, max_chars: int = 14000, ignore=()) -> str:
    """Return the slice of `corpus_text` relevant to `question`: rank chunks by overlap with the
    question's strong keys (quoted phrases / entity-like ids, weighted ×10) and content words
    (weighted ×1), take the top chunks up to `max_chars`, restored to document order. Raw text, no
    parsing — works for natural-language operator questions as well as tagged engineer ones.
    """
    ig = {x.lower() for x in ignore}
    quoted = [a or b for a, b in re.findall(r'"([^"]+)"|\'([^\']+)\'', question) if (a or b)]
    ids = [t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_:\-]{2,}", question) if re.search(r"[0-9_:]", t)]
    strong = [k.lower() for k in (quoted + ids) if len(k) >= 3 and k.lower() not in ig]
    words = {w for w in re.findall(r"[a-z]{4,}", question.lower()) if w not in _STOP and w not in ig}
    chunks = _chunks(corpus_text)

    def score(c: str) -> int:
        cl = c.lower()
        return 10 * sum(1 for k in strong if k in cl) + sum(1 for w in words if w in cl)

    ranked = sorted(range(len(chunks)), key=lambda i: (score(chunks[i]), -i), reverse=True)
    picked: list[int] = []
    total = 0
    for i in ranked:
        if score(chunks[i]) <= 0:
            break
        c = chunks[i]
        if picked and total + len(c) > max_chars:
            break
        picked.append(i)
        total += len(c)
    if not picked:
        return corpus_text[:max_chars]
    return ("\n\n".join(chunks[i] for i in sorted(picked)))[:max_chars]
