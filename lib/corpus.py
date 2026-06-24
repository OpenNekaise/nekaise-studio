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

def _question_keys(question: str) -> list[str]:
    """Distinctive search keys from a question: quoted phrases + entity-like identifier tokens."""
    quoted = [a or b for a, b in re.findall(r'"([^"]+)"|\'([^\']+)\'', question) if (a or b)]
    phrases = [q for q in quoted if len(q) >= 6] or quoted          # drop short quotes (e.g. building name)
    ids = [t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_:\-]{2,}", question)
           if re.search(r"[0-9_:]", t)]                              # entity-like tokens (have a digit/_/:)
    return phrases + ids


def retrieve(corpus_text: str, question: str, max_chars: int = 14000, ignore=()) -> str:
    """Return the slice of `corpus_text` relevant to `question` — the full statement BLOCK around
    each line matching the question's quoted phrases / entity ids. Raw text, no parsing: a block is
    a contiguous run bounded by a blank line or a statement terminator (` .`), so an entity's whole
    set of triples (type, comment, every s223:cnx) is captured. Short by design so training/eval
    stay cheap and realistic (data >> small-model window).
    """
    ig = {x.lower() for x in ignore}
    keys = [k.lower() for k in _question_keys(question) if len(k) >= 4 and k.lower() not in ig]
    if not keys:
        return corpus_text[:max_chars]
    lines = corpus_text.split("\n")
    low = [l.lower() for l in lines]
    hits = [i for i, l in enumerate(low) if any(k in l for k in keys)]
    if not hits:
        return corpus_text[:max_chars]
    keep: set[int] = set()
    for i in hits:
        a = i
        while a > 0 and lines[a - 1].strip() and not lines[a - 1].rstrip().endswith("."):
            a -= 1
        b = i
        while b < len(lines) - 1 and lines[b].strip() and not lines[b].rstrip().endswith("."):
            b += 1
        keep.update(range(a, b + 1))
    return ("\n".join(lines[j] for j in sorted(keep)))[:max_chars]
