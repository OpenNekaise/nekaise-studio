"""corpusprep — shared corpus loading + cleaning for CPT-style recipes. FIXED plumbing.

Extracted from experiments/granite-4.1-3b-building/build_cpt_data.py so recipes stop
carrying private copies of the cleaner. Also absorbs the corpus's manifest reshard:
`manifest.jsonl` (single file, legacy) and `manifest/*.jsonl` (sharded) both load.

    load_manifest(corpus_dir)   -> [{"id","topic","status","text_path",...}, ...]
    clean_body(text)            -> cleaned plain text (headers/footers/refs/noise removed)
    load_clean_docs(corpus_dir) -> [{"id","topic","chars","text"}], sorted by id, thin docs dropped
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

MIN_DOC_CHARS = 1500      # drop a doc if it cleans down below this (thin stubs / failed extractions)

_TAIL = re.compile(
    r"^(references?|bibliography|external links|further reading|see also|notes|"
    r"citations?|sources|acknowledge?ments?|works cited)\s*:?\s*$", re.I)


def load_manifest(corpus_dir: Path) -> list[dict]:
    """Read the corpus manifest — single-file `manifest.jsonl` or sharded `manifest/*.jsonl`."""
    corpus_dir = Path(corpus_dir)
    single = corpus_dir / "manifest.jsonl"
    if single.exists():
        paths = [single]
    else:
        shard_dir = corpus_dir / "manifest"
        paths = sorted(shard_dir.glob("*.jsonl")) if shard_dir.is_dir() else []
    if not paths:
        raise FileNotFoundError(f"no manifest.jsonl or manifest/*.jsonl under {corpus_dir}")
    rows: list[dict] = []
    for p in paths:
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows


def strip_header(text: str) -> str:
    """Drop the corpus front-matter (# title / source / license / topic / ---)."""
    i = text.find("\n---\n\n")
    return text[i + 6:] if i != -1 else text


def _is_noise(s: str) -> bool:
    low = s.lower()
    if re.fullmatch(r"\d{1,4}", s):                       # bare page number
        return True
    if re.fullmatch(r"[ivxlcdm]{1,7}", low):              # roman numeral page
        return True
    if low.startswith(("doi:", "arxiv:", "issn", "isbn", "http://", "https://", "www.")):
        return True
    if re.match(r"^\[\d+\]", s) or re.match(r"^\(\d+\)\s", s):   # [12] / (12) reference entry
        return True
    if any(p in low for p in ("retrieved from", "all rights reserved", "downloaded from",
                              "creative commons", "this page was last edited")):
        return True
    if low.startswith("this article") and any(
            w in low for w in ("issue", "help improve", "verif", "additional citations", "sources")):
        return True
    letters = sum(c.isalpha() for c in s)                 # tables / equations / citation soup
    if len(s) >= 8 and letters / len(s) < 0.5:
        return True
    return False


def clean_body(text: str) -> str:
    text = strip_header(text)
    text = re.sub(r"-\n(?=[a-z])", "", text)              # de-hyphenate line-wrapped words
    raw = [ln.rstrip() for ln in text.splitlines()]
    n = len(raw)

    # truncate the references/appendix tail, but ONLY a tail heading in the back portion of the
    # doc -- a front-matter "Acknowledgments"/"Notes" must not nuke the whole body.
    for idx in range(n):
        if idx > 0.55 * n and _TAIL.match(raw[idx].strip()):
            raw = raw[:idx]
            break

    def _nf(s):
        return re.sub(r"\s*(page\s+)?\d+(\s+of\s+\d+)?\s*$", "", s, flags=re.I).strip()

    freq = Counter(_nf(ln.strip()) for ln in raw if len(_nf(ln.strip())) >= 15)
    repeated = {k for k, c in freq.items() if c >= 4}

    out, blanks = [], 0
    for ln in raw:
        s = ln.strip()
        if not s:
            blanks += 1
            if blanks <= 1 and out:
                out.append("")
            continue
        blanks = 0
        if _nf(s) in repeated or _is_noise(s):
            continue
        out.append(s)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


CORE_TOPICS = ("building_energy", "equipment_systems", "controls_bas",
               "commissioning_fdd", "standards_protocols")


def select_docs(manifest: list[dict], n_docs: int = 1200,
                topics: tuple = CORE_TOPICS) -> list[dict]:
    """Deterministic curated subset of a large corpus for CPT + probes.

    The full corpus is ~75k docs of mixed language/genre; CPT and the probe referee both
    need the same curated slice. Filter: status ok, core topic, English-dominant and
    domain-dense per the manifest's quality window; rank by domain-term density; take the
    top n_docs; sort by id so downstream iteration order is stable.
    """
    cands = []
    for r in manifest:
        if r.get("status") != "ok" or not r.get("text_path") or r.get("topic") not in topics:
            continue
        q = (r.get("quality") or {}).get("w20") or {}
        words, en, domain = q.get("words", 0), q.get("en", 0), q.get("domain", 0)
        if not words or en / words < 0.25 or domain < 20:
            continue
        cands.append((domain / words, r))
    cands.sort(key=lambda t: (-t[0], t[1]["id"]))
    picked = [r for _, r in cands[:n_docs]]
    picked.sort(key=lambda r: r["id"])
    return picked


def is_heldout_doc(doc_id: str) -> bool:
    """Deterministic ~5% doc-level holdout: md5(id) % 20 == 0.

    Id-hashed (not index-based) so the assignment is stable as the corpus grows. This is
    THE doc split shared by CPT training data (train on ~95%) and the corpus_probes pack
    (absorption probes from train docs, transfer probes from heldout docs). Do not change
    the hash or modulus — probe kinds and CPT sets must stay aligned.
    """
    import hashlib
    return int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % 20 == 0


def load_clean_docs(corpus_dir: Path, min_doc_chars: int = MIN_DOC_CHARS,
                    n_docs: int = 1200) -> list[dict]:
    """Manifest -> curated subset (select_docs) -> cleaned doc dicts, sorted by id."""
    corpus_dir = Path(corpus_dir)
    docs = []
    for r in select_docs(load_manifest(corpus_dir), n_docs=n_docs):
        p = corpus_dir / r["text_path"]
        if not p.exists():
            continue
        body = clean_body(p.read_text())
        if len(body) < min_doc_chars:
            continue
        docs.append({"id": r["id"], "topic": r.get("topic", ""), "chars": len(body), "text": body})
    return docs
