#!/usr/bin/env python3
"""build_data.py — WHAT: the corpus-QA distillation set for the CEILING phase (MUTABLE recipe).

A LOCAL teacher (default ollama:qwen3.6:27b — free, offline, no API key) reads chunks of the
CLEANED corpus — the same text CPT trains on (../granite-4.1-3b-building/data/cpt/train.jsonl,
built by build_cpt_data.py) — and authors short factual QA pairs in the closed-book style of
nekaise-bench's open track. SFT on these distills the teacher's reading of the corpus into
the student and restores instruction-following after CPT, using no data beyond the corpus.

Grounding gate: a pair is kept only if its answer (normalized, or numerically) appears in the
source chunk — cheap insurance against teacher hallucination. Provenance (doc id, chunk) is
kept on every row.

    python experiments/ceiling-sub4b/build_data.py        # cached via datakit -> data/LATEST
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "lib"))
import datakit  # noqa: E402
import llm  # noqa: E402

CPT_TRAIN = REPO / "experiments" / "granite-4.1-3b-building" / "data" / "cpt" / "train.jsonl"

# ==================================== KNOBS ====================================
TEACHER       = os.environ.get("NEKAISE_TEACHER", "ollama:qwen3.6:27b")
N_CHUNKS      = int(os.environ.get("NEKAISE_N_CHUNKS", 160))   # corpus chunks the teacher reads
QA_PER_CHUNK  = 4
CHUNK_CHARS   = 2400
WORKERS       = 4
SEED          = 3407
# ===============================================================================

# The student answers exactly like the bench's open track asks (format-aligned, content-free).
STUDENT_SYS = ("Answer the question about building energy concisely. Give ONLY the short "
               "factual answer (a number, term, or name). No explanation.")

TEACHER_PROMPT = """You are a senior building-energy engineer writing exam questions.
From the text below, write {k} standalone factual question-answer pairs about building
energy. Rules:
- each question must stand alone, answerable by a knowledgeable engineer without seeing the
  text (never write "according to the text/study/table")
- ask about TECHNICAL building-energy facts: systems, equipment, controls, physics,
  protocols, standards, typical magnitudes. NEVER about authors, affiliations, publication
  dates, report titles, or any document metadata
- the answer is SHORT — a number (with unit), a term, or a name — and must appear in the text
- cover different facts; no two questions about the same fact
Output ONLY a JSON array: [{{"q": "...", "a": "..."}}, ...]

--- TEXT ---
{chunk}
--- END TEXT ---"""


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w.\-/% ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _numbers(s: str) -> set[float]:
    out = set()
    for x in re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", "")):
        try:
            out.add(float(x))
        except ValueError:
            pass
    return out


def grounded(answer: str, chunk: str) -> bool:
    """Keep only answers the chunk actually states (text or numeric match)."""
    a, c = _norm(answer), _norm(chunk)
    if a and not re.fullmatch(r"-?[\d,.]+\s*%?", a) and a in c:
        return True
    nums = _numbers(answer)
    return bool(nums) and nums <= _numbers(chunk)


def corpus_chunks() -> list[dict]:
    """Deterministic sample of ~N_CHUNKS chunks, round-robin over docs for topic coverage."""
    if not CPT_TRAIN.exists():
        sys.exit(f"missing {CPT_TRAIN} — run build_cpt_data.py first")
    docs = [json.loads(l) for l in CPT_TRAIN.read_text().splitlines() if l.strip()]
    random.Random(SEED).shuffle(docs)
    per_doc: list[list[dict]] = []
    for d in docs:
        paras, buf, size = [], [], 0
        for p in d["text"].split("\n\n"):
            if size + len(p) > CHUNK_CHARS and buf:
                paras.append({"doc": d["id"], "topic": d.get("topic", ""), "text": "\n\n".join(buf)})
                buf, size = [], 0
            buf.append(p)
            size += len(p) + 2
        if buf:
            paras.append({"doc": d["id"], "topic": d.get("topic", ""), "text": "\n\n".join(buf)})
        per_doc.append(paras)
    out, i = [], 0
    while len(out) < N_CHUNKS and any(per_doc):
        for chunks in per_doc:
            if i < len(chunks) and len(out) < N_CHUNKS:
                out.append(chunks[i])
        i += 1
        if i > 50:  # corpus smaller than requested sample
            break
    return out


def author(chunk: dict) -> list[dict]:
    user = TEACHER_PROMPT.format(k=QA_PER_CHUNK, chunk=chunk["text"])
    try:
        try:  # thinking teachers (qwen3.6) need think=False or content comes back empty
            rep = llm.generate(TEACHER, user=user, temperature=0.3, max_tokens=1200, think=False)
        except Exception:
            rep = llm.generate(TEACHER, user=user, temperature=0.3, max_tokens=1200)
        m = re.search(r"\[.*\]", rep, re.S)
        pairs = json.loads(m.group(0)) if m else []
    except Exception as e:
        print(f"  ! teacher failed on {chunk['doc']}: {e}", file=sys.stderr)
        return []
    rows = []
    for p in pairs:
        q, a = str(p.get("q", "")).strip(), str(p.get("a", "")).strip()
        if q and a and len(a) < 120 and grounded(a, chunk["text"]):
            rows.append({"messages": [{"role": "system", "content": STUDENT_SYS},
                                      {"role": "user", "content": q},
                                      {"role": "assistant", "content": a}],
                         "doc": chunk["doc"], "topic": chunk["topic"]})
    return rows


def build() -> list[dict]:
    chunks = corpus_chunks()
    print(f"[build] {len(chunks)} chunks -> teacher {TEACHER} ({QA_PER_CHUNK} QA each)")
    rows, seen = [], set()
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, part in enumerate(ex.map(author, chunks)):
            for r in part:
                key = _norm(r["messages"][1]["content"])
                if key not in seen:
                    seen.add(key)
                    rows.append(r)
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(chunks)}] kept {len(rows)} pairs", flush=True)
    return rows


def main() -> None:
    spec = {"kind": "sft", "task": "corpus_qa_distill", "teacher": TEACHER,
            "n_chunks": N_CHUNKS, "qa_per_chunk": QA_PER_CHUNK,
            "chunk_chars": CHUNK_CHARS, "seed": SEED, "source": "cpt/train.jsonl"}
    d = datakit.cached(EXP_DIR, spec, build,
                       stats=lambda rows: {"pairs": len(rows),
                                           "docs": len({r["doc"] for r in rows})})
    prov = datakit.provenance(d)
    print(f"[build] dataset {prov['dataset_id']}: {prov['n']} rows "
          f"({prov['stats']}) -> {d}")


if __name__ == "__main__":
    main()
