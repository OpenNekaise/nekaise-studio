#!/usr/bin/env python3
"""build_cpt_data.py — build the CONTINUED-PRETRAINING (next-token) dataset from the HVAC corpus.

Reads nekaise_data/hvac_corpus (manifest + text/), CLEANS the extracted text (next-token training
imitates the text literally, so PDF/Wikipedia noise must go), drops thin/empty docs, holds out ~5%
of docs for perplexity, and writes:

    experiments/granite-4.1-3b-building/data/cpt/
        train.jsonl     {"id","topic","chars","text"}   -> run_cpt() trains on this (raw text)
        heldout.jsonl   same schema                       -> perplexity eval (ceiling metric)
        provenance.json counts + settings

Run (no GPU/network needed):
    python experiments/granite-4.1-3b-building/build_cpt_data.py
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO = EXP_DIR.parents[1]
CORPUS = REPO / "nekaise_data" / "hvac_corpus"
OUT = EXP_DIR / "data" / "cpt"

MIN_DOC_CHARS = 1500      # drop a doc if it cleans down below this (thin stubs / failed extractions)
HELDOUT_EVERY = 20        # ~5% of docs held out for perplexity (whole docs, no chunk leakage)

# tail headings: everything from here to end of doc is references / appendix noise -> truncate
_TAIL = re.compile(
    r"^(references?|bibliography|external links|further reading|see also|notes|"
    r"citations?|sources|acknowledge?ments?|works cited)\s*:?\s*$", re.I)


def strip_header(text: str) -> str:
    """Drop the build_corpus.py front-matter (# title / source / license / topic / ---)."""
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

    # drop running headers/footers: lines whose page-number-stripped form repeats across pages
    from collections import Counter

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


def main() -> None:
    manifest = [json.loads(l) for l in (CORPUS / "manifest.jsonl").read_text().splitlines() if l.strip()]
    ok = [r for r in manifest if r.get("status") == "ok" and r.get("text_path")]
    ok.sort(key=lambda r: r["id"])

    kept, dropped = [], []
    for r in ok:
        body = clean_body((CORPUS / r["text_path"]).read_text())
        if len(body) < MIN_DOC_CHARS:
            dropped.append((r["id"], len(body)))
            continue
        kept.append({"id": r["id"], "topic": r["topic"], "chars": len(body), "text": body})

    train = [d for i, d in enumerate(kept) if i % HELDOUT_EVERY != 0]
    heldout = [d for i, d in enumerate(kept) if i % HELDOUT_EVERY == 0]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in train))
    (OUT / "heldout.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in heldout))

    by_topic: dict = {}
    for d in train:
        by_topic[d["topic"]] = by_topic.get(d["topic"], 0) + 1
    tr_chars = sum(d["chars"] for d in train)
    prov = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "nekaise_data/hvac_corpus",
        "n_train": len(train), "n_heldout": len(heldout), "n_dropped_thin": len(dropped),
        "train_chars": tr_chars, "train_tokens_est": tr_chars // 4,
        "min_doc_chars": MIN_DOC_CHARS, "heldout_every": HELDOUT_EVERY,
        "by_topic_train": by_topic, "dropped": dropped,
    }
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False))

    print(f"[cpt-data] train {len(train)} docs / {tr_chars/1e6:.2f}M chars (~{tr_chars//4/1e6:.2f}M tok) "
          f"| heldout {len(heldout)} | dropped thin {len(dropped)}")
    print(f"[cpt-data] by topic: {by_topic}")
    print(f"[cpt-data] dropped: {[d[0] for d in dropped]}")
    print(f"[cpt-data] wrote -> {OUT}")


if __name__ == "__main__":
    main()
