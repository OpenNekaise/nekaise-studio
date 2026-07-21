#!/usr/bin/env python3
"""build_probes.py — mint the corpus-absorption probe set ONCE. Maintainer tool, not the loop's.

The corpus_probes pack is the studio-owned loop metric for pure-CPT phases: numeric cloze
continuations extracted DETERMINISTICALLY (no LLM, no randomness) from the cleaned HVAC
corpus. A probe is a sentence prefix that ends right before a salient number+unit span; the
model must continue with that value. Accuracy = "did this fact enter the weights" — dense
and sensitive where nekaise-bench (hardened, milestone-only) is deliberately not.

Two kinds, split by the shared corpusprep.is_heldout_doc rule:
    absorption — from CPT TRAIN docs (did training inject what it read?)  -> the loop metric
    transfer   — from CPT HELDOUT docs (does it generalize to unseen docs?) -> diagnostic

Probe ids are content-derived; dev/frozen split is id-hashed (md5 % 5 == 0 -> frozen).
Output (COMMITTED — the referee is fixed once minted; rebuild only as a versioned reform):
    packs/corpus_probes/probes.jsonl
    packs/corpus_probes/provenance.json   (counts + corpus fingerprint)

    python packs/corpus_probes/build_probes.py            # refuses to overwrite
    python packs/corpus_probes/build_probes.py --force
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parent
REPO = PACK_DIR.parents[1]
sys.path.insert(0, str(REPO / "lib"))
import corpusprep  # noqa: E402

CORPUS = REPO / "nekaise_data" / "hvac_corpus"
MAX_PER_DOC = 3
MIN_PROMPT_CHARS = 60
MAX_ANSWER_CHARS = 20

# number + required unit-ish anchor: bare numbers are ambiguous, unit-bound values are facts
_UNIT = (r"%|°\s?[CF]|kWh?|MWh|GWh|Wh|W\b|Btu|MBtu|kBtu|therms?|tons?\b|Pa\b|kPa|psi[ag]?|bar\b|"
         r"cfm|m3/s|m³/s|L/s|gpm|K\b|ppm|Hz|volts?|V\b|amps?|A\b|mm\b|cm\b|m2|m²|ft2|ft²|"
         r"years?|hours?|minutes?|seconds?|°")
_SPAN = re.compile(r"\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s?(" + _UNIT + r")",
                   re.UNICODE)
_SENT = re.compile(r"(?<=[.!?])\s+")
_NUMS = re.compile(r"\d+(?:\.\d+)?")


def probes_from_doc(doc: dict) -> list[dict]:
    out = []
    for para in doc["text"].split("\n\n"):
        para = " ".join(para.split())
        if len(para) < 80 or para.startswith("#"):
            continue
        for sent in _SENT.split(para):
            if not (80 <= len(sent) <= 400):
                continue
            matches = list(_SPAN.finditer(sent))
            if not matches:
                continue
            m = matches[-1]                              # last span -> longest natural prefix
            prompt, answer = sent[:m.start()].rstrip(), m.group(0).strip()
            if len(prompt) < MIN_PROMPT_CHARS or len(answer) > MAX_ANSWER_CHARS:
                continue
            if m.group(1) in prompt:                     # value already visible -> copyable
                continue
            compact = prompt.replace(" ", "")
            if sum(c.isalpha() for c in prompt) < 0.65 * len(compact):
                continue
            if sum(c.isascii() for c in compact) < 0.85 * len(compact):   # non-English doc
                continue
            if len(_NUMS.findall(prompt)) > 4:           # table soup, not prose
                continue
            pid = hashlib.md5(f"{doc['id']}|{prompt}|{answer}".encode()).hexdigest()[:12]
            out.append({
                "id": f"cp-{pid}",
                "kind": "transfer" if corpusprep.is_heldout_doc(doc["id"]) else "absorption",
                "doc_id": doc["id"], "topic": doc["topic"],
                "prompt": prompt, "answer": answer, "value": m.group(1).replace(",", ""),
            })
            if len(out) >= MAX_PER_DOC:
                return out
    return out


def main() -> None:
    out_path = PACK_DIR / "probes.jsonl"
    if out_path.exists() and "--force" not in sys.argv:
        sys.exit(f"{out_path} exists — the referee is fixed once minted (--force to re-mint, "
                 f"which is a versioned reform: every recorded probe metric re-baselines)")
    docs = corpusprep.load_clean_docs(CORPUS)
    probes, seen_prompt = [], set()
    for doc in docs:
        for p in probes_from_doc(doc):
            key = p["prompt"][-60:].lower()
            if key in seen_prompt:                       # boilerplate repeated across docs
                continue
            seen_prompt.add(key)
            probes.append(p)
    n_frozen = sum(1 for p in probes
                   if int(hashlib.md5(p["id"].encode()).hexdigest(), 16) % 5 == 0)
    out_path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in probes))
    kinds = {k: sum(1 for p in probes if p["kind"] == k) for k in ("absorption", "transfer")}
    prov = {"built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_docs": len(docs), "n_probes": len(probes), "by_kind": kinds,
            "n_frozen": n_frozen,
            "fingerprint": hashlib.sha256(out_path.read_bytes()).hexdigest()[:16]}
    (PACK_DIR / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"[probes] {len(probes)} probes from {len(docs)} docs  {kinds}  "
          f"frozen={n_frozen}  fingerprint={prov['fingerprint']}")


if __name__ == "__main__":
    main()
