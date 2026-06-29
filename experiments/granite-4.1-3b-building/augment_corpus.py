#!/usr/bin/env python3
"""augment_corpus.py — knowledge-augmented CPT data.

Raw next-token CPT tends to memorize a fact's surface FORM; to store a fact GENERALIZABLY (so it is
recallable in any phrasing) the fact should appear in DIVERSE forms during training (Allen-Zhu,
"Physics of Language Models" 3.x). For each training doc the teacher rewrites the key facts several
ways and writes a few QA pairs; we concatenate original + paraphrases + QA into an augmented corpus
that `run_cpt` can train on.

    set -a; source <KebAgent>/.env; set +a
    python augment_corpus.py --n 200            # augment all training docs
    NEKAISE_CPT_TRAIN=.../data/cpt/train_aug.jsonl NEKAISE_METHOD=cpt ... python train.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "lib"))
import llm  # noqa: E402

TEACHER = "anthropic:claude-opus-4-8"
SRC = EXP / "data" / "cpt" / "train.jsonl"
OUT = EXP / "data" / "cpt" / "train_aug.jsonl"

AUG_SYS = (
    "You turn a source passage into DIVERSE training text that teaches its SPECIFIC facts so a model "
    "can recall them in any phrasing. Output ONLY JSON: "
    '{"paraphrases": ["restate the key facts in different wording AND structure", ... 5 total], '
    '"qa": [{"q": "...", "a": "..."}, ... 4 total]}. '
    "Cover the passage's concrete facts (numbers, values, named methods, definitions). Vary sentence "
    "structure/order across paraphrases; keep each self-contained and factually faithful to the "
    "passage. The QA answers must be stated explicitly, not hedged.")


def doc_body(text: str) -> str:
    return (text.split("\n---\n\n", 1)[-1] if "\n---\n\n" in text else text).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="max docs to augment")
    args = ap.parse_args()
    docs = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    docs.sort(key=lambda d: d["id"])
    step = max(1, len(docs) // max(1, args.n))
    picked = docs[::step][:args.n]

    rows, ok = [], 0
    for d in picked:
        full = doc_body(d["text"])         # keep the FULL doc so the corpus isn't shrunk
        pw = full[:2500]                   # the slice the teacher reads to author paraphrases/QA
        try:
            rep = llm.generate(TEACHER, system=AUG_SYS, user="PASSAGE:\n" + pw, max_tokens=1600)
            v = json.loads(re.search(r"\{.*\}", rep, re.S).group(0))
            parts = [full]                 # FULL original + diverse forms of its key facts
            parts += [p for p in v.get("paraphrases", []) if isinstance(p, str) and p.strip()]
            parts += [f"Q: {x['q']}\nA: {x['a']}" for x in v.get("qa", [])
                      if isinstance(x, dict) and x.get("q") and x.get("a")]
            rows.append({"id": d["id"] + "-aug", "topic": d["topic"], "text": "\n\n".join(parts)})
            ok += 1
        except Exception as e:
            print(f"  skip {d['id']}: {e}", file=sys.stderr)
            rows.append({"id": d["id"], "topic": d["topic"], "text": full})  # fallback: full original

    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tot = sum(len(r["text"]) for r in rows)
    print(f"augmented {ok}/{len(picked)} docs -> {OUT} (~{tot // 4 / 1e6:.2f}M tokens, "
          f"{tot / max(1, sum(len(passage(d['text'])) for d in picked)):.1f}x raw)")


if __name__ == "__main__":
    main()
