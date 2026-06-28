#!/usr/bin/env python3
"""gen_corpus_quiz.py — teacher-authored MCQ grounded in the CPT corpus.

Each question asks a SPECIFIC fact stated in a passage — a value / threshold / named method /
precise definition a knowledgeable general reader would NOT already know without having read that
source. Run base vs a CPT checkpoint CLOSED-BOOK via eval_domain.py --quiz.

  --source train    -> questions from the CPT *training* docs (tests memorization/absorption)
  --source heldout  -> questions from the *held-out* docs (tests generalization; the model never
                       trained on these, so a gain there is real understanding, not recall)

    set -a; source <KebAgent>/.env; set +a
    python gen_corpus_quiz.py --source heldout --windows 3 --n 40
    python eval_domain.py --quiz data/corpus_quiz_heldout.jsonl --model outputs/cpt_e15
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "lib"))
import llm  # noqa: E402

TEACHER = "anthropic:claude-opus-4-8"
GEN_SYS = (
    "You write ONE multiple-choice question that tests a SPECIFIC fact stated in the given passage "
    "— a number, value, threshold, named method/algorithm, or precise definition that a "
    "knowledgeable general reader would NOT already know without having read this exact source. "
    "The question must be answerable ONLY from this passage's specific content, NOT from general "
    "domain knowledge. Give 4 options, exactly one correct, with plausible distractors of similar "
    "type/magnitude. Output ONLY JSON: "
    '{"q": "...", "choices": ["A text","B text","C text","D text"], "answer": "A|B|C|D", '
    '"fact": "the specific fact tested"}')


def window(text: str, idx: int, n: int = 1800) -> str:
    body = text.split("\n---\n\n", 1)[-1] if "\n---\n\n" in text else text
    start = idx * n
    return body[start:start + n].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--source", default="train", choices=["train", "heldout"])
    ap.add_argument("--windows", type=int, default=1, help="passages to sample per doc")
    args = ap.parse_args()

    src = EXP / "data" / "cpt" / f"{args.source}.jsonl"
    out = EXP / "data" / ("corpus_quiz.jsonl" if args.source == "train"
                          else "corpus_quiz_heldout.jsonl")
    docs = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    docs.sort(key=lambda d: d["id"])
    step = max(1, len(docs) // max(1, args.n))
    picked = docs[::step][:args.n]

    out_rows = []
    for di, d in enumerate(picked):
        for w in range(args.windows):
            pw = window(d["text"], w)
            if len(pw) < 400:
                continue
            try:
                rep = llm.generate(TEACHER, system=GEN_SYS, user="PASSAGE:\n" + pw, max_tokens=600)
                v = json.loads(re.search(r"\{.*\}", rep, re.S).group(0))
                if isinstance(v.get("choices"), list) and len(v["choices"]) == 4 and v.get("answer") in "ABCD":
                    out_rows.append({"id": f"cq-{di:03d}-{w}", "topic": d["topic"], "q": v["q"],
                                     "choices": v["choices"], "answer": v["answer"], "src": d["id"]})
            except Exception as e:
                print(f"  skip {d['id']} w{w}: {e}", file=sys.stderr)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out_rows))
    by = Counter(x["topic"] for x in out_rows)
    print(f"wrote {len(out_rows)} MCQs from {len(picked)} {args.source} docs -> {out}")
    print("by topic:", dict(by))


if __name__ == "__main__":
    main()
