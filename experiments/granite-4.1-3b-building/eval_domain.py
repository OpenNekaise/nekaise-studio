#!/usr/bin/env python3
"""eval_domain.py — CLOSED-BOOK building / HVAC / building-energy knowledge quiz.

A CEILING metric: measures domain knowledge held in the model's WEIGHTS, with NO retrieval and
NO building-specific context. This is what continued-pretraining (CPT) actually changes, and it is
exactly what building_judge (building-specific, open-book, retrieval-confounded) cannot see.

    python eval_domain.py --model base                 # the base model
    python eval_domain.py --model outputs/cpt_full      # a CPT checkpoint
    python eval_domain.py --model outputs/grpo_cptfull  # end of the chain

Prints a DOMAIN_ACC line (overall closed-book multiple-choice accuracy) + per-topic breakdown.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parent
BASE_MODEL = "unsloth/granite-4.1-3b"
QUIZ = EXP / "domain_quiz.jsonl"
LETTERS = "ABCD"
SYS = ("You are an expert in building systems, HVAC, and building energy. "
       "Answer the multiple-choice question with only the single letter (A, B, C, or D) "
       "of the best choice.")


def load(model_arg):
    from unsloth import FastLanguageModel
    src = BASE_MODEL if model_arg in ("base", "") else str(EXP / model_arg)
    model, tok = FastLanguageModel.from_pretrained(
        model_name=src, max_seq_length=2048, load_in_4bit=True, dtype=None)
    FastLanguageModel.for_inference(model)
    return model, tok


def ask(model, tok, q) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(q["choices"]))
    user = f"{q['q']}\n\n{opts}\n\nAnswer:"
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=5, do_sample=False)
    dec = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    m = re.search(r"[ABCD]", dec.upper())
    return m.group(0) if m else "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base")
    ap.add_argument("--quiz", default=str(QUIZ), help="path to a quiz .jsonl")
    args = ap.parse_args()
    qs = [json.loads(l) for l in Path(args.quiz).read_text().splitlines() if l.strip()]
    model, tok = load(args.model)
    correct, by, hit = 0, Counter(), Counter()
    for q in qs:
        by[q["topic"]] += 1
        ok = ask(model, tok, q) == q["answer"]
        correct += ok
        hit[q["topic"]] += int(ok)
    acc = correct / len(qs)
    print("[domain] by topic:", {k: f"{hit[k]}/{by[k]}" for k in sorted(by)})
    print(f"DOMAIN_ACC model={args.model} quiz={Path(args.quiz).name} acc={acc:.4f} n={len(qs)}")


if __name__ == "__main__":
    main()
