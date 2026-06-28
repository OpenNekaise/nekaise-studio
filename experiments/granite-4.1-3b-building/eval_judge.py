#!/usr/bin/env python3
"""eval_judge.py — the realistic metric: answer the frozen exam, then judge by anchors.

Runnable form of `skills/judge.md` (eval mode). Every answerer gets the SAME harness — same system
prompt + retrieved slice + question. The judge (a frontier model, blind) scores each answer = the
fraction of its `anchors` present (strict on the fact, lenient on phrasing). Reports `building_judge`.

    set -a; source <your>/.env; set +a                                  # ANTHROPIC_API_KEY
    python .../eval_judge.py answer teacher          # teacher (API)  -> answers/teacher.json
    python .../eval_judge.py answer base|student     # local model (GPU) -> answers/<mode>.json
    python .../eval_judge.py judge base|student|teacher                  # -> building_judge
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "lib"))
sys.path.insert(0, str(REPO / "packs" / "building"))
import prepare  # noqa: E402
from corpus import building_corpus, retrieve  # noqa: E402

TEACHER = "anthropic:claude-opus-4-8"
HOLDOUT = os.environ.get("NEKAISE_HOLDOUT") or prepare.default_holdout()
ANS_DIR = EXP_DIR / "data" / "answers"   # git-ignored (experiments/**/data/)
EXAM = [json.loads(l) for l in (REPO / "packs" / "building" / "eval_open.jsonl").read_text().splitlines() if l.strip()]

SYSTEM_PROMPT = (
    "You are a knowledgeable building engineer assistant. Answer the question accurately and "
    "specifically, grounding your answer in the building's equipment, sensors, topology, and "
    "semantic model (Brick / ASHRAE 223P). Name the relevant entities and explain your reasoning.")

JUDGE_SYS = ("You are an expert grader for building QA. You get a question, the gold answer, the "
    "REQUIRED ANCHORS (facts the answer must contain), and a candidate answer. For each anchor decide "
    "if the candidate states it — match the FACT strictly (numbers, vendor tags incl. suffixes, file "
    "paths incl. prefixes/extension, component names, time-window start AND end) but allow different "
    "phrasing/synonyms/order. Output ONLY JSON: "
    '{"matched": ["<anchor>", ...], "missed": ["<anchor>", ...], "contradiction": true|false}. '
    "Set contradiction=true if the candidate confidently asserts a WRONG value/tag/path where the gold is specific.")


def ob(ctx: str, q: str) -> str:
    return ("Use only the building's data below to answer.\n\n"
            f"--- BUILDING DATA ---\n{ctx}\n--- END BUILDING DATA ---\n\nQuestion: {q}")


def _contexts() -> dict:
    corpus, _ = building_corpus(prepare.DATA / HOLDOUT, max_chars=300000)
    return {q["id"]: retrieve(corpus, q["question"], ignore=[HOLDOUT]) for q in EXAM}


def answer(mode: str, model_path: str | None = None) -> None:
    ANS_DIR.mkdir(parents=True, exist_ok=True)
    ctx = _contexts()
    out = []
    if mode == "teacher":
        import llm
        for q in EXAM:
            cand = llm.generate(TEACHER, system=SYSTEM_PROMPT, user=ob(ctx[q["id"]], q["question"]), max_tokens=400)
            out.append({"id": q["id"], "intent": q.get("intent", ""), "candidate": cand})
    else:
        from unsloth import FastLanguageModel
        # mode is just the output label; model_path (relative to EXP_DIR) picks the checkpoint.
        if model_path:
            src = str(EXP_DIR / model_path)
        elif mode == "base":
            src = "unsloth/granite-4.1-3b"
        else:
            src = str(EXP_DIR / "outputs" / ("sft" if mode == "student" else mode))
        model, tok = FastLanguageModel.from_pretrained(model_name=src, max_seq_length=16384, load_in_4bit=True, dtype=None)
        FastLanguageModel.for_inference(model)
        for q in EXAM:
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ob(ctx[q["id"]], q["question"])}],
                tokenize=False, add_generation_prompt=True)
            inp = tok(prompt, return_tensors="pt").to(model.device)
            o = model.generate(**inp, max_new_tokens=400, do_sample=False)
            out.append({"id": q["id"], "intent": q.get("intent", ""),
                        "candidate": tok.decode(o[0][inp.input_ids.shape[1]:], skip_special_tokens=True)})
    (ANS_DIR / f"{mode}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"{mode} answered {len(out)} -> {ANS_DIR / (mode + '.json')}")


def judge(label: str) -> None:
    import llm
    exam = {q["id"]: q for q in EXAM}
    ans = json.loads((ANS_DIR / f"{label}.json").read_text())
    rows, by_intent = [], defaultdict(list)
    for a in ans:
        q = exam[a["id"]]
        anchors = q["anchors"]
        prompt = (f"Question: {q['question']}\nGold answer: {q['ground_truth']}\n"
                  f"Required anchors: {json.dumps(anchors, ensure_ascii=False)}\n"
                  f"Candidate answer: {a['candidate']}")
        try:
            rep = llm.generate(TEACHER, system=JUDGE_SYS, user=prompt, max_tokens=500)
            v = json.loads(re.search(r"\{.*\}", rep, re.S).group(0))
            score = len([x for x in v.get("matched", []) if x in anchors]) / max(1, len(anchors))
            if v.get("contradiction"):
                score = min(score, 0.5)
        except Exception as e:
            score = 0.0
            print(f"  judge err {a['id']}: {e}", file=sys.stderr)
        rows.append({"id": a["id"], "intent": a.get("intent", ""), "score": round(score, 3)})
        by_intent[a.get("intent", "")].append(score)
    bj = sum(r["score"] for r in rows) / len(rows)
    buckets = {"1.0": sum(1 for r in rows if r["score"] >= 0.999),
               "partial": sum(1 for r in rows if 0 < r["score"] < 0.999),
               "0": sum(1 for r in rows if r["score"] == 0)}
    (ANS_DIR / f"judge_{label}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"JUDGE[{label}] building_judge={bj:.4f} n={len(rows)} buckets={buckets}")
    print("  per-intent:", {k: round(sum(v) / len(v), 2) for k, v in sorted(by_intent.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))})


if __name__ == "__main__":
    if sys.argv[1] == "answer":
        answer(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif sys.argv[1] == "judge":
        judge(sys.argv[2])
