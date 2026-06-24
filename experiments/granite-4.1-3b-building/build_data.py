#!/usr/bin/env python3
"""build_data.py — author realistic engineer/operator Q&A, then build SFT demos + the frozen exam.

Runnable form of `skills/prepare-trainset.md`: the teacher (a frontier model) authors the questions
a building **engineer or operator** actually asks about each building — no fixed taxonomy — with a
grounded answer and its `anchors` (the must-match facts). Training buildings become open-book SFT
demos (answered over a retrieved slice, via `lib/datakit`); the holdout building becomes the frozen
realistic exam `packs/building/eval_open.jsonl`. The judge (`eval_judge.py`) grades by anchors.

    set -a; source <your>/.env; set +a                                 # ANTHROPIC_API_KEY
    python experiments/granite-4.1-3b-building/build_data.py            # full
    python experiments/granite-4.1-3b-building/build_data.py --smoke    # 1 building, 5 Q&A, print

Edit the SPEC / prompts to change the recipe. NEVER edit packs/*/scorer.py or prepare.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "lib"))
sys.path.insert(0, str(REPO / "packs" / "building"))
import datakit  # noqa: E402
import llm  # noqa: E402
import prepare  # noqa: E402  (building data locator; default_holdout)
from corpus import building_corpus, retrieve  # noqa: E402

TEACHER = "anthropic:claude-opus-4-8"
HOLDOUT = os.environ.get("NEKAISE_HOLDOUT") or prepare.default_holdout()
N = 35
QA_DIR = EXP_DIR / "data" / "realistic_qa"   # git-ignored (experiments/**/data/)

SYSTEM_PROMPT = (  # the student persona; must match train.py
    "You are a knowledgeable building engineer assistant. Answer the question accurately and "
    "specifically, grounding your answer in the building's equipment, sensors, topology, and "
    "semantic model (Brick / ASHRAE 223P). Name the relevant entities and explain your reasoning.")

SENIOR_SYS = ("You are a senior building services engineer (HVAC, BMS/controls, commissioning). "
    "Below is ALL available data for one building — study it carefully.\n\n"
    "--- BUILDING DATA ---\n{corpus}\n--- END BUILDING DATA ---")

TASK = """Author {n} questions that a real building ENGINEER or OPERATOR would actually ask about THIS building — in their own words and shorthand. Do NOT use a fixed taxonomy; just span what they really need: look up a spec/value, explain the structure (what connects to what), explain control logic / a sequence, DIAGNOSE (what would you check, and why, when something is wrong), and fetch/inspect time-series. Mix the voice: some engineer (precise, uses tags), some operator (casual). Vary phrasing.

For each, give the correct grounded `answer` and its `anchors` — the specific facts the answer MUST contain to be judged right: numeric values, vendor tags, file paths, component names, time windows, or the items of a checklist. Every anchor MUST come from the data above — invent nothing; keep anchors minimal-but-complete.

Return ONLY a JSON array, no prose, no markdown fences:
[{{"persona":"engineer|operator","intent":"<one free-form word>","question":"...","answer":"...","anchors":["...","..."],"source":"<file or entity>"}}, ...]"""


def ob(ctx: str, q: str) -> str:  # must match train.py eval prompt
    return ("Use only the building's data below to answer.\n\n"
            f"--- BUILDING DATA ---\n{ctx}\n--- END BUILDING DATA ---\n\nQuestion: {q}")


def parse_array(text: str) -> list[dict]:
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    s = t.find("[")
    if s == -1:
        return []
    body = t[s:]
    for cand in ((body[: body.rfind("]") + 1]) if "]" in body else "",
                 (body[: body.rfind("}") + 1] + "]") if "}" in body else ""):
        if not cand:
            continue
        try:
            arr = json.loads(cand)
        except json.JSONDecodeError:
            continue
        out = [x for x in arr if isinstance(x, dict) and x.get("question") and x.get("answer") and x.get("anchors")]
        if out:
            return out
    return []


def author(bdir: Path, n: int = N, max_tokens: int = 16000) -> list[dict]:
    corpus, _ = building_corpus(bdir, max_chars=200000)
    reply = llm.generate(TEACHER, system=SENIOR_SYS.format(corpus=corpus),
                         user=TASK.format(n=n), max_tokens=max_tokens)
    items = parse_array(reply)
    for i, it in enumerate(items):
        it["id"] = f"{bdir.name[:3]}-{i:02d}"
    return items


def sft_row(ctx: str, it: dict) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ob(ctx, it["question"])},
        {"role": "assistant", "content": it["answer"]}]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="author 5 Q&A from one training building and print")
    args = ap.parse_args()

    train_dirs = [d for d in prepare.building_dirs() if d.name != HOLDOUT]
    if args.smoke:
        items = author(train_dirs[0], n=5, max_tokens=4000)
        print(f"=== {len(items)} realistic Q&A for '{train_dirs[0].name}' (engineer/operator) ===\n")
        for it in items:
            print(f"[{it['persona']}/{it.get('intent','')}] {it['question']}\n  A: {it['answer']}\n  anchors: {it['anchors']}\n")
        assert items, "teacher returned no parseable Q&A"
        return

    QA_DIR.mkdir(parents=True, exist_ok=True)
    rows, exam = [], []
    for d in prepare.building_dirs():
        b = d.name
        items = author(d)
        (QA_DIR / f"{b}.json").write_text(json.dumps(items, ensure_ascii=False, indent=2))
        print(f"  {b}: {len(items)} Q&A", file=sys.stderr)
        if b == HOLDOUT:
            for it in items:
                exam.append({"id": it["id"], "persona": it.get("persona", ""), "intent": it.get("intent", ""),
                             "question": it["question"], "ground_truth": it["answer"],
                             "anchors": it["anchors"], "source": it.get("source", "")})
        else:
            corpus, _ = building_corpus(d, max_chars=300000)
            for it in items:
                rows.append(sft_row(retrieve(corpus, it["question"], ignore=[b]), it))

    spec = {"pack": "building", "teacher": TEACHER, "method": "realistic-operator-qa", "holdout": HOLDOUT}
    dd = datakit.write(EXP_DIR, spec, rows, stats={"examples": len(rows), "teacher": TEACHER})
    exam_path = REPO / "packs" / "building" / "eval_open.jsonl"
    exam_path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in exam) + "\n")
    print(f"SFT demos: {len(rows)} -> {dd}  (LATEST={datakit.dataset_id(spec)})")
    print(f"frozen exam: {len(exam)} questions -> {exam_path}  (git-ignored)")
    print("train: python experiments/.../train.py   |   judge-eval: python experiments/.../eval_judge.py")


if __name__ == "__main__":
    main()
