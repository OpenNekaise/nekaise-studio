#!/usr/bin/env python3
"""build_data.py — author realistic engineer/operator Q&A, then build GROUNDED, GATED SFT demos.

Runnable form of `skills/prepare-trainset.md`. Two-step, task-aligned distillation that fixes the
grounding-poison confound (answer authored over the FULL corpus while the student only sees a
retrieved slice):

  1. AUTHOR  — the teacher reads each training building's full corpus and writes the realistic
     questions an engineer/operator asks, each with grounded `anchors` (the must-match facts).
  2. RE-ANSWER ON SLICE — for every question, the teacher answers **using only the retrieved slice
     the student will see** (`lib/corpus.retrieve`), instructed to be complete and name every
     relevant entity/value. This makes the demo answer grounded in the student's actual input.
  3. GATE — keep a demo only if its slice-grounded answer matches **every anchor that is present in
     the slice** (deterministic, lenient-on-separators), with no contradiction. Demos whose answer
     would have to state a fact absent from the slice are dropped — never train ungrounded recall.

The holdout building becomes the frozen realistic exam `packs/building/eval_open.jsonl` (authored
ONCE, then frozen — guarded here so a rebuild can't perturb the metric; set NEKAISE_REGEN_EXAM=1 to
re-author it deliberately). The judge (`eval_judge.py`) grades the student by anchors.

    set -a; source <your>/.env; set +a                                 # ANTHROPIC_API_KEY
    python experiments/granite-4.1-3b-building/build_data.py            # full (training demos)
    python experiments/granite-4.1-3b-building/build_data.py --smoke    # 1 building, 6 Q&A, print

Edit the SPEC / prompts to change the recipe. NEVER edit packs/*/scorer.py or prepare.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
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
N = 45                      # questions authored per training building (pre-gate)
QA_DIR = EXP_DIR / "data" / "realistic_qa"   # git-ignored (experiments/**/data/)
MIN_ANCHORS = 2             # drop demos with <2 in-slice anchors (too little signal)
WORKERS = 10

SYSTEM_PROMPT = (  # the student persona; must match train.py
    "You are a knowledgeable building engineer assistant. Answer the question accurately and "
    "specifically, grounding your answer in the building's equipment, sensors, topology, and "
    "semantic model (Brick / ASHRAE 223P). Name the relevant entities and explain your reasoning.")

SENIOR_SYS = ("You are a senior building services engineer (HVAC, BMS/controls, commissioning). "
    "Below is ALL available data for one building — study it carefully.\n\n"
    "--- BUILDING DATA ---\n{corpus}\n--- END BUILDING DATA ---")

TASK = """Author {n} questions that a real building ENGINEER or OPERATOR would actually ask about THIS building — in their own words and shorthand. Do NOT use a fixed taxonomy; just span what they really need: look up a spec/value, IDENTIFY which component does X, TRACE/STRUCTURE what connects to what (air & water paths), explain control LOGIC / a sequence, DIAGNOSE (what would you check, and why, when something is wrong), CLARIFY an ambiguous/mislabeled tag, check SAFETY interlocks, and fetch/inspect TIME-SERIES. Mix the voice: some engineer (precise, uses tags), some operator (casual). Vary phrasing. Favor questions whose answer NAMES SEVERAL entities/tags/values (trace, structure, diagnose, clarify) — those teach completeness — alongside simple single-fact lookups.

For each, give the correct grounded `answer` and its `anchors` — the specific facts the answer MUST contain to be judged right: numeric values, vendor tags, file paths, component names, time windows, or the items of a checklist. Every anchor MUST come from the data above — invent nothing; keep anchors minimal-but-complete.

Return ONLY a JSON array, no prose, no markdown fences:
[{{"persona":"engineer|operator","intent":"<one free-form word>","question":"...","answer":"...","anchors":["...","..."],"source":"<file or entity>"}}, ...]"""

REANSWER_SYS = (  # v3 "complete" style — the best SFT init (run C); completeness beats brevity for anchor recall
    "You are a knowledgeable building engineer assistant. Answer using ONLY the building data slice "
    "below, but be THOROUGH — an expert answer states EVERY relevant fact the slice contains, because "
    "graders check for each one. For each entity involved, give whichever of these the slice provides: "
    "its exact tag/name, its plain-English description or role, its ontology class (brick:/s223:), its "
    "file path / ExternalReference, and what it connects to. When the answer is a set (connection "
    "points, a path, matching sensors, diagnostic checks), list ALL of them in order — NEVER collapse "
    "to one example or a vague summary. Copy tags, paths, classes and numbers VERBATIM (exact prefix, "
    "suffix, extension, digits). No preamble. A complete, specific answer is always better than a "
    "short one. If a needed fact is genuinely absent from the slice, answer with what IS there and do "
    "not invent the rest.")


def ob(ctx: str, q: str) -> str:  # must match train.py / eval_judge.py eval prompt
    return ("Use only the building's data below to answer.\n\n"
            f"--- BUILDING DATA ---\n{ctx}\n--- END BUILDING DATA ---\n\nQuestion: {q}")


# --- deterministic, lenient-on-separators anchor matcher (for the grounding gate) -------------
def _norm(s: str) -> str:
    s = s.lower().replace("_", " ")
    s = re.sub(r"[^a-z0-9./:\- ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def anchor_in(anchor: str, text: str) -> bool:
    a = anchor.strip()
    na, nt = _norm(a), _norm(text)
    if not na:
        return False
    if re.fullmatch(r"-?\d+(\.\d+)?", a):                       # bare number: token boundary
        return re.search(rf"(?<![\d.]){re.escape(a)}(?![\d.])", text) is not None
    if "/" in a or a.endswith((".txt", ".ttl", ".csv", ".md", ".pdf", ".png")):
        base = _norm(a.rsplit("/", 1)[-1])
        return na in nt or (len(base) >= 4 and base in nt)
    if na in nt:
        return True
    toks = [t for t in na.split() if len(t) >= 2]
    return len(toks) >= 2 and toks[0] in nt and sum(t in nt for t in toks) >= max(2, len(toks) - 1)


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


def reanswer_on_slice(slice_text: str, q: str, max_tokens: int = 650) -> str:
    """Teacher answers using ONLY the slice the student sees → grounded demo gold."""
    return llm.generate(TEACHER, system=REANSWER_SYS, user=ob(slice_text, q), max_tokens=max_tokens)


def grounded_demo(it: dict, building: str, corpus: str) -> dict | None:
    """Retrieve the student's slice, re-answer on it, gate by in-slice anchors. None = dropped."""
    slice_text = retrieve(corpus, it["question"], ignore=[building])
    in_slice = [a for a in it["anchors"] if anchor_in(a, slice_text)]
    if len(in_slice) < MIN_ANCHORS:
        return None                                            # retrieval too poor → no signal
    ans = reanswer_on_slice(slice_text, it["question"])
    if not ans or not ans.strip():
        return None
    # Grounding is ensured by answering ON the slice; the gate is a completeness filter that also
    # tolerates deterministic-matcher noise — keep if the demo names >=80% of the in-slice anchors.
    import math
    matched = [a for a in in_slice if anchor_in(a, ans)]
    if len(matched) < max(MIN_ANCHORS, math.ceil(0.8 * len(in_slice))):
        return None                                            # too incomplete → weak demo
    row = {"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": ob(slice_text, it["question"])},
                        {"role": "assistant", "content": ans.strip()}],
           "intent": it.get("intent", ""), "persona": it.get("persona", ""),
           "source": it.get("source", ""), "id": it.get("id", ""),
           "n_anchors": len(in_slice)}
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="author 6 Q&A from one training building, gate, print")
    args = ap.parse_args()

    train_dirs = [d for d in prepare.building_dirs() if d.name != HOLDOUT]
    if args.smoke:
        d = train_dirs[0]
        corpus, _ = building_corpus(d, max_chars=300000)
        items = author(d, n=6, max_tokens=5000)
        print(f"=== {len(items)} authored Q&A for '{d.name}'; gating on slice ===\n", file=sys.stderr)
        kept = 0
        for it in items:
            row = grounded_demo(it, d.name, corpus)
            tag = "KEEP" if row else "drop"
            print(f"[{tag}] ({it.get('intent','')}) {it['question']}")
            if row:
                kept += 1
                print(f"   demo A: {row['messages'][2]['content'][:240]}")
            print(f"   anchors: {it['anchors']}\n")
        print(f"kept {kept}/{len(items)}", file=sys.stderr)
        assert items, "teacher returned no parseable Q&A"
        return

    QA_DIR.mkdir(parents=True, exist_ok=True)
    rows, intent_counts = [], {}
    for d in train_dirs:
        b = d.name
        corpus, _ = building_corpus(d, max_chars=300000)
        qf = QA_DIR / f"{b}.json"
        if qf.exists() and os.environ.get("NEKAISE_REAUTHOR") != "1":
            items = json.loads(qf.read_text())          # reuse authored Q&A; re-answer only (cheap, isolates style)
        else:
            items = author(d)
            qf.write_text(json.dumps(items, ensure_ascii=False, indent=2))
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            built = list(ex.map(lambda it: grounded_demo(it, b, corpus), items))
        kept = [r for r in built if r]
        for r in kept:
            intent_counts[r["intent"]] = intent_counts.get(r["intent"], 0) + 1
        rows.extend(kept)
        print(f"  {b}: authored {len(items)} -> kept {len(kept)} grounded demos "
              f"(dropped {len(items) - len(kept)})", file=sys.stderr)

    spec = {"pack": "building", "teacher": TEACHER, "method": "slice-grounded-reanswer-gated",
            "holdout": HOLDOUT, "v": 3}   # v3 "complete" style = best SFT (run C 0.459). (v4 dense/v5 natural scored lower.)
    dd = datakit.write(EXP_DIR, spec, rows,
                       stats={"examples": len(rows), "teacher": TEACHER, "intents": intent_counts,
                              "kind": "slice-grounded gated demos"})
    print(f"SFT demos: {len(rows)} -> {dd}  (LATEST={datakit.dataset_id(spec)})", file=sys.stderr)
    print(f"  intents: {intent_counts}", file=sys.stderr)

    # Frozen exam guard: author the holdout exam ONCE; never perturb it on a rebuild.
    exam_path = REPO / "packs" / "building" / "eval_open.jsonl"
    if exam_path.exists() and os.environ.get("NEKAISE_REGEN_EXAM") != "1":
        print(f"frozen exam preserved: {exam_path} (set NEKAISE_REGEN_EXAM=1 to re-author)", file=sys.stderr)
    else:
        items = author(prepare.DATA / HOLDOUT if (prepare.DATA / HOLDOUT).exists()
                       else next(d for d in prepare.building_dirs() if d.name == HOLDOUT))
        exam = [{"id": it["id"], "persona": it.get("persona", ""), "intent": it.get("intent", ""),
                 "question": it["question"], "ground_truth": it["answer"],
                 "anchors": it["anchors"], "source": it.get("source", "")} for it in items]
        exam_path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in exam) + "\n")
        print(f"frozen exam (RE-AUTHORED): {len(exam)} -> {exam_path}", file=sys.stderr)
    print("train: python experiments/.../train.py   |   judge-eval: python experiments/.../eval_judge.py")


if __name__ == "__main__":
    main()
