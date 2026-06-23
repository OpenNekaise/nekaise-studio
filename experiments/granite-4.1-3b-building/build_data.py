#!/usr/bin/env python3
"""build_data.py — DATA recipe for the building pack: distill a senior building engineer.

Opus 4.8, prompted AS a senior building / HVAC / BMS engineer, reads a building's real
semantic model (Brick + ASHRAE 223P ontology) and writes English question-answer pairs that
would train a junior building engineer to understand and operate THAT building. The pairs
become SFT data for the small model. Evaluation stays separate and verifiable — the fixed
ontology scorer in packs/building/scorer.py measures progress objectively.

Cross-building: training Q&A is generated only from the NON-held-out buildings; the held-out
building (NEKAISE_HOLDOUT, default rio10) is reserved for eval. Each building's ontology is
sent as the CACHED system context, so repeated/large calls bill ~0.1x on the shared prefix.

    set -a; source /home/zengp/Code/KebAgent/.env; set +a   # provides ANTHROPIC_API_KEY
    python experiments/granite-4.1-3b-building/build_data.py            # build + cache full set
    python experiments/granite-4.1-3b-building/build_data.py --smoke    # 1 building, 5 Q&A, print

Edit SPEC / the prompts to change the recipe. NEVER edit packs/*/scorer.py (the referee).
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
import prepare  # noqa: E402  (building data locator)

HOLDOUT_BUILDING = os.environ.get("NEKAISE_HOLDOUT", "rio10")
MAX_ONTOLOGY_CHARS = 60_000  # cap grounding context per building (TTLs are small; safety bound)

# ================================== SPEC (the knobs) =================================
SPEC = {
    "pack": "building",
    "teacher": "anthropic:claude-opus-4-8",
    "questions_per_building": 40,
    "max_tokens": 12000,
    "holdout": HOLDOUT_BUILDING,
}

# The teacher's persona + the building's ontology (this is the CACHED system prefix).
TEACHER_SYSTEM = """You are a senior building services engineer with 20+ years of experience in \
HVAC, building automation / BMS, controls, and commissioning. You are fluent in semantic \
building models — Brick Schema and ASHRAE Standard 223P — and you mentor junior engineers.

Below is the semantic model (RDF/Turtle) for one specific building: "{building}". Study it \
carefully — it defines the building's equipment, sensors, points, systems, and how they \
connect (s223:cnx connections, brick/s223 classes, rdfs:comment descriptions).

--- BUILDING ONTOLOGY: {building} ---
{ontology}
--- END ONTOLOGY ---"""

# The generation task (user turn).
TEACHER_TASK = """Design {n} high-quality question-and-answer training pairs, in English, that \
would teach a junior building engineer to understand and operate THIS building.

Ground every pair in the ontology above — reference the building's REAL entities, equipment, \
sensors, points, and connections by their actual names/identifiers. Do not invent anything \
that is not in the model.

Cover a mix of:
- Equipment & classification: what a given entity is, its Brick/223P class, what it does.
- Topology: what connects to what (s223:cnx), what serves or feeds what, the air/water path.
- Sensors & points: what is measured where, and how a reading would be interpreted.
- Operational reasoning: practical "what would you check, and why" troubleshooting a mentor asks.

Each answer must be correct per the ontology, specific (name the entities), and explain the \
reasoning a senior engineer would use — not a one-word fact. Vary difficulty from basic \
identification to multi-step reasoning.

Return ONLY a JSON array, with no prose and no markdown fences:
[{{"question": "...", "answer": "..."}}, ...]"""

# What the STUDENT (small model) is trained to be.
STUDENT_SYSTEM = (
    "You are a knowledgeable building engineer assistant. Answer questions about the building "
    "accurately and specifically, grounding your answers in its equipment, sensors, topology, "
    "and semantic model. Explain your reasoning like a senior engineer mentoring a colleague."
)
# =====================================================================================


def building_ontology(building: str) -> str:
    """Concatenated Turtle text for a building (the teacher's grounding context)."""
    d = prepare.DATA / building
    parts = []
    for ttl in sorted(d.rglob("*.ttl")):
        try:
            parts.append(f"# file: {ttl.name}\n{ttl.read_text(errors='replace')}")
        except Exception as e:
            print(f"  ! {building}: skip {ttl.name} ({e})", file=sys.stderr)
    return ("\n\n".join(parts))[:MAX_ONTOLOGY_CHARS]


def training_buildings() -> list[str]:
    """All buildings with an ontology, minus the held-out one."""
    return [d.name for d in prepare.building_dirs() if d.name != SPEC["holdout"]]


def _parse_qa(text: str) -> list[dict]:
    """Pull the JSON array of {question, answer} out of the teacher's reply, robustly.

    Tolerates truncation: if the array didn't close (output hit max_tokens), salvage by
    closing it after the last complete object, so we keep the Q&A that did come through.
    """
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start = t.find("[")
    if start == -1:
        return []
    body = t[start:]
    candidates = []
    end = body.rfind("]")
    if end != -1:
        candidates.append(body[: end + 1])           # well-formed array
    last_obj = body.rfind("}")
    if last_obj != -1:
        candidates.append(body[: last_obj + 1] + "]")  # salvage truncated tail
    for c in candidates:
        try:
            arr = json.loads(c)
        except json.JSONDecodeError:
            continue
        out = [
            {"question": str(x["question"]), "answer": str(x["answer"])}
            for x in arr
            if isinstance(x, dict) and x.get("question") and x.get("answer")
        ]
        if out:
            return out
    return []


def generate_for_building(building: str, n: int, max_tokens: int) -> list[dict]:
    system = TEACHER_SYSTEM.format(building=building, ontology=building_ontology(building))
    task = TEACHER_TASK.format(n=n)
    reply = llm.generate(SPEC["teacher"], system=system, user=task, max_tokens=max_tokens)
    return _parse_qa(reply)


def _to_sft(qa: dict) -> dict:
    return {"messages": [
        {"role": "system", "content": STUDENT_SYSTEM},
        {"role": "user", "content": qa["question"]},
        {"role": "assistant", "content": qa["answer"]},
    ]}


def build(spec: dict):
    for b in training_buildings():
        qa = generate_for_building(b, spec["questions_per_building"], spec["max_tokens"])
        print(f"  {b}: {len(qa)} Q&A", file=sys.stderr)
        for item in qa:
            yield _to_sft(item)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="generate 5 Q&A from one training building and print them (tiny cost)")
    args = ap.parse_args()

    if args.smoke:
        return smoke()

    d = datakit.cached(
        EXP_DIR, SPEC, lambda: build(SPEC),
        stats=lambda rows: {"examples": len(rows), "teacher": SPEC["teacher"],
                            "buildings": training_buildings(), "holdout": SPEC["holdout"]},
    )
    prov = datakit.provenance(d)
    print(f"dataset {prov['dataset_id']}: {prov['n']} Q&A examples -> {d}")
    print(f"  teacher={SPEC['teacher']}  train_buildings={training_buildings()}  holdout={SPEC['holdout']}")
    print("train with: DATASET=auto  (train.py reads data/LATEST); eval is the fixed building pack")


def smoke() -> None:
    b = training_buildings()[0]
    print(f"[smoke] teacher={SPEC['teacher']}  building={b}", file=sys.stderr)
    qa = generate_for_building(b, 5, 4000)
    print(f"\n=== {len(qa)} Q&A generated for '{b}' (senior building engineer → junior) ===\n")
    for i, item in enumerate(qa, 1):
        print(f"Q{i}: {item['question']}")
        print(f"A{i}: {item['answer']}\n")
    assert qa, "teacher returned no parseable Q&A"


if __name__ == "__main__":
    main()
