#!/usr/bin/env python3
"""build_data.py — the DATA recipe (MUTABLE: edit this to change WHAT the model trains on).

Builds a cached, provenance-tracked SFT dataset by sampling solutions from a model and
keeping only the ones the pack's FIXED scorer marks correct (rejection sampling). One
mechanism, two methods — only the `source` model differs:

  - distillation : source = a strong TEACHER (anthropic:claude-… or ollama:qwen3.6:27b),
                   n_samples=1 — keep the teacher's correct chain-of-thought solutions.
  - RFT / STaR   : source = the STUDENT being trained, n_samples>1 with temperature —
                   keep each correct sample; SFT on the model's own successes.

`train.py` then trains on the resulting artifact (DATASET="auto" reads data/LATEST).

    python experiments/granite-4.1-3b-gsm8k/build_data.py             # build per SPEC, cache
    python experiments/granite-4.1-3b-gsm8k/build_data.py --self-test # offline-ish sanity check

Edit SPEC to change the recipe. NEVER edit packs/*/scorer.py (the referee). A changed
SPEC produces a new cached artifact; an unchanged SPEC reuses the cache.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "lib"))

import datakit  # noqa: E402
import llm  # noqa: E402
from pack import load as load_pack  # noqa: E402

# ================================== SPEC (the knob) ==================================
SPEC = {
    "pack": "gsm8k",
    "split": "train",
    "kind": "sft",
    "source": "ollama:qwen3.6:27b",   # teacher (distill). For RFT, point at the student.
    "n_samples": 1,                    # >1 = rejection sampling; raise temperature too.
    "temperature": 0.0,
    "max_questions": 300,
    "max_tokens": 640,
    "system": (
        "You are a careful math tutor. Solve the problem step by step, then give the "
        "final numeric answer on a new line as '#### <answer>'."
    ),
}
# ====================================================================================


def build(spec: dict):
    """Yield SFT rows: questions paired with model solutions the FIXED scorer accepts."""
    pack = load_pack(spec["pack"])
    rows = pack.load_split(spec["split"], spec["max_questions"])
    kept = 0
    for i, ex in enumerate(rows):
        q, gold = ex["question"], ex["answer"]
        for _ in range(spec["n_samples"]):
            sol = llm.generate(
                spec["source"], system=spec["system"], user=q,
                temperature=spec["temperature"], max_tokens=spec["max_tokens"],
            )
            if pack.is_correct(sol, gold):
                kept += 1
                yield {"messages": [
                    {"role": "system", "content": spec["system"]},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": sol},
                ]}
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(rows)} questions, {kept} kept", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true",
                    help="offline-ish pipeline check on 3 inline questions (needs only Ollama)")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    d = datakit.cached(
        EXP_DIR, SPEC, lambda: build(SPEC),
        stats=lambda rows: {"kept": len(rows), "source": SPEC["source"]},
    )
    prov = datakit.provenance(d)
    print(f"dataset {prov['dataset_id']}: {prov['n']} examples -> {d}")
    print(f"  source={SPEC['source']} split={SPEC['split']} n_samples={SPEC['n_samples']}")
    print("train with: DATASET=auto   (train.py reads data/LATEST)")


def self_test() -> None:
    """Exercise the full data layer locally: Ollama generate -> scorer filter -> cache."""
    pack = load_pack("gsm8k")
    model = "ollama:granite4.1:3b"  # small + fast; we only need the pipeline to run
    toy = [
        {"question": "What is 2 + 2? Give the final answer as '#### <n>'.", "answer": "#### 4"},
        {"question": "What is 10 - 3? Give the final answer as '#### <n>'.", "answer": "#### 7"},
        {"question": "What is 5 * 6? Give the final answer as '#### <n>'.", "answer": "#### 30"},
    ]
    spec = {"selftest": True, "source": model, "n": len(toy)}

    def gen():
        kept = 0
        for ex in toy:
            sol = llm.generate(model, system=SPEC["system"], user=ex["question"],
                               temperature=0.0, max_tokens=256)
            ok = pack.is_correct(sol, ex["answer"])
            print(f"  {ex['question'][:24]!r} correct={ok}  sol={sol.strip()[:50]!r}", file=sys.stderr)
            if ok:
                kept += 1
                yield {"messages": [
                    {"role": "user", "content": ex["question"]},
                    {"role": "assistant", "content": sol},
                ]}
        print(f"  kept {kept}/{len(toy)}", file=sys.stderr)

    d = datakit.cached(EXP_DIR, spec, gen, stats=lambda r: {"kept": len(r)})
    prov = datakit.provenance(d)
    rows = datakit.read_dir(d)
    assert (d / "provenance.json").exists()
    assert datakit.exists(EXP_DIR, spec), "cache lookup should hit after build"
    assert datakit.latest_dir(EXP_DIR) == d, "LATEST pointer should point at this artifact"
    print(f"SELF-TEST OK: id={prov['dataset_id']} n={prov['n']} rows_read={len(rows)} dir={d}")


if __name__ == "__main__":
    main()
