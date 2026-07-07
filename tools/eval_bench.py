#!/usr/bin/env python3
"""Score a model on nekaise-bench — the INDEPENDENT building-energy QA benchmark.

Third probe of the harness, next to `building_judge` (the gap) and `domain_quiz` (the
ceiling): its questions were authored and hardened OUTSIDE this repo's corpus, so a model
that merely memorized corpus text (e.g. via CPT) inflates corpus-derived quizzes but NOT
this one. Two modes:

  # Loop metric (ceiling phase): evaluate a checkpoint directly on GPU — fast, no export.
  # Grading functions are IMPORTED from the bench so scoring matches the official harness.
  python tools/eval_bench.py --checkpoint experiments/<exp>/outputs/<stage> --split dev
  python tools/eval_bench.py --checkpoint unsloth/granite-4.1-3b --split dev   # baseline

  # Deployment parity (milestones): export to Ollama first, run the official harness as-is.
  python serve/to_ollama.py --exp <exp> --name nekaise-candidate
  python tools/eval_bench.py --model nekaise-candidate

Splits (checkpoint mode) are deterministic by question-id hash and never touch the bench
repo: `dev` (~75%) is the loop's keep/revert signal, `test` (~25%) stays FROZEN for rare
milestone checks — optimizing against `test` (or running it often) burns the benchmark.
Checkpoint mode generates greedily with a small token budget (no thinking); absolute
numbers can differ slightly from the Ollama harness — compare checkpoint-mode numbers
with checkpoint-mode numbers.

Needs a local clone of https://github.com/OpenNekaise/nekaise-bench ; location from
NEKAISE_BENCH_DIR, defaulting to ../nekaise-bench next to this repo.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = Path(os.environ.get("NEKAISE_BENCH_DIR", REPO.parent / "nekaise-bench"))


def bench_module():
    """Import the bench's eval module so grading is EXACTLY the official harness's."""
    runner = BENCH / "eval_ollama.py"
    if not runner.exists():
        sys.exit(f"nekaise-bench not found at {BENCH} — clone "
                 f"https://github.com/OpenNekaise/nekaise-bench there, or set NEKAISE_BENCH_DIR.")
    spec = importlib.util.spec_from_file_location("nekaise_bench_eval", runner)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def in_split(qid: str, split: str) -> bool:
    if split == "all":
        return True
    frozen = int(hashlib.md5(qid.encode()).hexdigest(), 16) % 4 == 0   # ~25%, deterministic
    return frozen if split == "test" else not frozen


def load_questions(split: str, limit: int) -> list[dict]:
    rows = [json.loads(l) for l in (BENCH / "questions.jsonl").open(encoding="utf-8") if l.strip()]
    rows = [q for q in rows if in_split(q["id"], split)]
    return rows[:limit] if limit else rows


def eval_checkpoint(args) -> dict:
    bench = bench_module()
    questions = load_questions(args.split, args.limit)
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.checkpoint, max_seq_length=4096,
        load_in_4bit=args.load_4bit, dtype=None)
    if hasattr(tok, "tokenizer"):  # multimodal processor (e.g. Qwen3.5/VL) -> use the text tokenizer
        tok = tok.tokenizer
    FastLanguageModel.for_inference(model)

    records, t0 = [], time.time()
    for i, q in enumerate(questions):
        if q["track"] == "mcq":
            system, user = bench.MCQ_SYSTEM, bench.mcq_prompt(q)
        else:
            system, user = bench.OPEN_SYSTEM, q["question"]
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        reply = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        ans = bench.strip_think(reply)
        if q["track"] == "mcq":
            pred = bench.extract_letter(ans, len(q["choices"]), q["choices"])
            ok = pred == q["answer"]
        else:
            pred, ok = ans, bench.grade_open(ans, q["answer"], q.get("aliases"))
        records.append({"id": q["id"], "track": q["track"], "topic": q["topic"],
                        "difficulty": q["difficulty"], "reply": reply,
                        "pred": pred, "gold": q["answer"], "correct": ok})
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(questions)}] running acc="
                  f"{sum(r['correct'] for r in records)/(i+1):.3f}", flush=True)

    def acc(rs):
        return round(sum(r["correct"] for r in rs) / len(rs), 4) if rs else None

    out_dir = REPO / "workspace" / "bench_eval"        # git-ignored scratch
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.-]+", "_", str(args.checkpoint))
    (out_dir / f"{safe}.{args.split}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n")
    return {"model": str(args.checkpoint), "split": args.split, "n": len(records),
            "overall": acc(records),
            "by_track": {t: acc([r for r in records if r["track"] == t]) for t in ("mcq", "open")},
            "minutes": round((time.time() - t0) / 60, 1)}


def eval_ollama(args) -> dict | None:
    cmd = [sys.executable, str(BENCH / "eval_ollama.py"), "--models", args.model]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    proc = subprocess.run(cmd, cwd=BENCH, capture_output=True, text=True)
    sys.stderr.write(proc.stderr)
    result = None
    for line in proc.stdout.splitlines():
        print(line)
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
    if proc.returncode != 0 or result is None:
        print("bench run failed", file=sys.stderr)
        return None
    result["split"] = "all"
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--model", help="Ollama model name (official harness, deployment parity)")
    who.add_argument("--checkpoint", help="local checkpoint dir or HF id (fast loop metric, GPU)")
    ap.add_argument("--split", choices=("dev", "test", "all"), default="dev",
                    help="checkpoint mode only; test is FROZEN for milestones (default: dev)")
    ap.add_argument("--limit", type=int, default=0, help="only first N questions (smoke test)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--load-4bit", action="store_true", help="4-bit load (faster, less faithful)")
    args = ap.parse_args()

    if args.model:
        bench_module()                      # existence check with a clear error
        result = eval_ollama(args)
    else:
        result = eval_checkpoint(args)
    if result is None:
        return 1
    # The studio-style line the loop logs next to METRIC.
    print(f"BENCH[{result['model']}@{result['split']}] nekaise_bench={result['overall']:.4f} "
          f"n={result['n']} by_track={result.get('by_track')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
