#!/usr/bin/env python3
"""Score a model on nekaise-bench — the independently-authored corpus-MASTERY benchmark.

Third probe of the harness, next to `building_judge` (the gap) and `domain_quiz` (the
ceiling). Questions are grounded in corpus documents but authored and hardened OUTSIDE this
training pipeline (verbatim-quote gate + dropped if two 27Bs answer closed-book), so it
measures whether corpus knowledge actually entered the WEIGHTS — and nothing the training
code can game. The pack (`packs/bench/scorer.py`) is the single source of truth for the
dev/test split and grading; this is the CLI over it. Two modes:

  # Milestone / advisory check (NOT a loop metric since the decoupling reform — the loop's
  # keep/revert signal is a studio-owned pack, e.g. corpus_probes): checkpoint mode, GPU.
  python tools/eval_bench.py --checkpoint experiments/<exp>/outputs/<stage> --split dev
  python tools/eval_bench.py --checkpoint unsloth/granite-4.1-3b --split dev   # baseline

  # Deployment parity (milestones): export to Ollama first, run the official harness as-is.
  python serve/to_ollama.py --exp <exp> --name nekaise-candidate
  python tools/eval_bench.py --model nekaise-candidate

The `test` split is FROZEN for milestones: it requires --milestone and every use is
appended to workspace/bench_eval/test_split_audit.log. Optimizing against it (or running
it casually) burns the benchmark.

Results are appended to the experiment's ledger (experiments/<exp>/results.jsonl) when the
checkpoint lives inside an experiment, or when --exp names one (use for base-model
baselines). Checkpoint-mode and Ollama-mode numbers are not interchangeable; compare like
with like.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lib"))
from pack import load as load_pack  # noqa: E402
from results import log_result  # noqa: E402


def eval_checkpoint(args, bench) -> dict:
    rows = bench.load_split(args.split, args.limit or None)
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.checkpoint, max_seq_length=4096,
        load_in_4bit=args.load_4bit, dtype=None)
    if hasattr(tok, "tokenizer"):  # multimodal processor (e.g. Qwen3.5/VL) -> text tokenizer
        tok = tok.tokenizer
    FastLanguageModel.for_inference(model)
    tok.padding_side = "left"      # batch generation: pad on the left so slicing is uniform
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    records, t0 = [], time.time()
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i:i + args.batch_size]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": bench.system_prompt(r["track"])},
             {"role": "user", "content": r["question"]}],
            tokenize=False, add_generation_prompt=True) for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for r, seq in zip(batch, out):
            reply = tok.decode(seq[enc.input_ids.shape[1]:], skip_special_tokens=True)
            records.append({"id": r["id"], "track": r["track"], "topic": r["topic"],
                            "difficulty": r["difficulty"], "reply": reply,
                            "pred": bench.extract_answer(reply), "gold": r["answer"],
                            "correct": bench.is_correct(reply, r["answer"])})
        done = len(records)
        if done % (args.batch_size * 2) < args.batch_size or done == len(rows):
            acc_so_far = sum(x["correct"] for x in records) / done
            print(f"  [{done}/{len(rows)}] running acc={acc_so_far:.3f}", flush=True)

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
            "mode": "checkpoint", "batch_size": args.batch_size,
            "minutes": round((time.time() - t0) / 60, 1)}


def eval_ollama(args, bench) -> dict | None:
    cmd = [sys.executable, str(bench.BENCH_DIR / "eval_ollama.py"), "--models", args.model]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    proc = subprocess.run(cmd, cwd=bench.BENCH_DIR, capture_output=True, text=True)
    sys.stderr.write(proc.stderr)
    result = None
    for line in proc.stdout.splitlines():
        print(line)
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
    if proc.returncode != 0 or result is None:
        print("bench run failed", file=sys.stderr)
        return None
    result.update(split="all", mode="ollama")
    return result


def ledger_dir(args) -> Path | None:
    """experiments/<exp>/ for the ledger: from --exp, or inferred from the checkpoint path."""
    if args.exp:
        return REPO / "experiments" / args.exp
    if args.checkpoint:
        p = Path(args.checkpoint).resolve()
        try:
            rel = p.relative_to(REPO / "experiments")
            return REPO / "experiments" / rel.parts[0]
        except ValueError:
            return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--model", help="Ollama model name (official harness, deployment parity)")
    who.add_argument("--checkpoint", help="local checkpoint dir or HF id (fast loop metric, GPU)")
    ap.add_argument("--split", choices=("dev", "test", "all"), default="dev",
                    help="checkpoint mode only; test is FROZEN and needs --milestone")
    ap.add_argument("--milestone", action="store_true",
                    help="required to run the frozen test split; every use is audited")
    ap.add_argument("--exp", help="experiment name for the results ledger (e.g. for base-model "
                                  "baselines whose checkpoint is a HF id)")
    ap.add_argument("--limit", type=int, default=0, help="only first N questions (smoke test)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--load-4bit", action="store_true", help="4-bit load (faster, less faithful)")
    args = ap.parse_args()

    if args.split == "test" and args.checkpoint:
        if not args.milestone:
            sys.exit("the test split is FROZEN for milestone checks — pass --milestone "
                     "(and log the verdict in LOG.md), or use --split dev for the loop")
        audit = REPO / "workspace" / "bench_eval" / "test_split_audit.log"
        audit.parent.mkdir(parents=True, exist_ok=True)
        with audit.open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M')} {args.checkpoint}\n")

    bench = load_pack("bench")
    result = eval_ollama(args, bench) if args.model else eval_checkpoint(args, bench)
    if result is None:
        return 1
    result["bench_version"] = bench.bench_version()
    exp = ledger_dir(args)
    if exp and exp.exists():
        log_result(exp, kind="bench_eval", **result)
    # The studio-style line the loop logs next to METRIC. bench_version is part of the
    # record: scores are only comparable within one bench dataset version.
    print(f"BENCH[{result['model']}@{result['split']}|{result['bench_version']}] "
          f"nekaise_bench={result['overall']:.4f} "
          f"n={result['n']} by_track={result.get('by_track')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
