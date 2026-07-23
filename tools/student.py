#!/usr/bin/env python3
"""student.py — batch student-model inference for the CoAPT loop (vLLM offline).

Three jobs, one engine, JSONL in / JSONL out:

    $NEKAISE_EVAL_PYTHON tools/student.py score  --run-id <id> --in docs.jsonl   --out nll.jsonl
    $NEKAISE_EVAL_PYTHON tools/student.py draft  --run-id <id> --in prompts.jsonl --out drafts.jsonl
    $NEKAISE_EVAL_PYTHON tools/student.py answer --run-id <id> --in questions.jsonl --out answers.jsonl

- `score`  reads rows with a `text` field and appends `nll` (mean per-token negative
  log-likelihood) + `scored_tokens`. The diagnosis signal: how foreign a document still
  is to the current student.
- `draft`  reads rows with a `prompt` field and appends `draft` — the student's raw
  continuation. CoAPT-CPT's r_S: the draft exists to EXPOSE the student's state, so it is
  generated greedily and never cleaned up here.
- `answer` reads rows with a `question` field, formats the frozen closed-book template
  (`Question: ...\nAnswer:`), and appends `answer`. CoAPT-SFT's a_S.

All other input fields pass through untouched, so provenance survives the round trip.
The model resolves like eval_probes: --run-id via the RunStore, or --checkpoint as a
path/HF id. Runs in the isolated eval environment (vLLM); no transformers.generate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ["PATH"] = f"{Path(sys.executable).resolve().parent}:{os.environ.get('PATH', '')}"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lib"))

# The frozen closed-book format. The CoAPT mixer serializes training QA with the same
# words, so what the student practices is exactly what the diagnosis asks.
ANSWER_TEMPLATE = "Question: {question}\nAnswer:"
ANSWER_STOP = ["\nQuestion:", "\n\n"]
MIN_SCORE_TOKENS = 8


def read_rows(path: Path, required: str) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if required not in row or not str(row[required]).strip():
            raise SystemExit(f"{path}:{i}: row missing required field {required!r}")
        if "id" not in row:
            raise SystemExit(f"{path}:{i}: row missing required field 'id'")
        rows.append(row)
    if not rows:
        raise SystemExit(f"no usable rows in {path}")
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def mean_nll(token_ids, prompt_logprobs) -> tuple[float | None, int]:
    """Mean NLL over prompt tokens from vLLM prompt_logprobs (first entry is None)."""
    total, count = 0.0, 0
    for tid, entry in zip(token_ids, prompt_logprobs or []):
        if entry is None:
            continue
        lp = entry.get(tid)
        if lp is None:
            continue
        value = lp.logprob if hasattr(lp, "logprob") else float(lp)
        if not math.isfinite(value):
            continue
        total += -value
        count += 1
    return (round(total / count, 4) if count else None), count


def answer_prompt(question: str) -> str:
    return ANSWER_TEMPLATE.format(question=question.strip())


class Student:
    """Thin offline-vLLM wrapper: deterministic, logprob-capable, completion-only."""

    def __init__(self, checkpoint: str, *, max_model_len: int = 4096, seed: int = 3407):
        from vllm import LLM
        self.llm = LLM(model=checkpoint, max_model_len=max_model_len, seed=seed,
                       dtype="bfloat16")
        self.tokenizer = self.llm.get_tokenizer()
        self.max_model_len = max_model_len

    def score(self, texts: list[str], *, max_len: int) -> list[tuple[float | None, int]]:
        from vllm import SamplingParams
        params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
        prompts, index = [], []
        token_lists: list[list[int]] = []
        for i, text in enumerate(texts):
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            ids = ids[:min(max_len, self.max_model_len - 1)]
            if len(ids) < MIN_SCORE_TOKENS:
                token_lists.append([])
                continue
            token_lists.append(ids)
            prompts.append({"prompt_token_ids": ids})
            index.append(i)
        results: list[tuple[float | None, int]] = [(None, 0)] * len(texts)
        if prompts:
            outs = self.llm.generate(prompts, params)
            for slot, out in zip(index, outs):
                results[slot] = mean_nll(token_lists[slot], out.prompt_logprobs)
        return results

    def complete(self, prompts: list[str], *, max_tokens: int, temperature: float,
                 stop: list[str] | None = None) -> list[str]:
        from vllm import SamplingParams
        params = SamplingParams(max_tokens=max_tokens, temperature=temperature,
                                stop=stop or [])
        outs = self.llm.generate(prompts, params)
        return [out.outputs[0].text for out in outs]


def resolve_checkpoint(args) -> str:
    if args.run_id:
        from runstore import RunStore
        return str(RunStore(REPO).resolve_checkpoint(args.run_id))
    return args.checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("score", "draft", "answer"))
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--run-id", help="immutable training run whose checkpoint to load")
    who.add_argument("--checkpoint", help="checkpoint path or HF id")
    ap.add_argument("--in", dest="input", required=True, metavar="JSONL")
    ap.add_argument("--out", dest="output", required=True, metavar="JSONL")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=2048,
                    help="score: max prompt tokens per row")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="draft/answer: generation budget (defaults: draft 300, answer 96)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    from workspace import Workspace
    Workspace.resolve(REPO).apply_environment()

    required = {"score": "text", "draft": "prompt", "answer": "question"}[args.mode]
    rows = read_rows(Path(args.input), required)
    if args.limit:
        rows = rows[:args.limit]
    checkpoint = resolve_checkpoint(args)
    print(f"[student] {args.mode} x{len(rows)} on {checkpoint}")
    student = Student(checkpoint, seed=args.seed)

    if args.mode == "score":
        scored = student.score([row["text"] for row in rows], max_len=args.max_len)
        out_rows = [{**{k: v for k, v in row.items() if k != "text"},
                     "nll": nll, "scored_tokens": n}
                    for row, (nll, n) in zip(rows, scored)]
        usable = [row["nll"] for row in out_rows if row["nll"] is not None]
        summary = {"mode": "score", "rows": len(out_rows), "scored": len(usable),
                   "mean_nll": round(sum(usable) / len(usable), 4) if usable else None}
    elif args.mode == "draft":
        max_tokens = args.max_tokens or 300
        drafts = student.complete([row["prompt"] for row in rows],
                                  max_tokens=max_tokens, temperature=args.temperature)
        out_rows = [{**row, "draft": draft} for row, draft in zip(rows, drafts)]
        summary = {"mode": "draft", "rows": len(out_rows), "max_tokens": max_tokens}
    else:
        max_tokens = args.max_tokens or 96
        answers = student.complete(
            [answer_prompt(row["question"]) for row in rows],
            max_tokens=max_tokens, temperature=args.temperature, stop=ANSWER_STOP)
        out_rows = [{**row, "answer": answer.strip()}
                    for row, answer in zip(rows, answers)]
        summary = {"mode": "answer", "rows": len(out_rows), "max_tokens": max_tokens}

    write_rows(Path(args.output), out_rows)
    print("STUDENT_RESULT " + json.dumps(
        {**summary, "checkpoint": checkpoint, "out": args.output},
        ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
