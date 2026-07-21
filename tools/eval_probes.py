#!/usr/bin/env python3
"""eval_probes.py — score a checkpoint on the corpus_probes pack (the pure-CPT loop metric).

Base-model COMPLETION mode: probes are raw sentence prefixes (no chat template) and the
model greedily continues; a probe is correct iff the first number in the continuation
equals the gold value (packs/corpus_probes/scorer.is_correct). The METRIC value is
absorption-probe accuracy on the requested split; transfer probes are reported alongside
as a diagnostic, never the keep/revert signal.

    python tools/eval_probes.py --checkpoint unsloth/Qwen3.5-0.8B --exp agentic-cpt
    python tools/eval_probes.py --checkpoint experiments/agentic-cpt/outputs/cpt --split frozen --milestone

The frozen split (~20%) is milestone-only: it requires --milestone and every use is
appended to workspace/probe_eval/frozen_split_audit.log.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lib"))
from pack import load as load_pack  # noqa: E402
from results import log_result  # noqa: E402

OUT_DIR = REPO / "workspace" / "probe_eval"


def eval_checkpoint(ckpt: str, rows: list[dict], probes, batch_size: int) -> dict:
    import torch
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        model_name=ckpt, max_seq_length=1024, load_in_4bit=False, dtype=None)
    if hasattr(tok, "tokenizer"):
        tok = tok.tokenizer
    FastLanguageModel.for_inference(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    records = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        enc = tok([r["question"] for r in batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=768).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=16, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for r, out_ids in zip(batch, gen):
            cont = tok.decode(out_ids[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            records.append({"id": r["id"], "kind": r["track"], "topic": r["topic"],
                            "continuation": cont, "gold": r["answer"],
                            "correct": probes.is_correct(cont, r["answer"])})
        if (i // batch_size) % 10 == 0:
            done = sum(x["correct"] for x in records)
            print(f"  [{Path(ckpt).name}] {len(records)}/{len(rows)} "
                  f"acc={done/len(records):.3f}", flush=True)
    return records


def acc(rs):
    return round(sum(r["correct"] for r in rs) / len(rs), 4) if rs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "frozen", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--exp", default=None, help="experiment name for the results ledger")
    ap.add_argument("--milestone", action="store_true")
    args = ap.parse_args()

    if args.split == "frozen":
        if not args.milestone:
            sys.exit("--split frozen is MILESTONE-ONLY: pass --milestone (and mean it). "
                     "The loop's signal is --split dev.")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with (OUT_DIR / "frozen_split_audit.log").open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                    f"checkpoint={args.checkpoint}\n")

    probes = load_pack("corpus_probes")
    rows = probes.load_split(args.split, n=args.limit)
    print(f"[probes] {args.checkpoint} on {len(rows)} {args.split} probes")
    records = eval_checkpoint(args.checkpoint, rows, probes, args.batch_size)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.-]+", "_", str(args.checkpoint).rstrip("/"))
    (OUT_DIR / f"{safe}.{args.split}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n")

    absorption = [r for r in records if r["kind"] == "absorption"]
    transfer = [r for r in records if r["kind"] == "transfer"]
    value = acc(absorption)
    print(f"PROBES[{args.checkpoint}] split={args.split} n={len(records)} "
          f"absorption={value} transfer={acc(transfer)}")
    if args.exp:
        log_result(REPO / "experiments" / args.exp, kind="eval", checkpoint=str(args.checkpoint),
                   split=args.split, metric=f"corpus_probes_{args.split}", value=value,
                   n=len(absorption), transfer=acc(transfer))
    print(f"METRIC corpus_probes_{args.split}={value:.4f}")


if __name__ == "__main__":
    main()
