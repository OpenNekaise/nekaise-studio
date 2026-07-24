#!/usr/bin/env python3
"""eval_probes.py — score a model on corpus_probes (the pure-CPT loop metric) via the
gym runner. Base-model COMPLETION mode; the verdict per probe is gym.verifiers.
numeric_cloze — the same function training rewards import (R1).

    python tools/eval_probes.py --run-id <run_id>
    python tools/eval_probes.py --target server:nekaise-1b@http://localhost:8000 --split dev
    python tools/eval_probes.py --checkpoint openbmb/MiniCPM5-1B-Base   # ckpt: shorthand
    python tools/eval_probes.py --run-id <run_id> --doc-ids workspace/coapt/pool.jsonl

Local checkpoints run on the vLLM offline engine; served/API models on the OpenAI-
compatible path — one runner, no transformers.generate.

With --doc-ids the probe set is restricted to a frozen document pool and the recorded
metric is `coapt_pool_absorption_<split>` (the CoAPT loop metric) instead of the
transfer/absorption family. Probe content is identical either way — the pool is a view,
never a regeneration.

The frozen split (~20%) is MILESTONE-ONLY: requires --milestone; every use is audited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Native helpers installed beside the selected Python (notably ``ninja`` used by
# vLLM/flashinfer) must be discoverable even when this script is launched with an
# absolute interpreter path instead of activating the environment.
os.environ["PATH"] = f"{Path(sys.executable).resolve().parent}:{os.environ.get('PATH', '')}"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lib"))

from runstore import RunStore, new_run_id  # noqa: E402
from workspace import Workspace  # noqa: E402

ACTIVE_WORKSPACE = Workspace.resolve(REPO).apply_environment()

from gym import tasks as gym_tasks  # noqa: E402
from gym.runner import evaluate, generator_for  # noqa: E402

OUT_DIR = ACTIVE_WORKSPACE.scratch_dir / "probe_eval"


def load_doc_ids(path: Path) -> set[str]:
    """Doc-id pool file: one id per line, or JSONL rows carrying a doc_id field."""
    ids: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            doc_id = json.loads(line).get("doc_id")
            if doc_id:
                ids.add(doc_id)
        else:
            ids.add(line)
    if not ids:
        raise SystemExit(f"empty doc-id pool: {path}")
    return ids


def run_id_from_target(target: str) -> str | None:
    if not target.startswith("ckpt:"):
        return None
    checkpoint = Path(target.removeprefix("ckpt:"))
    if not checkpoint.is_absolute():
        checkpoint = REPO / checkpoint
    checkpoint_meta = checkpoint / "meta.json"
    if not checkpoint_meta.is_file():
        return None
    return json.loads(checkpoint_meta.read_text()).get("run_id")


def main() -> None:
    ap = argparse.ArgumentParser()
    who = ap.add_mutually_exclusive_group()
    who.add_argument("--target", help="ckpt:<path|hf-id> or server:<model>@<base_url>")
    who.add_argument("--checkpoint", help="shorthand for ckpt:<value>")
    ap.add_argument("--run-id",
                    help="immutable training run to evaluate; combine with --target to "
                         "run inference through a persistent server while recording "
                         "metrics on this run")
    ap.add_argument("--split", default="dev", choices=["dev", "frozen", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--doc-ids", metavar="PATH",
                    help="restrict probes to a frozen doc-id pool (one id per line, or "
                         "JSONL rows with a doc_id field); records the CoAPT pool metric")
    ap.add_argument("--milestone", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="bounded plumbing check: do not write an effectiveness result")
    ap.add_argument("--record-reference", metavar="EXPERIMENT",
                    help="register an untrained target as an R0 reference run")
    args = ap.parse_args()
    if not (args.run_id or args.target or args.checkpoint):
        sys.exit("one of --run-id, --target, or --checkpoint is required")
    store = RunStore(REPO)
    run_id = args.run_id
    if run_id and not (args.target or args.checkpoint):
        run = store.get_run(run_id)
        if not run.get("checkpoint_path") and run.get("kind") == "reference" \
                and (run.get("metadata") or {}).get("reference_target"):
            target = run["metadata"]["reference_target"]
        else:
            target = f"ckpt:{store.resolve_checkpoint(run_id)}"
    else:
        target = args.target or f"ckpt:{args.checkpoint}"
        if run_id:
            store.get_run(run_id)          # fail fast on a bad run id
        else:
            inferred = run_id_from_target(target)
            run_id = inferred if inferred and store.exists(inferred) else None

    if args.record_reference:
        if run_id:
            sys.exit("--record-reference requires an unregistered --checkpoint/--target")
        run_id = new_run_id()
        store.create_run(
            experiment=args.record_reference, stage="cpt", kind="reference",
            model=target.removeprefix("ckpt:"), seed=3407,
            command=[sys.executable, *sys.argv],
            metadata={"reference_target": target, "untrained": True},
            run_id=run_id, status="trained",
        )

    if args.smoke and (args.limit <= 0 or args.split != "dev"):
        sys.exit("--smoke requires a positive --limit and --split dev")

    if args.split == "frozen":
        if not args.milestone:
            sys.exit("--split frozen is MILESTONE-ONLY: pass --milestone (and mean it). "
                     "The loop's signal is --split dev.")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with (OUT_DIR / "frozen_split_audit.log").open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} target={target}\n")

    rows = gym_tasks.load("corpus_probes", split=args.split,
                          n=0 if args.doc_ids else args.limit)
    pool_ids: set[str] | None = None
    if args.doc_ids:
        pool_ids = load_doc_ids(Path(args.doc_ids))
        rows = [row for row in rows if row.tags.get("doc_id") in pool_ids]
        if args.limit:
            rows = rows[:args.limit]
        if not rows:
            sys.exit(f"no {args.split} probes match the doc-id pool {args.doc_ids}")
    split_hash = hashlib.sha256(json.dumps(
        [{"id": row.id, "prompt": row.prompt} for row in rows],
        ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    probe_provenance = json.loads(
        (REPO / "gym/tasks/corpus_probes/provenance.json").read_text())
    evaluator_version = hashlib.sha256(
        probe_provenance["fingerprint"].encode() + Path(__file__).read_bytes()
    ).hexdigest()[:16]
    print(f"[probes] {target} on {len(rows)} {args.split} probes")
    if run_id:
        store.transition(run_id, "evaluating")

    def progress(done, total, records):
        if done % (args.batch_size * 10) < args.batch_size or done == total:
            acc = sum(r["correct"] for r in records) / done
            print(f"  [{done}/{total}] running acc={acc:.3f}", flush=True)

    try:
        report = evaluate(rows, generator_for(target), max_tokens=args.max_tokens,
                          batch_size=args.batch_size, progress=progress)
    except BaseException as exc:
        if run_id:
            store.fail(run_id, f"evaluation failed: {type(exc).__name__}: {exc}")
        raise

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.-]+", "_", target)
    suffix = (".pool" if args.doc_ids else "") + (".smoke" if args.smoke else "")
    records_file = OUT_DIR / f"{safe}.{args.split}{suffix}.jsonl"
    records_file.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in report["records"]) + "\n")

    if pool_ids is not None:
        pool_records = [r for r in report["records"] if r.get("track") == "absorption"]
        pool_absorption = (round(sum(r["correct"] for r in pool_records)
                                 / len(pool_records), 4) if pool_records else None)
        by_doc = {}
        for doc_id in sorted({r["doc_id"] for r in pool_records}):
            doc_rows = [r for r in pool_records if r["doc_id"] == doc_id]
            by_doc[doc_id] = round(sum(r["correct"] for r in doc_rows) / len(doc_rows), 4)
        print(f"PROBES[{target}] split={args.split} pool_docs={len(by_doc)} "
              f"n={len(pool_records)} coapt_pool_absorption={pool_absorption}")
        if run_id:
            diagnostic = args.smoke or args.limit > 0
            store.record_metric(
                run_id, name=f"coapt_pool_absorption_{args.split}",
                value=pool_absorption, n=len(pool_records), split=args.split,
                split_hash=split_hash, evaluator="corpus_probes",
                evaluator_version=evaluator_version, diagnostic=diagnostic,
                metadata={"by_doc": by_doc, "target": target,
                          "pool_size": len(pool_ids),
                          "probe_fingerprint": probe_provenance["fingerprint"]},
            )
            if args.smoke:
                store.transition(run_id, "succeeded")
        if args.smoke:
            print(f"SMOKE probe_eval=pass n={report['n']} (scores are diagnostic only)")
        elif pool_absorption is not None and args.limit <= 0:
            print(f"METRIC coapt_pool_absorption_{args.split}={pool_absorption:.4f}")
        print("EVAL_RESULT " + json.dumps({
            "run_id": run_id, "target": target, "split": args.split,
            "n": report["n"], "pool_docs": len(by_doc),
            "coapt_pool_absorption": pool_absorption,
            "records_file": str(records_file),
            "diagnostic": args.smoke, "split_hash": split_hash,
        }, ensure_ascii=False, sort_keys=True))
        return

    absorption = report["by_track"].get("absorption")
    transfer_micro = report["by_track"].get("transfer")
    transfer_records = [r for r in report["records"] if r.get("track") == "transfer"]
    transfer_by_topic = {}
    for topic in sorted({r["topic"] for r in transfer_records}):
        topic_rows = [r for r in transfer_records if r["topic"] == topic]
        transfer_by_topic[topic] = round(
            sum(r["correct"] for r in topic_rows) / len(topic_rows), 4)
    transfer_macro = (round(sum(transfer_by_topic.values()) / len(transfer_by_topic), 4)
                      if transfer_by_topic else None)
    print(f"PROBES[{target}] split={args.split} n={report['n']} "
          f"transfer_macro={transfer_macro} transfer_micro={transfer_micro} "
          f"absorption={absorption}")
    if run_id:
        diagnostic = args.smoke or args.limit > 0
        store.record_metric(
            run_id, name=f"corpus_transfer_macro_{args.split}", value=transfer_macro,
            n=len(transfer_records), split=args.split, split_hash=split_hash,
            evaluator="corpus_probes", evaluator_version=evaluator_version,
            diagnostic=diagnostic,
            metadata={"by_topic": transfer_by_topic, "target": target,
                      "probe_fingerprint": probe_provenance["fingerprint"]},
        )
        if transfer_micro is not None:
            store.record_metric(
                run_id, name=f"corpus_transfer_micro_{args.split}",
                value=transfer_micro, n=len(transfer_records), split=args.split,
                split_hash=split_hash, evaluator="corpus_probes",
                evaluator_version=evaluator_version, diagnostic=True,
            )
        if absorption is not None:
            absorption_n = sum(
                r.get("track") == "absorption" for r in report["records"])
            store.record_metric(
                run_id, name=f"corpus_absorption_{args.split}", value=absorption,
                n=absorption_n, split=args.split, split_hash=split_hash,
                evaluator="corpus_probes", evaluator_version=evaluator_version,
                diagnostic=True,
            )
        if args.smoke:
            store.transition(run_id, "succeeded")
    if args.smoke:
        print(f"SMOKE probe_eval=pass n={report['n']} (scores are diagnostic only)")
    if transfer_macro is not None and not args.smoke and args.limit <= 0:
        print(f"METRIC corpus_transfer_macro_{args.split}={transfer_macro:.4f}")
    print("EVAL_RESULT " + json.dumps({
        "run_id": run_id, "target": target, "split": args.split,
        "n": report["n"], "absorption": absorption,
        "transfer_micro": transfer_micro, "transfer_macro": transfer_macro,
        "transfer_by_topic": transfer_by_topic,
        "records_file": str(records_file),
        "diagnostic": args.smoke, "split_hash": split_hash,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
    # The vLLM offline engine can SIGABRT during teardown AFTER all metrics are
    # recorded and records written; exit deterministically for drivers.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
