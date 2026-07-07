#!/usr/bin/env python3
"""train.py — HOW: SFT-distill for the CEILING phase (MUTABLE recipe).

Loads a sub-4B base (or a corpus-CPT checkpoint via NEKAISE_INIT_FROM — repo-relative or
absolute path), LoRA-SFTs it on the corpus-QA distillation set (data/LATEST from
build_data.py), saves outputs/<stage>/ with provenance, then scores the PHASE METRIC —
nekaise-bench dev split — and prints the METRIC line the loop reads. Results land in the
experiment ledger (results.jsonl).

    NEKAISE_BASE_MODEL=unsloth/Qwen3.5-0.8B \
    NEKAISE_INIT_FROM=experiments/granite-4.1-3b-building/outputs/cpt_qwen08_chunk \
    NEKAISE_STAGE=distill_qwen08 python experiments/ceiling-sub4b/train.py

The frozen bench `test` split is NEVER run here — milestones only (see run-experiment skill).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
OUT_DIR = EXP_DIR / "outputs"
sys.path.insert(0, str(REPO / "lib"))
import datakit  # noqa: E402
import trainkit  # noqa: E402
from results import log_result  # noqa: E402

try:
    from runlog import RunLogger, trainer_callback  # noqa: E402
except Exception:
    RunLogger = None

# ==================================== KNOBS ====================================
BASE_MODEL   = os.environ.get("NEKAISE_BASE_MODEL", "unsloth/granite-4.1-3b")
INIT_FROM    = os.environ.get("NEKAISE_INIT_FROM")          # CPT checkpoint path (repo-rel/abs)
STAGE        = os.environ.get("NEKAISE_STAGE", "distill")
MAX_SEQ_LEN  = 2048
TIME_BUDGET_MIN = float(os.environ.get("NEKAISE_BUDGET_MIN", 60))
EVAL_AFTER   = os.environ.get("NEKAISE_EVAL_AFTER", "1") == "1"   # bench dev after training
LORA  = dict(r=16, lora_alpha=32, lora_dropout=0.0,
             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"])
TRAIN = dict(per_device_train_batch_size=4, gradient_accumulation_steps=4,
             num_train_epochs=float(os.environ.get("NEKAISE_EPOCHS", 2)),
             learning_rate=float(os.environ.get("NEKAISE_LR", 2e-4)),
             warmup_steps=10, logging_steps=5, seed=3407, lr_scheduler_type="linear")
# ===============================================================================


def bench_dev(stage_dir: Path) -> float | None:
    """The phase metric via the shared tool (fresh subprocess = clean VRAM)."""
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools" / "eval_bench.py"),
         "--checkpoint", str(stage_dir), "--split", "dev"],
        capture_output=True, text=True)
    sys.stdout.write(proc.stdout[-2000:])
    m = re.search(r"nekaise_bench=([0-9.]+)", proc.stdout)
    return float(m.group(1)) if m else None


def main() -> None:
    d = datakit.latest_dir(EXP_DIR)
    if not d:
        raise FileNotFoundError("no dataset — run build_data.py first")
    rows = datakit.read_dir(d)
    dataset_id = datakit.provenance(d)["dataset_id"]
    src = str(REPO / INIT_FROM) if INIT_FROM and not Path(INIT_FROM).is_absolute() \
        else (INIT_FROM or BASE_MODEL)
    print(f"[train] SFT-distill {src} on {dataset_id} ({len(rows)} rows) -> outputs/{STAGE}")

    model, tok = trainkit.load_model(src, max_seq_len=MAX_SEQ_LEN, lora=LORA,
                                     seed=TRAIN["seed"])
    logger = RunLogger("ceiling-sub4b", model=src, pack="bench",
                       metric="nekaise_bench_dev") if RunLogger else None
    callbacks = [trainkit.time_budget_callback(TIME_BUDGET_MIN)] \
        + ([trainer_callback(logger)] if logger else [])
    trainkit.run_sft(model, tok, rows, max_seq_len=MAX_SEQ_LEN, train_args=TRAIN,
                     out_dir=OUT_DIR, callbacks=callbacks)

    dest = trainkit.save_checkpoint(model, tok, OUT_DIR / STAGE, provenance={
        "stage": STAGE, "method": "sft-distill", "base_model": BASE_MODEL,
        "init_from": INIT_FROM, "dataset_id": dataset_id, "n_rows": len(rows),
        "lora": LORA, "train": TRAIN})
    print(f"[train] saved -> {dest}")

    del model  # free training VRAM before the eval subprocess spins up its own copy
    import torch
    torch.cuda.empty_cache()

    value = bench_dev(dest) if EVAL_AFTER else None
    if value is not None:
        trainkit.update_best(OUT_DIR, STAGE, "nekaise_bench_dev", value)
        log_result(EXP_DIR, kind="train", stage=STAGE, method="sft-distill",
                   base_model=BASE_MODEL, init_from=INIT_FROM, dataset_id=dataset_id,
                   metric="nekaise_bench_dev", value=value)
        print(f"METRIC nekaise_bench_dev={value:.4f} stage={STAGE} init={src}")
    else:
        print(f"METRIC pending — run: python tools/eval_bench.py "
              f"--checkpoint experiments/ceiling-sub4b/outputs/{STAGE} --split dev")
    if logger:
        logger.finish(after=value)


if __name__ == "__main__":
    main()
