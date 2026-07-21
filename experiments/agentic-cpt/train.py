#!/usr/bin/env python3
"""train.py — HOW: pure agentic CPT (MUTABLE recipe).

Continued (next-token) pretraining of a small base on the curated corpus slice — no
teacher, no QA pairs, no chat template. The PHASE METRIC is the studio-owned
corpus_probes pack (absorption accuracy, dev split) via tools/eval_probes.py.
nekaise-bench is NOT consulted here: it is a milestone-only external referee since the
decoupling reform (see STATUS.md).

    NEKAISE_BASE_MODEL=unsloth/Qwen3.5-0.8B NEKAISE_STAGE=cpt_qwen08 \
        python experiments/agentic-cpt/train.py

The frozen probe split is NEVER run here — milestones only.
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
BASE_MODEL   = os.environ.get("NEKAISE_BASE_MODEL", "unsloth/Qwen3.5-0.8B")
INIT_FROM    = os.environ.get("NEKAISE_INIT_FROM")          # continue a checkpoint (repo-rel/abs)
STAGE        = os.environ.get("NEKAISE_STAGE", "cpt")
MAX_SEQ_LEN  = 2048
TIME_BUDGET_MIN = float(os.environ.get("NEKAISE_BUDGET_MIN", 45))
EVAL_AFTER   = os.environ.get("NEKAISE_EVAL_AFTER", "1") == "1"   # probe dev after training
FULL_PARAM   = os.environ.get("NEKAISE_FULL_PARAM", "0") == "1"   # sub-2B collapses full-param
LORA  = dict(r=32, lora_alpha=64, lora_dropout=0.0,
             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"])
TRAIN = dict(per_device_train_batch_size=2, gradient_accumulation_steps=8,
             num_train_epochs=float(os.environ.get("NEKAISE_EPOCHS", 1)),
             learning_rate=float(os.environ.get("NEKAISE_LR", 1e-4)),
             warmup_steps=20, logging_steps=5, seed=3407, lr_scheduler_type="cosine")
# ===============================================================================


def probe_dev(stage_dir: Path) -> float | None:
    """The phase metric via the shared tool (fresh subprocess = clean VRAM)."""
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools" / "eval_probes.py"),
         "--checkpoint", str(stage_dir), "--split", "dev", "--exp", EXP_DIR.name],
        capture_output=True, text=True)
    sys.stdout.write(proc.stdout[-2000:])
    m = re.search(r"METRIC corpus_probes_dev=([0-9.]+)", proc.stdout)
    return float(m.group(1)) if m else None


def main() -> None:
    d = datakit.latest_dir(EXP_DIR)
    if not d:
        raise FileNotFoundError("no dataset — run build_data.py first")
    rows = datakit.read_dir(d)
    dataset_id = datakit.provenance(d)["dataset_id"]
    src = str(REPO / INIT_FROM) if INIT_FROM and not Path(INIT_FROM).is_absolute() \
        else (INIT_FROM or BASE_MODEL)
    texts = [r["text"] for r in rows]
    print(f"[train] pure-CPT {src} on {dataset_id} ({len(texts)} docs, "
          f"~{sum(map(len, texts))//4/1e6:.1f}M tok) -> outputs/{STAGE}")

    model, tok = trainkit.load_model(src, max_seq_len=MAX_SEQ_LEN,
                                     full_finetuning=FULL_PARAM,
                                     lora=None if FULL_PARAM else LORA,
                                     seed=TRAIN["seed"])
    logger = RunLogger(EXP_DIR.name, model=src, pack="corpus_probes",
                       metric="corpus_probes_dev") if RunLogger else None
    callbacks = [trainkit.time_budget_callback(TIME_BUDGET_MIN)] \
        + ([trainer_callback(logger)] if logger else [])
    trainkit.run_cpt(model, tok, texts, max_seq_len=MAX_SEQ_LEN, train_args=TRAIN,
                     out_dir=OUT_DIR, callbacks=callbacks)

    dest = trainkit.save_checkpoint(model, tok, OUT_DIR / STAGE, provenance={
        "stage": STAGE, "method": "cpt", "base_model": BASE_MODEL,
        "init_from": INIT_FROM, "dataset_id": dataset_id, "n_docs": len(texts),
        "full_param": FULL_PARAM, "lora": None if FULL_PARAM else LORA, "train": TRAIN})
    print(f"[train] saved -> {dest}")

    del model  # free training VRAM before the eval subprocess spins up its own copy
    import torch
    torch.cuda.empty_cache()

    value = probe_dev(dest) if EVAL_AFTER else None
    if value is not None:
        trainkit.update_best(OUT_DIR, STAGE, "corpus_probes_dev", value)
        log_result(EXP_DIR, kind="train", stage=STAGE, method="cpt",
                   base_model=BASE_MODEL, init_from=INIT_FROM, dataset_id=dataset_id,
                   metric="corpus_probes_dev", value=value)
        print(f"METRIC corpus_probes_dev={value:.4f} stage={STAGE} init={src}")
    else:
        print(f"METRIC pending — run: python tools/eval_probes.py "
              f"--checkpoint experiments/{EXP_DIR.name}/outputs/{STAGE} --split dev")
    if logger:
        logger.finish(after=value)


if __name__ == "__main__":
    main()
