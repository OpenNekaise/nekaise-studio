#!/usr/bin/env python3
"""train.py — HOW: SFT-distill for the CEILING phase (MUTABLE recipe).

Loads a sub-4B base (or a corpus-CPT checkpoint via NEKAISE_INIT_FROM — repo-relative or
absolute path), LoRA-SFTs it on the corpus-QA distillation set (data/LATEST from
build_data.py), saves outputs/<stage>/, then scores it on the PHASE METRIC — nekaise-bench
dev split via tools/eval_bench.py — and prints the METRIC line the loop reads.

    NEKAISE_BASE_MODEL=unsloth/Qwen3.5-0.8B \
    NEKAISE_INIT_FROM=experiments/granite-4.1-3b-building/outputs/cpt_qwen08_chunk \
    NEKAISE_STAGE=distill_qwen08 python experiments/ceiling-sub4b/train.py

The frozen bench `test` split is NEVER run here — milestones only (see run-experiment skill).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
OUT_DIR = EXP_DIR / "outputs"
sys.path.insert(0, str(REPO / "lib"))
import datakit  # noqa: E402

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


def load_rows() -> list[dict]:
    d = datakit.latest_dir(EXP_DIR)
    if not d:
        raise FileNotFoundError("no dataset — run build_data.py first")
    print(f"[train] dataset {datakit.provenance(d)['dataset_id']} ({len(datakit.read_dir(d))} rows)")
    return datakit.read_dir(d)


def load_model():
    from unsloth import FastLanguageModel
    src = str(REPO / INIT_FROM) if INIT_FROM and not Path(INIT_FROM).is_absolute() else (INIT_FROM or BASE_MODEL)
    model, tok = FastLanguageModel.from_pretrained(
        model_name=src, max_seq_length=MAX_SEQ_LEN, load_in_4bit=False, dtype=None)
    if hasattr(tok, "tokenizer"):  # multimodal processor (e.g. Qwen3.5/VL) -> text tokenizer
        tok = tok.tokenizer
    # A checkpoint that already carries a LoRA adapter continues training it; a merged/base
    # model gets a fresh adapter (second get_peft_model on a peft model would error).
    if not (Path(src) / "adapter_config.json").exists():
        model = FastLanguageModel.get_peft_model(
            model, use_gradient_checkpointing="unsloth", random_state=TRAIN["seed"], **LORA)
    return model, tok, src


def time_budget_callback():
    from transformers import TrainerCallback

    class _TB(TrainerCallback):
        deadline = time.time() + TIME_BUDGET_MIN * 60
        def on_step_end(self, args, state, control, **kw):
            if time.time() > self.deadline:
                control.should_training_stop = True
            return control
    return _TB()


def bench_dev(stage_dir: Path) -> float | None:
    """The phase metric: nekaise-bench dev, via the shared tool (fresh subprocess = clean VRAM)."""
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools" / "eval_bench.py"),
         "--checkpoint", str(stage_dir), "--split", "dev"],
        capture_output=True, text=True)
    sys.stdout.write(proc.stdout[-2000:])
    m = re.search(r"nekaise_bench=([0-9.]+)", proc.stdout)
    return float(m.group(1)) if m else None


def main() -> None:
    rows = load_rows()
    model, tok, src = load_model()
    print(f"[train] SFT-distill {src} -> outputs/{STAGE} ({len(rows)} rows)")
    logger = RunLogger("ceiling-sub4b", model=src, pack="nekaise-bench",
                       metric="nekaise_bench_dev") if RunLogger else None
    callbacks = [time_budget_callback()] + ([trainer_callback(logger)] if logger else [])

    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    ds = Dataset.from_list([{"text": tok.apply_chat_template(r["messages"], tokenize=False)}
                            for r in rows])
    SFTTrainer(model=model, processing_class=tok, train_dataset=ds, callbacks=callbacks,
               args=SFTConfig(dataset_text_field="text", max_length=MAX_SEQ_LEN,
                              output_dir=str(OUT_DIR / "_trainer"), report_to="none", **TRAIN),
               ).train()

    dest = OUT_DIR / STAGE
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dest))
    tok.save_pretrained(str(dest))
    print(f"[train] saved -> {dest}")

    # free training VRAM before the eval subprocess spins up its own copy
    del model
    import torch
    torch.cuda.empty_cache()

    value = bench_dev(dest) if EVAL_AFTER else None
    if value is not None:
        best_path = OUT_DIR / "best.json"
        best = json.loads(best_path.read_text()) if best_path.exists() else None
        if best is None or value > best.get("value", float("-inf")):
            best_path.write_text(json.dumps({"stage": STAGE, "metric": "nekaise_bench_dev",
                                             "value": value, "path": f"outputs/{STAGE}"}, indent=2))
        print(f"METRIC nekaise_bench_dev={value:.4f} stage={STAGE} init={src}")
    else:
        print(f"METRIC pending — run: python tools/eval_bench.py "
              f"--checkpoint experiments/ceiling-sub4b/outputs/{STAGE} --split dev")
    if logger:
        logger.finish(after=value)


if __name__ == "__main__":
    main()
