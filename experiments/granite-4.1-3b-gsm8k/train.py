#!/usr/bin/env python3
"""Editable fine-tune baseline: Granite-4.1-3B on the GSM8K task pack, via Unsloth.

This is the ONE file the autoresearch agent edits (see skills/run-experiment.md).
It fine-tunes, then evaluates on the pack's *fixed* scorer, and prints a METRIC line
the loop reads. Tune anything below; do NOT edit packs/gsm8k/scorer.py.

    python experiments/granite-4.1-3b-gsm8k/train.py

Note: exact TRL/Unsloth kwargs drift between versions — adjust to the installed one.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# --- make the task pack's fixed scorer importable -------------------------------
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "packs" / "gsm8k"))
import scorer  # noqa: E402  (packs/gsm8k/scorer.py — the fitness function; do not edit)

# optional: stream loss + metric to the local dashboard (dashboard/server.py)
sys.path.insert(0, str(REPO / "dashboard"))
try:
    from runlog import RunLogger, trainer_callback  # noqa: E402
except Exception:
    RunLogger = None

# ================================ KNOBS =========================================
BASE_MODEL      = "unsloth/granite-4.1-3b"
MAX_SEQ_LEN     = 2048
LOAD_IN_4BIT    = True
TIME_BUDGET_MIN = 10          # wall-clock training cap (fair comparison across runs)
EVAL_N          = 200         # GSM8K test examples to score
MAX_NEW_TOKENS  = 512

LORA = dict(r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])

TRAIN = dict(per_device_train_batch_size=2, gradient_accumulation_steps=4,
             warmup_steps=5, max_steps=1000, learning_rate=2e-4,
             optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="linear",
             logging_steps=10, seed=3407)

SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step, "
    "then give the final numeric answer on a new line as '#### <answer>'."
)
# ================================================================================

from unsloth import FastLanguageModel  # noqa: E402
from datasets import load_dataset       # noqa: E402
from trl import SFTTrainer, SFTConfig   # noqa: E402
from transformers import TrainerCallback # noqa: E402


class TimeBudget(TrainerCallback):
    """Stop training after TIME_BUDGET_MIN minutes — the autoresearch time box."""
    def __init__(self, minutes): self.deadline = time.time() + minutes * 60
    def on_step_end(self, args, state, control, **kw):
        if time.time() > self.deadline:
            control.should_training_stop = True
        return control


def main() -> None:
    log = RunLogger("granite-4.1-3b-gsm8k", BASE_MODEL, "gsm8k", "gsm8k_acc") if RunLogger else None

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=LOAD_IN_4BIT, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, use_gradient_checkpointing="unsloth", random_state=TRAIN["seed"], **LORA)

    # --- data: render GSM8K train into chat text ---
    train_ds = load_dataset("openai/gsm8k", "main", split="train")

    def render(ex):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": ex["answer"]}]
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}

    train_ds = train_ds.map(render, remove_columns=train_ds.column_names)

    # --- train (time-boxed) ---
    t0 = time.time()
    SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=train_ds,
        callbacks=[TimeBudget(TIME_BUDGET_MIN)] + ([trainer_callback(log)] if log else []),
        args=SFTConfig(dataset_text_field="text", max_seq_length=MAX_SEQ_LEN,
                       output_dir="outputs", report_to="none", **TRAIN),
    ).train()
    train_min = (time.time() - t0) / 60

    # --- evaluate on the pack's fixed scorer ---
    FastLanguageModel.for_inference(model)
    test = scorer.load_test(EVAL_N)
    correct = 0
    for ex in test:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": ex["question"]}],
            tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        pred = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        correct += scorer.is_correct(pred, ex["answer"])

    acc = correct / len(test)
    if log:
        log.finish(after=acc)
    print(f"METRIC gsm8k_acc={acc:.4f} n={len(test)} train_min={train_min:.1f}")


if __name__ == "__main__":
    main()
