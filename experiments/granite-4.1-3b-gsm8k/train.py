#!/usr/bin/env python3
"""train.py — the TRAIN recipe (MUTABLE: edit this to change HOW the model learns).

One of two files the autoresearch agent edits (the other is build_data.py — WHAT it
learns on). It loads a model, trains it with the selected METHOD, evaluates on the
pack's FIXED scorer, saves a checkpoint, and prints the `METRIC` line the loop reads.

    python experiments/granite-4.1-3b-gsm8k/train.py

Methods (all use the same pack referee, so results stay comparable):
  - sft   : supervised fine-tune on an SFT dataset (gold-rendered, distilled, or RFT).
  - dpo   : preference optimization on {prompt, chosen, rejected} pairs.
  - grpo  : RL with the pack's graded `reward()` as the verifiable reward (DeepSeek-style).

Datasets come from build_data.py (cached under data/<id>/). Stages hand off through
outputs/<stage>/, so a pipeline like  build_data → sft → grpo  is just three runs,
each changing one knob. NEVER edit packs/*/scorer.py (the referee).

Note: exact TRL/Unsloth kwargs drift between versions — adjust to the installed one.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
OUT_DIR = EXP_DIR / "outputs"
sys.path.insert(0, str(REPO / "lib"))

import datakit  # noqa: E402  (fixed plumbing)
from pack import load as load_pack  # noqa: E402  (fixed referee loader)

try:
    from runlog import RunLogger, trainer_callback  # noqa: E402  (optional dashboard telemetry)
except Exception:
    RunLogger = None

# ==================================== KNOBS ====================================
PACK            = "gsm8k"
METHOD          = "sft"        # "sft" | "dpo" | "grpo"
DATASET         = "render"     # "render" (SFT on gold) | "auto" (data/LATEST) | a dataset_id
INIT_FROM       = None         # None = BASE_MODEL; or "outputs/<stage>" to continue a pipeline
STAGE           = "sft"        # checkpoint is saved to outputs/<STAGE>/

BASE_MODEL      = "unsloth/granite-4.1-3b"
MAX_SEQ_LEN     = 2048
LOAD_IN_4BIT    = True
TIME_BUDGET_MIN = 10           # wall-clock training cap (fair comparison across runs)
EVAL_N          = 200          # test examples to score
MAX_NEW_TOKENS  = 512

LORA = dict(r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])

TRAIN = dict(per_device_train_batch_size=2, gradient_accumulation_steps=4,
             warmup_steps=5, max_steps=1000, learning_rate=2e-4,
             optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="linear",
             logging_steps=10, seed=3407)

# GRPO-only knobs (ignored by sft/dpo)
GRPO = dict(num_generations=8, max_prompt_length=512, max_completion_length=MAX_NEW_TOKENS,
            train_questions=500)

SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step, "
    "then give the final numeric answer on a new line as '#### <answer>'."
)
# ===============================================================================


# --------------------------------- helpers ------------------------------------
def resolve_dataset(pack) -> list[dict]:
    """Return SFT rows ({"messages": [...]}) for the configured DATASET."""
    if DATASET == "render":
        # Baseline: SFT directly on the pack's gold answers (no generation step).
        rows = []
        for ex in pack.load_split("train", None):
            rows.append({"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": ex["answer"]},
            ]})
        return rows
    d = datakit.latest_dir(EXP_DIR) if DATASET == "auto" else (EXP_DIR / "data" / DATASET)
    if not d or not (d / "data.jsonl").exists():
        raise FileNotFoundError(
            f"DATASET={DATASET!r} not found. Run build_data.py first, or use DATASET='render'.")
    print(f"[train] dataset: {datakit.provenance(d).get('dataset_id', d.name)} "
          f"({len(datakit.read_dir(d))} rows) from {d}")
    return datakit.read_dir(d)


def load_model(init_from: str | None):
    """Load base (or a prior stage) and attach a fresh LoRA adapter."""
    from unsloth import FastLanguageModel
    src = BASE_MODEL if init_from is None else str(EXP_DIR / init_from)
    model, tok = FastLanguageModel.from_pretrained(
        model_name=src, max_seq_length=MAX_SEQ_LEN, load_in_4bit=LOAD_IN_4BIT, dtype=None)
    model = FastLanguageModel.get_peft_model(
        model, use_gradient_checkpointing="unsloth", random_state=TRAIN["seed"], **LORA)
    return model, tok


def time_budget_callback():
    """A TrainerCallback that stops training after TIME_BUDGET_MIN (the time box)."""
    from transformers import TrainerCallback

    class _TB(TrainerCallback):
        deadline = time.time() + TIME_BUDGET_MIN * 60
        def on_step_end(self, args, state, control, **kw):
            if time.time() > self.deadline:
                control.should_training_stop = True
            return control
    return _TB()


def evaluate(model, tok, pack, n: int) -> float:
    """Greedy-decode the pack's test split and return scorer accuracy."""
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    test = pack.load_split("test", n)
    correct = 0
    for ex in test:
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": ex["question"]}],
            tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        pred = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        correct += pack.is_correct(pred, ex["answer"])
    return correct / len(test)


def save_checkpoint(model, tok, stage: str, metric: str, value: float) -> None:
    """Save the adapter to outputs/<stage>/ and update outputs/best.json if it wins."""
    dest = OUT_DIR / stage
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dest))
    tok.save_pretrained(str(dest))
    best_path = OUT_DIR / "best.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else None
    if best is None or value > best.get("value", float("-inf")):
        best_path.write_text(json.dumps(
            {"stage": stage, "metric": metric, "value": round(value, 4),
             "path": f"outputs/{stage}"}, indent=2))
    print(f"[train] saved checkpoint -> {dest}")


# --------------------------------- methods ------------------------------------
def run_sft(model, tok, rows, callbacks):
    """SFT on rows of {"messages": [...]} (covers render / distillation / RFT data)."""
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    ds = Dataset.from_list([
        {"text": tok.apply_chat_template(r["messages"], tokenize=False)} for r in rows])
    SFTTrainer(
        model=model, tokenizer=tok, train_dataset=ds, callbacks=callbacks,
        args=SFTConfig(dataset_text_field="text", max_seq_length=MAX_SEQ_LEN,
                       output_dir=str(OUT_DIR / "_trainer"), report_to="none", **TRAIN),
    ).train()


def run_dpo(model, tok, rows, callbacks):
    """Preference optimization on rows of {"prompt", "chosen", "rejected"}."""
    from datasets import Dataset
    from trl import DPOTrainer, DPOConfig

    ds = Dataset.from_list(rows)
    DPOTrainer(
        model=model, tokenizer=tok, train_dataset=ds, callbacks=callbacks,
        args=DPOConfig(max_length=MAX_SEQ_LEN, output_dir=str(OUT_DIR / "_trainer"),
                       report_to="none", **TRAIN),
    ).train()


def run_grpo(model, tok, pack, callbacks):
    """RL with the pack's graded reward() as a verifiable reward (DeepSeek-style)."""
    from datasets import Dataset
    from trl import GRPOTrainer, GRPOConfig

    # Prompts + gold; gold rides along as a dataset column GRPO passes to the reward fn.
    train = pack.load_split("train", GRPO["train_questions"])
    ds = Dataset.from_list([{
        "prompt": tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": ex["question"]}],
            tokenize=False, add_generation_prompt=True),
        "answer": ex["answer"],
    } for ex in train])

    def reward_fn(completions, answer, **kw):
        return [pack.reward(c, a) for c, a in zip(completions, answer)]

    GRPOTrainer(
        model=model, processing_class=tok, train_dataset=ds, reward_funcs=[reward_fn],
        callbacks=callbacks,
        args=GRPOConfig(
            num_generations=GRPO["num_generations"],
            max_prompt_length=GRPO["max_prompt_length"],
            max_completion_length=GRPO["max_completion_length"],
            output_dir=str(OUT_DIR / "_trainer"), report_to="none", **TRAIN),
    ).train()


# ----------------------------------- main -------------------------------------
def main() -> None:
    pack = load_pack(PACK)
    log = RunLogger(EXP_DIR.name, BASE_MODEL, PACK, f"{PACK}_acc") if RunLogger else None
    callbacks = [time_budget_callback()] + ([trainer_callback(log)] if log else [])

    model, tok = load_model(INIT_FROM)

    t0 = time.time()
    if METHOD == "sft":
        run_sft(model, tok, resolve_dataset(pack), callbacks)
    elif METHOD == "dpo":
        run_dpo(model, tok, resolve_dataset(pack), callbacks)
    elif METHOD == "grpo":
        run_grpo(model, tok, pack, callbacks)
    else:
        raise ValueError(f"unknown METHOD={METHOD!r} (use sft|dpo|grpo)")
    train_min = (time.time() - t0) / 60

    acc = evaluate(model, tok, pack, EVAL_N)
    save_checkpoint(model, tok, STAGE, f"{PACK}_acc", acc)
    if log:
        log.log_step(TRAIN["max_steps"], eval_acc=acc)
        log.finish(after=acc)
    print(f"METRIC {PACK}_acc={acc:.4f} n={EVAL_N} method={METHOD} "
          f"dataset={DATASET} stage={STAGE} train_min={train_min:.1f}")


if __name__ == "__main__":
    main()
