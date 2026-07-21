"""trainkit — the training scaffolding every recipe shares. FIXED plumbing.

Recipes (experiments/<name>/train.py) stay small and mutable: knobs + method choice. The
mechanics that must not drift between experiments live here:

  load_model(src, ...)        unsloth load; multimodal-tokenizer unwrap; LoRA attach rules
  run_sft(...)                chat-template render + TRL SFTTrainer
  time_budget_callback(min)   stop training at the wall-clock box
  save_checkpoint(...)        weights + tokenizer + meta.json PROVENANCE (base, init,
                              dataset, knobs, recipe git sha) — a checkpoint you can't
                              trace to its recipe is a checkpoint you can't trust
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_model(src: str, *, max_seq_len: int = 2048, load_in_4bit: bool = False,
               full_finetuning: bool = False, lora: dict | None = None, seed: int = 3407):
    """Load a base / checkpoint with unsloth and return (model, tok).

    - multimodal processors (Qwen3.5/VL) are unwrapped to their text tokenizer
    - if `lora` is given: a fresh adapter is attached UNLESS `src` already carries one
      (continuing an adapter checkpoint; a second get_peft_model would error)
    - full_finetuning=True trains all params (no adapter)
    """
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        model_name=src, max_seq_length=max_seq_len,
        load_in_4bit=(load_in_4bit and not full_finetuning),
        full_finetuning=full_finetuning, dtype=None)
    if hasattr(tok, "tokenizer"):
        tok = tok.tokenizer
    if lora and not full_finetuning and not (Path(src) / "adapter_config.json").exists():
        model = FastLanguageModel.get_peft_model(
            model, use_gradient_checkpointing="unsloth", random_state=seed, **lora)
    return model, tok


def time_budget_callback(minutes: float):
    """TrainerCallback that stops training when the time box expires."""
    from transformers import TrainerCallback

    class _TB(TrainerCallback):
        deadline = time.time() + minutes * 60

        def on_step_end(self, args, state, control, **kw):
            if time.time() > self.deadline:
                control.should_training_stop = True
            return control
    return _TB()


def run_sft(model, tok, rows: list[dict], *, max_seq_len: int, train_args: dict,
            out_dir: Path, callbacks: list | None = None) -> None:
    """SFT on chat rows [{"messages": [...]}] via TRL, rendered with the model's template."""
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    ds = Dataset.from_list(
        [{"text": tok.apply_chat_template(r["messages"], tokenize=False)} for r in rows])
    SFTTrainer(
        model=model, processing_class=tok, train_dataset=ds, callbacks=callbacks or [],
        args=SFTConfig(dataset_text_field="text", max_length=max_seq_len,
                       output_dir=str(Path(out_dir) / "_trainer"), report_to="none",
                       **train_args),
    ).train()


def run_cpt(model, tok, texts: list[str], *, max_seq_len: int, train_args: dict,
            out_dir: Path, callbacks: list | None = None) -> None:
    """Continued pretraining: next-token on raw corpus text (EOS-joined, packed)."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    ds = Dataset.from_list([{"text": t + tok.eos_token} for t in texts])
    SFTTrainer(
        model=model, processing_class=tok, train_dataset=ds, callbacks=callbacks or [],
        args=SFTConfig(dataset_text_field="text", max_length=max_seq_len, packing=True,
                       output_dir=str(Path(out_dir) / "_trainer"), report_to="none",
                       **train_args),
    ).train()


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def save_checkpoint(model, tok, stage_dir: Path, provenance: dict) -> Path:
    """Save weights + tokenizer + meta.json so the checkpoint is traceable to its recipe."""
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(stage_dir))
    tok.save_pretrained(str(stage_dir))
    (stage_dir / "meta.json").write_text(json.dumps(
        {"saved": time.time(), "recipe_git_sha": _git_sha(), **provenance}, indent=2))
    return stage_dir


def update_best(out_dir: Path, stage: str, metric: str, value: float) -> bool:
    """Track the experiment's best checkpoint (outputs/best.json). Returns True if new best."""
    best_path = Path(out_dir) / "best.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else None
    if best is None or value > best.get("value", float("-inf")):
        best_path.write_text(json.dumps({"stage": stage, "metric": metric,
                                         "value": round(value, 4),
                                         "path": f"outputs/{stage}"}, indent=2))
        return True
    return False
