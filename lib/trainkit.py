"""trainkit — the training scaffolding every recipe shares. FIXED plumbing.

Stage entry points stay small and fixed. The mechanics that must not drift between
experiments live here:

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

    - multimodal processors are unwrapped to their text tokenizer
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


def run_cpt(model, tok, data_paths: list[Path], *, max_seq_len: int, train_args: dict,
            out_dir: Path, cache_dir: Path | None = None,
            callbacks: list | None = None) -> None:
    """Continued pretraining from memory-mapped JSONL (EOS-joined, packed).

    The source corpus is never materialized as a Python list. Hugging Face datasets
    converts each immutable JSONL artifact to an Arrow cache once, then maps it from
    disk across runs and seeds.
    """
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    paths = [str(Path(path)) for path in data_paths]
    if not paths:
        raise ValueError("run_cpt requires at least one dataset path")
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        "json", data_files=paths, split="train", keep_in_memory=False,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    ).select_columns(["text"])

    if not tok.eos_token:
        raise ValueError("CPT tokenizer has no eos_token")
    workers = max(1, int(train_args.pop("dataset_num_proc", 1)))
    SFTTrainer(
        model=model, processing_class=tok, train_dataset=ds, callbacks=callbacks or [],
        args=SFTConfig(dataset_text_field="text", max_length=max_seq_len, packing=True,
                       dataset_num_proc=workers,
                       output_dir=str(Path(out_dir) / "_trainer"), report_to="none",
                       **train_args),
    ).train()


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def save_checkpoint(model, tok, *, store, run_id: str, provenance: dict) -> dict:
    """Atomically commit an immutable, content-addressed checkpoint artifact."""
    temporary = store.artifact_temp("checkpoint", run_id)
    try:
        model.save_pretrained(str(temporary))
        tok.save_pretrained(str(temporary))
        (temporary / "meta.json").write_text(json.dumps(
            {"saved": time.time(), "recipe_git_sha": _git_sha(),
             "run_id": run_id, **provenance}, indent=2))
        return store.commit_artifact(
            temporary, kind="checkpoint", run_id=run_id, role="checkpoint",
            metadata=provenance,
        )
    except BaseException:
        # Leave the uniquely named temp directory for forensic inspection. GC may remove
        # it later; never risk deleting a committed artifact after a partial failure.
        raise
