"""Opt-in ML worker: independent padded microbatches, unchanged causal updates."""
from __future__ import annotations

from collections import defaultdict
import json
import hashlib
import math
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nekaise_loop.training import update_batches, recipe_hash
from nekaise_loop.training_runtime import runtime_hash, optimizer_transition
from nekaise_loop.workers.model import emit, file_hash


def microbatches(rows, size):
    if size < 1:
        raise ValueError("Microbatch size must be positive")
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def collate(rows, pad_token_id, device):
    import torch
    lengths = [len(row["input_ids"]) for row in rows]
    if not lengths or min(lengths) < 2:
        raise ValueError("Every sequence must contain a causal target")
    width = max(lengths)
    ids = torch.tensor([r["input_ids"] + [pad_token_id] * (width - n)
                        for r, n in zip(rows, lengths)], device=device)
    mask = torch.arange(width, device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]
    labels = ids.masked_fill(~mask, -100)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}, sum(n - 1 for n in lengths)


def backward_update(model, rows, pad_token_id, microbatch_size, device):
    import torch
    total = sum(len(r["input_ids"]) - 1 for r in rows)
    value = torch.zeros((), device=device)
    for part in microbatches(rows, microbatch_size):
        inputs, count = collate(part, pad_token_id, device)
        labels = inputs.pop("labels")
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(**inputs).logits
        # Explicit FP32 sum makes normalization independent of Transformers'
        # batch-loss defaults. Current training is full-sequence CPT+SFT; only
        # newly introduced pads are ignored. No existing labels are overwritten.
        summed = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        weighted = summed / total
        weighted.backward()
        value.add_(weighted.detach())
        del logits, inputs, labels, weighted, summed
    return value


def load_training(data, *, legacy_optimizer_loading=False):
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    config = data["config"]
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not data.get("allow_cpu", False):
        raise RuntimeError("CUDA is unavailable; no silent CPU training fallback")
    tokenizer = AutoTokenizer.from_pretrained(data["checkpoint"], local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer requires EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(data["checkpoint"], local_files_only=True,
        trust_remote_code=False, dtype=torch.float32, attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    if config.get("training_activation_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    parameters = list(model.named_parameters())
    layout = [(n, list(p.shape), str(p.dtype)) for n, p in parameters]
    layout_hash = hashlib.sha256(json.dumps(layout).encode()).hexdigest()
    progress = {"global_step": 0, "global_tokens": 0, "optimizer_origin": "initialized",
                "optimizer_transition": None, "parameter_layout_sha256": layout_hash}
    state_path = Path(data["checkpoint"]) / "training_state.pt"
    if config["inherit_optimizer"] and state_path.exists():
        # Non-capturable Adam's step scalars belong on CPU. Loading them onto CUDA
        # forces per-parameter scalar synchronizations inside optimizer.step().
        # The diagnostic serial reference explicitly retains the old placement.
        state = torch.load(state_path, map_location=device if legacy_optimizer_loading else "cpu", weights_only=True)
        manifest = json.loads((state_path.parent / "checkpoint.json").read_text())
        for key in ("training_code", "recipe_hash", "global_step", "global_tokens"):
            if state.get(key) != manifest.get(key):
                raise ValueError(f"Optimizer state and checkpoint manifest disagree: {key}")
        transition = optimizer_transition(manifest, config)
        if manifest.get("versions") != {"torch": torch.__version__, "transformers": transformers.__version__}:
            raise ValueError("Adam inheritance requires its validated torch/transformers versions")
        if transition["policy"] == "identical_runtime" and manifest.get("parameter_layout_sha256") != layout_hash:
            raise ValueError("Adam parameter-name ordering differs from the saved runtime")
        groups = state["optimizer"]["param_groups"]
        if len(groups) != 1 or len(groups[0]["params"]) != len(parameters):
            raise ValueError("Adam parameter grouping differs from the validated trainer")
        for (_, parameter), key in zip(parameters, groups[0]["params"]):
            old = state["optimizer"]["state"].get(key, {})
            if set(old) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("Adam state is incomplete or uses an unsupported optimizer variant")
            for name in ("exp_avg", "exp_avg_sq"):
                if old[name].shape != parameter.shape or old[name].dtype != parameter.dtype:
                    raise ValueError("Adam moment shape/dtype does not match FP32 parameters")
        optimizer.load_state_dict(state["optimizer"])
        # load_state_dict maps parameters by index. Same exact predecessor code,
        # model checkpoint and library versions establish order; prove every
        # loaded step and moment was retained, and record the current name layout.
        for (_, parameter), key in zip(parameters, groups[0]["params"]):
            for name, old in state["optimizer"]["state"][key].items():
                if not torch.equal(optimizer.state[parameter][name].to(old.device), old):
                    raise ValueError("Adam state changed while loading")
        transition["parameter_layout_sha256"] = layout_hash
        transition["loaded_state_equal"] = True
        transition["adam_step_device"] = str(next(iter(optimizer.state.values()))["step"].device)
        progress.update(global_step=state["global_step"], global_tokens=state["global_tokens"],
                        optimizer_origin="inherited", optimizer_transition=transition)
        del state
    elif config["inherit_optimizer"] and (Path(data["checkpoint"]) / "checkpoint.json").exists():
        raise ValueError("Recorded checkpoint has no resumable Adam state; refusing silent initialization")
    elif not config["inherit_optimizer"]:
        progress["optimizer_origin"] = "explicit_reset"
    return model, tokenizer, optimizer, device, progress


def step_update(model, optimizer, rows, config, pad_token_id, device, global_tokens):
    import torch
    count = sum(len(r["input_ids"]) - 1 for r in rows)
    lr = config["learning_rate"] * min(1.0, (global_tokens + count) / max(1, config["warmup_tokens"]))
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    loss = backward_update(model, rows, pad_token_id, config["training_microbatch_size"], device)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # One scalar transfer per update, before any parameter mutation. A nonfinite
    # microbatch propagates into the total; never apply an invalid update.
    value, grad_norm = torch.stack((loss, norm)).tolist()
    if not math.isfinite(value) or not math.isfinite(grad_norm):
        raise ValueError("Non-finite training loss or gradients")
    optimizer.step()
    return {"loss": value, "grad_norm": grad_norm, "learning_rate": lr, "tokens": count}


def train(data):
    import torch
    config, dataset = data["config"], data["dataset"]
    output = Path(data["output"])
    temporary = output.with_name(output.name + ".partial")
    if output.exists() or temporary.exists():
        raise FileExistsError("Use a fresh stage attempt for checkpoint output")
    samples = dataset["samples"]
    if not samples or any(len(r["input_ids"]) < 2 for r in samples):
        raise ValueError("Dataset contains no causal training targets")
    model, tokenizer, optimizer, device, progress = load_training(data)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started, count, losses = time.monotonic(), 0, []
    streams = defaultdict(int)
    total_steps = config["train_steps"] or config["train_epochs"] * math.ceil(dataset["ledger"]["total_tokens"] / config["tokens_per_update"])
    for step, rows in enumerate(update_batches(samples, config["tokens_per_update"], config["train_epochs"], config["seed"], config["train_steps"]), 1):
        result = step_update(model, optimizer, rows, config, tokenizer.pad_token_id, device, progress["global_tokens"])
        if device == "cuda":
            torch.cuda.synchronize()
        count += result["tokens"]
        progress["global_tokens"] += result["tokens"]
        progress["global_step"] += 1
        for row in rows:
            streams[row["stream"]] += len(row["input_ids"]) - 1
        elapsed = time.monotonic() - started
        losses.append(result["loss"])
        emit("metric", {**result, "step": step, "total_steps": total_steps,
            "global_step": progress["global_step"], "global_tokens": progress["global_tokens"],
            "tokens": count, "stream_tokens": dict(streams), "elapsed_seconds": elapsed,
            "tokens_per_second": count / max(elapsed, .001),
            "gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0,
            "gpu_reserved_gb": torch.cuda.max_memory_reserved() / 1e9 if device == "cuda" else 0,
            "training_microbatch_size": config["training_microbatch_size"]})
    if not losses:
        raise ValueError("Training worker received no positive updates")
    temporary.mkdir(parents=True)
    model.config.use_cache = True
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    code, recipe = runtime_hash(config), recipe_hash(config)
    torch.save({"optimizer": optimizer.state_dict(), "global_step": progress["global_step"],
        "global_tokens": progress["global_tokens"], "recipe_hash": recipe, "training_code": code}, temporary / "training_state.pt")
    import transformers
    manifest = {"parent": data["checkpoint"], "dataset_hash": data["dataset_hash"], "config": config,
        "steps": len(losses), "tokens": count, **progress, "training_code": code, "recipe_hash": recipe,
        "stream_tokens": dict(streams), "mean_loss": sum(losses) / len(losses),
        "files": {p.name: file_hash(p) for p in temporary.iterdir() if p.is_file()},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__}, "device": device}
    (temporary / "checkpoint.json").write_text(json.dumps(manifest, indent=2))
    os.replace(temporary, output)
    emit("result", {"checkpoint": str(output), "manifest": manifest})


if __name__ == "__main__":
    task, path = sys.argv[1:]
    if task != "train":
        raise ValueError("Batched training worker only accepts train")
    train(json.loads(Path(path).read_text()))
