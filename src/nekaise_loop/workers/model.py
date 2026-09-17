"""ML subprocess entrypoint. Deliberately independent of FastAPI and Pydantic.

Run with a Python environment containing torch + transformers. Output protocol:
LOOP {"type": "metric" | "answer" | "result", "data": ...}
Training retries at stage boundaries; completed rounds carry optimizer state forward.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nekaise_loop.training import prepare_dataset, update_batches, recipe_hash, training_code_hash
from nekaise_loop.serialization import checkpoint_eos_ids, student_prompt


def emit(kind, data):
    print("LOOP " + json.dumps({"type": kind, "data": data}, ensure_ascii=False, allow_nan=False), flush=True)


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit_generation(model, encoded, output):
    """Observe raw next-token predictions without reusing generation's KV cache.

    This is teacher forcing on the observed response, not a second free-running
    answer or a correctness gate. Logit processors and numerical ties can explain
    mismatches; retain margins and generation settings for operational review.
    """
    import torch

    prompt_length = encoded.input_ids.shape[1]
    generated = output[0, prompt_length:]
    if not len(generated):
        return {"method": "uncached_teacher_forcing", "tokens_checked": 0}
    inputs = {"input_ids": output[:, :-1], "attention_mask": torch.cat(
        [encoded.attention_mask, encoded.attention_mask.new_ones((1, len(generated)-1))], dim=1)}
    # Preserve tokenizer-provided segment IDs where the model uses them.
    if "token_type_ids" in encoded:
        inputs["token_type_ids"] = torch.cat([encoded.token_type_ids,
            encoded.token_type_ids[:, -1:].expand(-1, len(generated)-1)], dim=1)
    started = time.monotonic()
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits[0, prompt_length-1:].float()
        finite = bool(torch.isfinite(logits).all().item())
        if not finite:
            return {"method": "uncached_teacher_forcing", "tokens_checked": len(generated),
                    "finite_logits": False, "seconds": round(time.monotonic()-started, 3)}
        values, ids = logits.topk(min(5, logits.shape[-1]), dim=-1)
        # topk does not promise argmax's first-index ordering for tied logits.
        # Use the same tie rule as greedy generation and retain zero-gap ties.
        argmax_ids = logits.argmax(dim=-1)
        mismatch = argmax_ids != generated
        positions = mismatch.nonzero().flatten().tolist()
        observations = [{"position": i, "generated_token_id": int(generated[i]),
                         "raw_argmax_token_id": int(argmax_ids[i]),
                         "logit_gap": float(values[i, 0]-logits[i, generated[i]])}
                        for i in positions]
        first_probs = logits[0].softmax(dim=-1)
        first_top = [{"token_id": int(token), "probability": float(first_probs[token])}
                     for token in ids[0]]
    return {"method": "uncached_teacher_forcing", "tokens_checked": len(generated),
            "finite_logits": True, "raw_argmax_token_ids": argmax_ids.tolist(),
            "mismatches": observations, "first_token_top": first_top,
            "seconds": round(time.monotonic()-started, 3)}


def main():
    task, input_path = sys.argv[1:]
    data = json.loads(Path(input_path).read_text())
    config = data["config"]
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if task == "train" and device != "cuda" and not data.get("allow_cpu", False):
        raise RuntimeError("CUDA is unavailable. Live training requires a GPU; no silent CPU fallback.")
    tokenizer = AutoTokenizer.from_pretrained(data["checkpoint"], local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("The student tokenizer must have an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if task == "prepare":
        emit("result", prepare_dataset(data["rows"], tokenizer, config, checkpoint_eos_ids(data["checkpoint"], tokenizer)))
        return
    if task == "generate":
        # Validate every boundary before loading weights or emitting any answers.
        # Keep the actual CPU encodings so generation uses the preflighted inputs.
        prepared_prompts = []
        for row in data["rows"]:
            try:
                rendered, add_special, serialization = student_prompt(tokenizer, row["prompt"], row.get("format", config.get("student_format", "raw_text")), max_tokens=config["max_seq_len"])
                encoded = tokenizer(rendered, add_special_tokens=add_special, return_tensors="pt", truncation=serialization["format"] == "raw_text", max_length=config["max_seq_len"])
                original_length = len(tokenizer.encode(rendered, add_special_tokens=add_special))
            except ValueError as exc:
                raise ValueError(f"Generation prompt row {row['id']!r}: {exc}") from exc
            prepared_prompts.append((row, encoded, original_length,
                                     {**serialization, "prompt_text": rendered}))
    # Preserve checkpoint precision in inference as well as master parameters.
    # Live BF16 diagnostics showed tied EOS logits and cache/forward divergence;
    # use FP32 for generation and its audit so weight rounding is not a confound.
    # Training still uses BF16 autocast with FP32 parameters and Adam states.
    model = AutoModelForCausalLM.from_pretrained(data["checkpoint"], local_files_only=True, trust_remote_code=False, dtype=torch.float32, attn_implementation="sdpa").to(device)
    if task == "generate":
        model.eval()
        # Checkpoints may define multiple response terminators. The tokenizer's
        # single EOS is only a fallback, not an override of generation settings.
        eos = model.generation_config.eos_token_id
        if eos is None:
            eos = tokenizer.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos)
        answers = []
        for row, encoded, original_length, serialization in prepared_prompts:
            encoded = encoded.to(device)
            started = time.monotonic()
            with torch.inference_mode():
                output = model.generate(**encoded, max_new_tokens=config["max_new_tokens"], do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=eos, use_cache=True)
            generated = output[0, encoded.input_ids.shape[1]:]
            answer = {"id": row["id"], "text": tokenizer.decode(generated, skip_special_tokens=True).strip(), "tokens": len(generated), "seconds": round(time.monotonic()-started, 3)}
            token_ids = generated.tolist()
            stop_reason = "eos" if token_ids and token_ids[-1] in eos_ids else "max_new_tokens" if len(token_ids) >= config["max_new_tokens"] else "other"
            answer.update({"generated_token_ids": token_ids, "raw_text": tokenizer.decode(generated, skip_special_tokens=False), "eos_token_ids": eos_ids, "stop_reason": stop_reason})
            answer.update({"prompt_tokens": int(encoded.input_ids.shape[1]), "prompt_truncated": original_length > encoded.input_ids.shape[1], "effective_prompt": tokenizer.decode(encoded.input_ids[0], skip_special_tokens=False)})
            answer["prompt_token_ids"] = encoded.input_ids[0].tolist()
            answer["prompt_serialization"] = serialization
            answer["generation_audit"] = audit_generation(model, encoded, output)
            answer["generation_settings"] = {**model.generation_config.to_diff_dict(),
                "max_new_tokens": config["max_new_tokens"], "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id, "eos_token_id": eos, "use_cache": True}
            import transformers
            answer["runtime"] = {"device": device, "dtype": str(model.dtype),
                "torch": torch.__version__, "transformers": transformers.__version__,
                "attention": model.config._attn_implementation}
            answers.append(answer)
            emit("answer", answer)
        emit("result", answers)
        return
    if task != "train":
        raise ValueError(f"Unknown task {task}")
    dataset = data["dataset"]
    samples = dataset["samples"]
    if not samples or any(len(row["input_ids"]) < 2 for row in samples):
        raise ValueError("Dataset contains no causal training targets")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    global_step, global_tokens = 0, 0
    optimizer_origin = "initialized"
    state_path = Path(data["checkpoint"]) / "training_state.pt"
    recipe = recipe_hash(config)
    training_code = training_code_hash()
    if state_path.exists() and config["inherit_optimizer"]:
        state = torch.load(state_path, map_location=device, weights_only=True)
        if state["recipe_hash"] != recipe or state["training_code"] != training_code:
            raise ValueError("Training recipe/code changed. Start a continuation with inherit_optimizer=false to explicitly reset its optimizer.")
        optimizer.load_state_dict(state["optimizer"])
        global_step, global_tokens = state["global_step"], state["global_tokens"]
        optimizer_origin = "inherited"
        del state
    elif not config["inherit_optimizer"]:
        optimizer_origin = "explicit_reset"
    started, token_count, losses = time.monotonic(), 0, []
    stream_tokens = defaultdict(int)
    total_steps = config["train_steps"] or config["train_epochs"] * math.ceil(dataset["ledger"]["total_tokens"] / config["tokens_per_update"])
    for step, batch in enumerate(update_batches(samples, config["tokens_per_update"], config["train_epochs"], config["seed"], config["train_steps"]), 1):
        batch_tokens = sum(len(row["input_ids"])-1 for row in batch)
        lr = config["learning_rate"] * min(1.0, (global_tokens+batch_tokens)/max(1, config["warmup_tokens"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        value = 0.0
        for row in batch:
            tokens = torch.tensor([row["input_ids"]], device=device)
            n = tokens.numel()-1
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(input_ids=tokens, labels=tokens).loss
            sample_loss = float(loss.detach().item())
            if not math.isfinite(sample_loss):
                raise ValueError("Non-finite training loss")
            (loss * (n/batch_tokens)).backward()
            value += sample_loss * (n/batch_tokens)
            stream_tokens[row["stream"]] += n
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        if not math.isfinite(grad_norm):
            raise ValueError("Non-finite gradients")
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        token_count += batch_tokens
        global_tokens += batch_tokens
        global_step += 1
        elapsed = time.monotonic()-started
        metric = {"step": step, "total_steps": total_steps, "global_step": global_step, "global_tokens": global_tokens, "loss": value, "learning_rate": lr, "grad_norm": grad_norm, "tokens": token_count, "stream_tokens": dict(stream_tokens), "tokens_per_second": token_count/max(elapsed, 0.001), "elapsed_seconds": elapsed, "gpu_memory_gb": torch.cuda.max_memory_allocated()/1e9 if device == "cuda" else 0}
        losses.append(value)
        emit("metric", metric)
    output = Path(data["output"])
    temporary = output.with_name(output.name + ".partial")
    if output.exists() or temporary.exists():
        raise FileExistsError("Checkpoint directory already exists; use a fresh stage attempt")
    temporary.mkdir(parents=True)
    model.config.use_cache = True
    # Retain FP32 master weights across short rounds and for inference.
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    torch.save({"optimizer": optimizer.state_dict(), "global_step": global_step, "global_tokens": global_tokens, "recipe_hash": recipe, "training_code": training_code}, temporary / "training_state.pt")
    files = {p.name: file_hash(p) for p in temporary.iterdir() if p.is_file()}
    import transformers
    manifest = {"parent": data["checkpoint"], "dataset_hash": data["dataset_hash"], "config": config, "steps": len(losses), "tokens": token_count, "global_step": global_step, "global_tokens": global_tokens, "optimizer_origin": optimizer_origin, "training_code": training_code, "recipe_hash": recipe, "stream_tokens": dict(stream_tokens), "mean_loss": sum(losses)/len(losses), "files": files, "versions": {"torch": torch.__version__, "transformers": transformers.__version__}, "device": device}
    (temporary / "checkpoint.json").write_text(json.dumps(manifest, indent=2))
    os.replace(temporary, output)
    emit("result", {"checkpoint": str(output), "manifest": manifest})


if __name__ == "__main__":
    main()
