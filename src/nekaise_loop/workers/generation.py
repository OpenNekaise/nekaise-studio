"""Owned, bounded single-checkpoint batch inference; no optimizer implementation.

The legacy model worker remains byte-identical for optimizer compatibility.
Failures return to the worker/orchestrator; this worker never silently changes
precision, batch limits or checkpoint identity to recover from a failure.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nekaise_loop.artifacts import digest
from nekaise_loop.serialization import student_prompt
from nekaise_loop.workers.model import audit_generation, emit


def plan_batches(lengths, groups, *, max_rows, max_tokens, max_new_tokens):
    """Partition declared independent groups without exceeding padded capacity."""
    if max_rows < 1 or max_tokens < 1 or max_new_tokens < 1:
        raise ValueError("Generation batch limits must be positive")
    if groups is None:
        groups = [list(lengths)] if lengths else []
    ids = [item for group in groups for item in group]
    if len(ids) != len(set(ids)) or set(ids) != set(lengths) or any(not g for g in groups):
        raise ValueError("Generation batch groups must partition the unique prompt IDs")
    batches = []
    for group in groups:
        batch, width = [], 0
        for item in group:
            length = lengths[item]
            if length < 1 or length + max_new_tokens > max_tokens:
                raise ValueError(f"Generation prompt {item!r} exceeds the batch token-position budget")
            next_width = max(width, length)
            if batch and (len(batch) == max_rows or (len(batch)+1)*(next_width+max_new_tokens) > max_tokens):
                batches.append(batch)
                batch, width = [], 0
            batch.append(item)
            width = max(width, length)
        if batch:
            batches.append(batch)
    return batches


def main():
    task, input_path = sys.argv[1:]
    if task != "generate":
        raise ValueError("The generation worker only accepts generate")
    data = json.loads(Path(input_path).read_text())
    config, rows = data["config"], data["rows"]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Generation requires unique prompt IDs")
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(data["checkpoint"], local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("The student tokenizer must have an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    prepared = {}
    for row in rows:
        try:
            rendered, special, serialization = student_prompt(tokenizer, row["prompt"],
                row.get("format", config.get("student_format", "raw_text")), max_tokens=config["max_seq_len"])
            encoded = tokenizer(rendered, add_special_tokens=special, return_tensors="pt",
                truncation=serialization["format"] == "raw_text", max_length=config["max_seq_len"])
            original = len(tokenizer.encode(rendered, add_special_tokens=special))
        except ValueError as exc:
            raise ValueError(f"Generation prompt row {row['id']!r}: {exc}") from exc
        prepared[row["id"]] = (encoded, original, {**serialization, "prompt_text": rendered})
    batches = plan_batches({key: x[0].input_ids.shape[1] for key, x in prepared.items()}, data.get("batch_groups"),
        max_rows=config.get("generation_batch_size", 4), max_tokens=config.get("generation_batch_tokens", 8192),
        max_new_tokens=config["max_new_tokens"])
    if not batches:
        emit("result", [])
        return
    emit("generation_plan", {"batches": batches, "prompt_lengths": {k: v[0].input_ids.shape[1] for k, v in prepared.items()},
        "max_rows": config.get("generation_batch_size", 4), "max_token_positions": config.get("generation_batch_tokens", 8192),
        "max_new_tokens": config["max_new_tokens"], "padding_side": "left"})
    model = AutoModelForCausalLM.from_pretrained(data["checkpoint"], local_files_only=True,
        trust_remote_code=False, dtype=torch.float32, attn_implementation="sdpa").to(device)
    model.eval()
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else list(eos)
    implementation = hashlib.sha256(Path(__file__).read_bytes()
        + Path(__file__).with_name("model.py").read_bytes()
        + Path(__file__).parents[1].joinpath("serialization.py").read_bytes()).hexdigest()
    results = {}
    for batch_index, ids in enumerate(batches):
        encodings = [prepared[key][0] for key in ids]
        encoded = tokenizer.pad([{k: v[0].tolist() for k, v in item.items()} for item in encodings],
                                padding=True, return_tensors="pt").to(device)
        width = encoded.input_ids.shape[1]
        prompt_ids = [item.input_ids[0].tolist() for item in encodings]
        context_hash = digest(prompt_ids)
        started = time.monotonic()
        with torch.inference_mode():
            outputs = model.generate(**encoded, max_new_tokens=config["max_new_tokens"], do_sample=False,
                num_beams=1, num_return_sequences=1, return_dict_in_generate=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=eos, use_cache=True)
        # Materialize all suffixes before stopping the shared batch wall clock.
        suffixes = outputs[:, width:].tolist()
        seconds = time.monotonic() - started
        for position, (key, suffix) in enumerate(zip(ids, suffixes)):
            # Finished rows are padded by generate while other rows continue.
            # Only the first actual terminator belongs to this observation.
            end = next((i+1 for i, token in enumerate(suffix) if token in eos_ids), len(suffix))
            token_ids = suffix[:end]
            original_encoding, original_length, serialization = prepared[key]
            one = original_encoding.to(device)
            output = torch.cat([one.input_ids, one.input_ids.new_tensor([token_ids])], dim=1)
            stop = "eos" if token_ids and token_ids[-1] in eos_ids else "max_new_tokens" if len(token_ids) >= config["max_new_tokens"] else "other"
            answer = {"id": key, "text": tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
                "tokens": len(token_ids), "seconds": round(seconds, 6), "timing_basis": "shared_batch_generation_wall_time",
                "generated_token_ids": token_ids, "raw_text": tokenizer.decode(token_ids, skip_special_tokens=False),
                "eos_token_ids": eos_ids, "stop_reason": stop, "prompt_tokens": len(prompt_ids[position]),
                "prompt_truncated": original_length > len(prompt_ids[position]),
                "effective_prompt": tokenizer.decode(prompt_ids[position], skip_special_tokens=False),
                "prompt_token_ids": prompt_ids[position], "prompt_serialization": serialization,
                "generation_audit": audit_generation(model, one, output),
                "generation_settings": {**model.generation_config.to_diff_dict(), "max_new_tokens": config["max_new_tokens"],
                    "do_sample": False, "num_beams": 1, "num_return_sequences": 1, "return_dict_in_generate": False,
                    "pad_token_id": tokenizer.pad_token_id, "eos_token_id": eos, "use_cache": True},
                "runtime": {"device": device, "dtype": str(model.dtype), "torch": torch.__version__,
                    "transformers": transformers.__version__, "attention": model.config._attn_implementation,
                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                    "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
                "generation_execution": {"worker": "batched_transformers_v1", "implementation_hash": implementation,
                    "batch_size": len(ids), "padded_prompt_tokens": width, "padding_side": "left",
                    "position_in_batch": position, "batch_prompt_hash": context_hash},
                "generation_batch": {"index": batch_index, "row_ids": ids, "size": len(ids),
                    "padded_prompt_tokens": width, "reserved_token_positions": len(ids)*(width+config["max_new_tokens"]),
                    "generation_seconds": round(seconds, 6)}}
            results[key] = answer
            emit("answer", answer)
    emit("result", [results[row["id"]] for row in rows])


if __name__ == "__main__":
    main()
