"""Read-only-checkpoint performance experiment; updated weights are never saved."""
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nekaise_loop.training import update_batches
from nekaise_loop.workers.batched_training import load_training, backward_update, step_update, microbatches
from nekaise_loop.workers.model import emit


def serial_backward(model, rows, device):
    import torch
    count = sum(len(r["input_ids"]) - 1 for r in rows)
    value = 0.
    for row in rows:
        tokens = torch.tensor([row["input_ids"]], device=device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            loss = model(input_ids=tokens, labels=tokens).loss
        sample = float(loss.detach().item())
        if not math.isfinite(sample):
            raise ValueError("Nonfinite legacy loss")
        weight = (len(row["input_ids"]) - 1) / count
        (loss * weight).backward()
        value += sample * weight
    return value


def batch_statistics(rows, size):
    groups = list(microbatches(rows, size))
    return {"forward_backward_calls": len(groups),
            "padded_input_tokens": sum(len(g) * max(len(r["input_ids"]) for r in g) for g in groups),
            "unpadded_input_tokens": sum(len(r["input_ids"]) for r in rows),
            "valid_targets": sum(len(r["input_ids"]) - 1 for r in rows),
            "max_padded_tokens": max((len(g) * max(len(r["input_ids"]) for r in g) for g in groups), default=0),
            "max_physical_rows": max(map(len, groups), default=0)}


def equivalence(model, tokenizer, optimizer, device, rows, *, sizes=(0, 1, 2, 4),
                checkpointings=(True, False), reference_size=0, reference_checkpointing=None):
    import torch
    if reference_checkpointing is False:
        model.gradient_checkpointing_disable()
    elif reference_checkpointing is True:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer.zero_grad(set_to_none=True)
    reference_loss = (float(backward_update(model, rows, tokenizer.pad_token_id, reference_size, device).item())
                      if reference_size else serial_backward(model, rows, device))
    reference = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
    findings = []
    for size in sizes:
        for checkpointing in checkpointings:
            if checkpointing:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                model.gradient_checkpointing_disable()
            optimizer.zero_grad(set_to_none=True)
            loss = (serial_backward(model, rows, device) if size == 0 else
                    float(backward_update(model, rows, tokenizer.pad_token_id, size, device).item()))
            squared_diff, squared_ref, max_abs = 0., 0., 0.
            for name, parameter in model.named_parameters():
                if name not in reference:
                    if parameter.grad is not None:
                        raise ValueError("Gradient parameter set changed")
                    continue
                current, expected = parameter.grad.detach().cpu(), reference[name]
                difference = current - expected
                squared_diff += float(difference.square().sum())
                squared_ref += float(expected.square().sum())
                max_abs = max(max_abs, float(difference.abs().max()))
            relative = math.sqrt(squared_diff / max(squared_ref, 1e-30))
            finding = {"microbatch_size": size, "activation_checkpointing": checkpointing,
                "reference_repeat": size == reference_size,
                "loss": loss, "reference_loss": reference_loss, "loss_absolute_difference": abs(loss-reference_loss),
                "gradient_relative_l2": relative, "gradient_max_absolute_difference": max_abs,
                "passed": math.isfinite(relative) and relative < .03 and abs(loss-reference_loss) < .01}
            findings.append(finding)
            emit("observation", finding)
    return {"kind": "numerical_equivalence",
            "reference": f"batched_v1 physical batch {reference_size}" if reference_size else "reimplementation of preserved serial BF16 loop",
            "reference_microbatch_size": reference_size, "valid_targets": sum(len(r["input_ids"]) - 1 for r in rows),
            "row_lengths": [len(r["input_ids"]) for r in rows],
            "rows": len(rows), "loss_absolute_tolerance": .01, "gradient_relative_l2_tolerance": .03,
            "results": findings,
            "passed": all(r["passed"] for r in findings)}


def main():
    import torch
    data = json.loads(Path(sys.argv[1]).read_text())
    config, case = data["config"], data["profile_case"]
    model, tokenizer, optimizer, device, progress = load_training(data, legacy_optimizer_loading=case.get("serial", False))
    samples = data["dataset"]["samples"]
    if case.get("equivalence"):
        # Deliberately uneven lengths and a short tail, from real frozen targets.
        headroom = case.get("headroom", False)
        lengths = (127, 61, 256, 17, 511, 93, 201, 33, 87, 459, 147, 307) if headroom else (127, 61, 256, 17)
        rows = [{**r, "input_ids": r["input_ids"][:length]} for r, length in zip(samples, lengths)]
        if headroom and len(rows) != len(lengths):
            raise ValueError("Headroom equivalence requires at least twelve real frozen rows")
        options = {"sizes": (4, 6, 8), "checkpointings": (False,), "reference_size": 4,
                   "reference_checkpointing": False} if headroom else {}
        result = equivalence(model, tokenizer, optimizer, device, rows, **options)
        updates = update_batches(samples, config["tokens_per_update"], 1, config["seed"])
        # Exercise many real short rows, not only four full-length sequences.
        full = max(updates, key=lambda batch: (sum(len(r["input_ids"])-1 for r in batch), len(batch))) if headroom else next(updates)
        result["full_update"] = equivalence(model, tokenizer, optimizer, device, full, **options)
        result["optimizer_transition"] = progress["optimizer_transition"]
        result["passed"] = result["passed"] and result["full_update"]["passed"]
        emit("result", result)
        return
    warmup, measured = case["warmup_updates"], case["measured_updates"]
    batches = list(update_batches(samples, config["tokens_per_update"], 1, config["seed"], warmup + measured))
    count, times, start = 0, [], None
    shapes = {"forward_backward_calls": 0, "padded_input_tokens": 0, "unpadded_input_tokens": 0,
              "valid_targets": 0, "max_physical_rows": 0, "max_padded_tokens": 0}
    measured_digest = hashlib.sha256()
    update_target_counts = []
    # Count shapes outside the timed window. The digest proves identical measured
    # row order and update boundaries across physical batch sizes and repeats.
    for rows in batches[warmup:]:
        measured_digest.update(json.dumps([(r["input_ids"], r["stream"]) for r in rows], separators=(",", ":")).encode())
        stats = batch_statistics(rows, 1 if case.get("serial") else config["training_microbatch_size"])
        update_target_counts.append(stats["valid_targets"])
        for key, value in stats.items():
            shapes[key] = max(shapes[key], value) if key.startswith("max_") else shapes[key] + value
    for index, rows in enumerate(batches):
        if index == warmup:
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
        step_start = time.monotonic()
        n = sum(len(r["input_ids"]) - 1 for r in rows)
        if case.get("serial"):
            lr = config["learning_rate"] * min(1., (progress["global_tokens"] + n) / max(1, config["warmup_tokens"]))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss = serial_backward(model, rows, device)
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.).item())
            if not math.isfinite(norm):
                raise ValueError("Nonfinite legacy gradient")
            optimizer.step()
        else:
            result = step_update(model, optimizer, rows, config, tokenizer.pad_token_id, device, progress["global_tokens"])
            loss = result["loss"]
        torch.cuda.synchronize()
        progress["global_tokens"] += n
        if index >= warmup:
            count += n
            times.append(time.monotonic() - step_start)
    elapsed = time.monotonic() - start
    if len(times) != measured:
        raise ValueError("Profiling did not complete its declared measurement updates")
    if shapes["valid_targets"] != count:
        raise ValueError("Measured target accounting disagrees with training updates")
    emit("result", {"kind": "throughput_diagnostic", "case": case, "targets": count,
        "seconds": elapsed, "tokens_per_second": count / elapsed, "step_seconds": times,
        "batch_statistics": shapes, "tokens_per_update": config["tokens_per_update"],
        "measured_input_sha256": measured_digest.hexdigest(), "update_target_counts": update_target_counts,
        "final_loss": loss, "peak_allocated_gb": torch.cuda.max_memory_allocated()/1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved()/1e9,
        "gpu_total_gb": torch.cuda.get_device_properties(device).total_memory / 1e9,
        "optimizer_transition": progress["optimizer_transition"],
        "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
        "official_training_exposure": 0, "checkpoint_saved": False})


if __name__ == "__main__":
    main()
