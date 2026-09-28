"""Read-only-checkpoint performance experiment; updated weights are never saved."""
import gc
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nekaise_loop.training import update_batches
from nekaise_loop.workers.batched_training import load_training, backward_update, step_update
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


def equivalence(model, tokenizer, optimizer, device, rows):
    import torch
    optimizer.zero_grad(set_to_none=True)
    reference_loss = serial_backward(model, rows, device)
    reference = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
    findings = []
    for size in (0, 1, 2, 4):
        for checkpointing in (True, False):
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
                "reference_repeat": size == 0,
                "loss": loss, "reference_loss": reference_loss, "loss_absolute_difference": abs(loss-reference_loss),
                "gradient_relative_l2": relative, "gradient_max_absolute_difference": max_abs,
                "passed": math.isfinite(relative) and relative < .03 and abs(loss-reference_loss) < .01}
            findings.append(finding)
            emit("observation", finding)
    return {"kind": "numerical_equivalence", "reference": "reimplementation of preserved serial BF16 loop", "results": findings,
            "passed": all(r["passed"] for r in findings)}


def main():
    import torch
    data = json.loads(Path(sys.argv[1]).read_text())
    config, case = data["config"], data["profile_case"]
    model, tokenizer, optimizer, device, progress = load_training(data, legacy_optimizer_loading=case.get("serial", False))
    samples = data["dataset"]["samples"]
    if case.get("equivalence"):
        # Deliberately uneven lengths and a short tail, from real frozen targets.
        rows = [{**r, "input_ids": r["input_ids"][:length]} for r, length in zip(samples[:4], (127, 61, 256, 17))]
        result = equivalence(model, tokenizer, optimizer, device, rows)
        full = next(update_batches(samples, config["tokens_per_update"], 1, config["seed"]))
        result["full_update"] = equivalence(model, tokenizer, optimizer, device, full)
        result["passed"] = result["passed"] and result["full_update"]["passed"]
        emit("result", result)
        return
    warmup, measured = case["warmup_updates"], case["measured_updates"]
    batches = update_batches(samples, config["tokens_per_update"], 1, config["seed"], warmup + measured)
    count, times, start = 0, [], None
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
    emit("result", {"kind": "throughput_diagnostic", "case": case, "targets": count,
        "seconds": elapsed, "tokens_per_second": count / elapsed, "step_seconds": times,
        "final_loss": loss, "peak_allocated_gb": torch.cuda.max_memory_allocated()/1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved()/1e9,
        "optimizer_transition": progress["optimizer_transition"],
        "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
        "official_training_exposure": 0, "checkpoint_saved": False})


if __name__ == "__main__":
    main()
