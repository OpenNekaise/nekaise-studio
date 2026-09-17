"""Weight-preserving scoring subprocess; launched only by the owning loop worker."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def score_sample(model, sample, eos_ids, device, *, autocast=False):
    import torch
    tokens = torch.tensor([sample["input_ids"]], device=device)
    if tokens.shape[1] < 2:
        raise ValueError("Scoring sample has no causal targets")
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast):
        # Preserve the complete training input shape and disable the KV cache.
        logits = model(input_ids=tokens, use_cache=False).logits[0, :-1].float()
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("Non-finite frozen-sample scoring logits")
    targets = tokens[0, 1:]
    log_probs = logits.log_softmax(-1)
    nll = -log_probs.gather(1, targets[:, None]).squeeze(1)
    eos_probs = log_probs[:, eos_ids].exp().sum(-1)
    boundary = sample.get("prompt_tokens")
    if boundary is not None and not 1 <= boundary <= tokens.shape[1]:
        raise ValueError("Invalid prompt boundary in scoring sample")
    parts = {name: {"tokens": 0, "nll_sum": 0.0} for name in ("prompt", "continuation", "unassigned")}
    observations = []
    for offset, target in enumerate(targets.tolist()):
        position = offset + 1
        region = "unassigned" if boundary is None else "prompt" if position < boundary else "continuation"
        parts[region]["tokens"] += 1
        parts[region]["nll_sum"] += float(nll[offset])
        observations.append({"position": position, "target_id": target, "region": region,
                             "nll": float(nll[offset]), "target_probability": float((-nll[offset]).exp()),
                             "eos_probability": float(eos_probs[offset]),
                             "argmax_id": int(logits[offset].argmax())})
    for part in parts.values():
        part["mean_nll"] = part["nll_sum"] / part["tokens"] if part["tokens"] else None
    return {"sample_index": sample["sample_index"], "row_id": sample["row_id"], "stream": sample["stream"],
            "input_ids": sample["input_ids"], "prompt_tokens": boundary,
            "tokens": len(targets), "mean_nll": float(nll.mean()), "parts": parts,
            "first_continuation": next((p for p in observations if p["region"] == "continuation"), None),
            "token_scores": observations}


def main():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nekaise_loop.workers.model import emit

    task, input_path = sys.argv[1:]
    if task != "score":
        raise ValueError("Unknown scoring task")
    data = json.loads(Path(input_path).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(data["checkpoint"], local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(data["checkpoint"], local_files_only=True,
        trust_remote_code=False, dtype=torch.float32, attn_implementation="sdpa").to(device)
    model.eval()
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Scoring requires an EOS identity")
    eos_ids = sorted(set([eos] if isinstance(eos, int) else eos))
    started = time.monotonic()
    variants = {}
    for precision in (["fp32", "bf16_autocast"] if device == "cuda" else ["fp32"]):
        variants[precision] = [score_sample(model, s, eos_ids, device, autocast=precision == "bf16_autocast") for s in data["samples"]]
    emit("result", {"method": "frozen_sample_teacher_forcing", "checkpoint": data["checkpoint"],
        "dataset_hash": data["dataset_hash"], "eos_token_ids": eos_ids, "variants": variants,
        "runtime": {"device": device, "parameter_dtype": str(model.dtype), "mode": "eval", "use_cache": False,
                    "attention": model.config._attn_implementation, "torch": torch.__version__,
                    "transformers": transformers.__version__, "cuda": torch.version.cuda,
                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                    "seconds": round(time.monotonic()-started, 3)},
        "limitations": "Teacher-forced likelihood, not generated answers or a learning gate. Eval mode does not replay stochastic training/dropout. Loss regions require exact recorded prompt alignment; unaligned or chunked samples stay unassigned. EOS probability includes effective generation terminators; targets remain the exact frozen IDs. No optimizer or weights were updated."})


if __name__ == "__main__":
    main()
