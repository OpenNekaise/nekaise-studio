"""Pure token planning. ML packages belong only in the model subprocess."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


def training_code_hash():
    return hashlib.sha256((Path(__file__).parent / "workers/model.py").read_bytes() + Path(__file__).read_bytes()).hexdigest()


def group_for(stream):
    return "teacher" if stream in {"cpt", "sft"} else stream


def prepare_dataset(rows, tokenizer, config):
    groups = {name: [] for name in ("teacher", "corpus", "replay")}
    for row in rows:
        group = group_for(row["stream"])
        ids = tokenizer.encode(row["text"], add_special_tokens=True)
        if not ids or ids[-1] != tokenizer.eos_token_id:
            ids.append(tokenizer.eos_token_id)
        # Overlap one conditioning token within an example; never across examples.
        for start in range(0, max(0, len(ids)-1), config["max_seq_len"]-1):
            chunk = ids[start:start+config["max_seq_len"]]
            if len(chunk) >= 2:
                groups[group].append({"row_id": row["id"], "stream": group, "input_ids": chunk})
    available = {name: sum(len(r["input_ids"])-1 for r in part) for name, part in groups.items()}
    requested = config["token_mix"]
    enabled = {name: share for name, share in requested.items() if available[name] and share > 0}
    total_share = sum(enabled.values())
    shares = {name: enabled.get(name, 0)/total_share if total_share else 0 for name in groups}
    # Each teacher-selected row is used once per pass; intentional duplicate rows
    # remain teaching decisions. Other streams fill their declared token ratio.
    anchor = next((name for name in groups if name in enabled), None)
    targets = {name: round(available[anchor] * share / shares[anchor]) if anchor else 0 for name, share in shares.items()}
    rng = random.Random(config["seed"])
    samples, ledger = [], {}
    for name, part in groups.items():
        part = part[:]
        rng.shuffle(part)
        left, index = targets[name], 0
        while left > 0:
            row = part[index % len(part)]
            ids = row["input_ids"][:left+1]
            samples.append({**row, "input_ids": ids})
            left -= len(ids)-1
            index += 1
        ledger[name] = {"available_tokens": available[name], "prepared_tokens": targets[name],
                        "requested_share": requested[name], "effective_share": shares[name],
                        "repeated_tokens": max(0, targets[name]-available[name])}
    rng.shuffle(samples)
    return {"rows": rows, "samples": samples, "ledger": {"basis": "causal_loss_tokens_including_eos", "anchor_stream": anchor, "streams": ledger, "total_tokens": sum(targets.values()), "missing_streams": [k for k in groups if requested[k] and not available[k]]}}


def update_batches(samples, tokens_per_update, epochs, seed, step_limit=0):
    """Exact token accumulation; short final updates flush without hidden replay."""
    if not samples:
        raise ValueError("No prepared training samples")
    rng, epoch, steps = random.Random(seed), 0, 0
    while epoch < epochs or step_limit > 0:
        ordered = samples[:]
        rng.shuffle(ordered)
        batch, count = [], 0
        for row in ordered:
            ids, offset = row["input_ids"], 0
            while offset < len(ids)-1:
                take = min(tokens_per_update-count, len(ids)-1-offset)
                batch.append({**row, "input_ids": ids[offset:offset+take+1]})
                count += take
                offset += take
                if count == tokens_per_update:
                    yield batch
                    steps += 1
                    if step_limit and steps >= step_limit:
                        return
                    batch, count = [], 0
        if batch:
            yield batch
            steps += 1
            if step_limit and steps >= step_limit:
                return
        epoch += 1


def optimizer_recipe(config):
    keys = ("learning_rate", "warmup_tokens", "tokens_per_update", "max_seq_len")
    return {"version": 1, "optimizer": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8,
            "weight_decay": 0.01, "clip_grad_norm": 1.0,
            **{k: config[k] for k in keys}}


def recipe_hash(config):
    return hashlib.sha256(json.dumps(optimizer_recipe(config), sort_keys=True).encode()).hexdigest()
