"""Preserve all generated targets and a whole contiguous prefix of corpus spans.

This changes dataset composition, not token serialization, loss or Adam semantics.
"""
from collections import Counter
import random

from .training import prepare_dataset, group_for


def inspect_targets(dataset, remediation_cap):
    rows = {r["id"]: r for r in dataset["rows"]}
    if len(rows) != len(dataset["rows"]):
        raise ValueError("Progressive row IDs must be unique")
    tracks, authors, units, teacher_gpc, author_gpc = Counter(), Counter(), set(), 0, 0
    counts = Counter()
    for sample in dataset["samples"]:
        row = rows[sample["row_id"]]
        n = len(sample["input_ids"]) - 1
        track = row.get("learning_track")
        if track not in {"corpus", "gpc", "remediation"} or n <= 0:
            raise ValueError("Prepared target has no valid learning track")
        tracks[track] += n
        counts[row["id"]] += n
        author = row.get("material_origin", {}).get("author_id")
        if author and track in {"corpus", "gpc"}:
            authors[author] += n
        if track == "gpc":
            units.add(row.get("curriculum_unit_id", ""))
            if author:
                author_gpc += n
            elif row["stream"] in {"cpt", "sft"}:
                teacher_gpc += n
    total = sum(tracks.values())
    if not tracks["corpus"] or not tracks["gpc"] or "" in units:
        raise ValueError("Both corpus and assigned GPC forward targets are required")
    if tracks["remediation"] > total * remediation_cap + 1e-8:
        raise ValueError(f"Remediation exceeds the operator cap: {tracks['remediation']}/{total} > {remediation_cap}; revise the teaching package, never crop trusted batches")
    return {"by_track": dict(tracks), "by_row": dict(counts), "forward_authors": dict(authors),
            "gpc_units": sorted(units), "teacher_gpc": teacher_gpc, "author_gpc": author_gpc,
            "total": total, "remediation_share": tracks["remediation"] / total}


def prepare_progressive(rows, tokenizer, config, eos_ids=None):
    if config["train_steps"]:
        raise ValueError("Progressive traversal cannot use a partial step override")
    if not config["train_epochs"]:
        return prepare_dataset(rows, tokenizer, config, eos_ids)
    policy = config["curriculum_loop"]
    prepared, samples, counts = [], [], {}
    raw = [r for r in rows if r.get("curriculum_span")]
    web = [r for r in rows if r.get("web_span")]
    generated = [r for r in rows if not r.get("curriculum_span") and not r.get("web_span")]

    def serialize(row):
        stream = group_for(row["stream"])
        one_hot = {k: float(k == stream) for k in ("teacher", "corpus", "replay")}
        result = prepare_dataset([row], tokenizer, {**config, "token_mix": one_hot}, eos_ids)
        n = result["ledger"]["total_tokens"]
        if not n:
            raise ValueError("A progressive row has no causal targets")
        return result, n

    def append(result, n):
        prepared.extend(result["rows"])
        samples.extend(result["samples"])
        counts[result["rows"][0]["id"]] = n

    gpc, domain = 0, 0
    for row in generated:
        result, n = serialize(row)
        append(result, n)
        gpc += n if row.get("learning_track") == "gpc" else 0
        domain += n if row.get("learning_track") == "corpus" else 0
    wanted_web = policy.get("web_target_tokens", 0)
    web_consumed, web_ids = 0, []
    for row in web:
        if web_consumed >= wanted_web:
            break
        if not policy.get("web_training"):
            raise ValueError("Direct web training requires the operator's explicit policy")
        result, n = serialize(row)
        append(result, n)
        web_consumed += n
        gpc += n
        web_ids.append(row["id"])
    share = policy["forward_corpus_share"]
    wanted_raw = policy.get("raw_target_tokens") or max(1, round(gpc * share / (1-share)) - domain)
    consumed, raw_ids = 0, []
    for row in raw:
        result, n = serialize(row)
        append(result, n)
        consumed += n
        raw_ids.append(row["id"])
        if consumed >= wanted_raw:
            break
    random.Random(config["seed"]).shuffle(samples)
    by_stream = Counter()
    for sample in samples:
        by_stream[sample["stream"]] += len(sample["input_ids"])-1
    total = sum(by_stream.values())
    dataset = {"rows": prepared, "samples": samples, "ledger": {
        "basis": "causal_loss_tokens_including_eos", "anchor_stream": "whole_rows",
        "streams": {k: {"available_tokens": by_stream[k], "prepared_tokens": by_stream[k],
            "requested_share": config["token_mix"][k], "effective_share": by_stream[k]/total if total else 0,
            "repeated_tokens": 0} for k in ("teacher", "corpus", "replay")},
        "total_tokens": total, "missing_streams": [],
        "composition_policy": "progressive_whole_rows; legacy token_mix does not crop or repeat streams"}}
    targets = inspect_targets(dataset, policy["remediation_cap"])
    if targets["by_row"] != counts:
        raise ValueError("Preparation lost full-row target coverage")
    dataset["progressive_preparation"] = {"corpus_row_ids": raw_ids, "targets": targets,
        "web_row_ids": web_ids, "web_targets_requested": wanted_web,
        "web_targets_prepared": web_consumed, "web_supply_shortfall": web_consumed < wanted_web,
        "requested_forward_corpus_share": share,
        "actual_forward_corpus_share": targets["by_track"]["corpus"]/(targets["by_track"]["corpus"]+gpc),
        "raw_targets_requested": wanted_raw, "raw_targets_prepared": consumed,
        "raw_allocation_basis": "teacher_explicit_target" if policy.get("raw_target_tokens") else "forward_share",
        "corpus_window_exhausted": consumed < wanted_raw,
        "whole_row_targets": counts, "basis": "All generated rows and whole raw spans, once per pass"}
    return dataset
