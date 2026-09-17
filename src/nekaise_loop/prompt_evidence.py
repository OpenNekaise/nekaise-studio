"""Read-only comparisons of recorded generation prefixes and frozen samples."""
from __future__ import annotations


def prompt_training_prefixes(lessons, dataset):
    """Describe each sample separately; never infer whole-row tokens from chunks.

    These observations neither select training material nor establish correctness.
    Missing generation metadata or omitted samples leave token agreement unknown.
    """
    rows = {r["id"]: r for r in dataset["rows"] if r["stream"] in {"cpt", "sft"}}
    samples = {}
    for index, sample in enumerate(dataset["samples"]):
        if sample["stream"] == "teacher":
            samples.setdefault(sample["row_id"], []).append((index, sample["input_ids"]))
    observations = []
    for lesson in lessons:
        prompt = lesson["student_prompt"]
        ids = lesson.get("prompt_token_ids")
        row = rows.get(lesson["id"])
        if row and row.get("training_tokenization") == "chat_response":
            # New generations retain the exact rendered string; decoding IDs can
            # normalize whitespace. Preserve the fallback for historical evidence.
            serialization = lesson.get("prompt_serialization", {})
            prompt = serialization.get("prompt_text", lesson.get("effective_prompt", prompt))
        candidates = samples.get(lesson["id"], [])
        evidence = {
            "lesson_id": lesson["id"],
            "training_tokenization": row.get("training_tokenization", "full_text") if row else None,
            "text_prefix_matches": row["text"].startswith(prompt) if row else None,
            "prompt_text_tail": prompt[-80:],
            "prompt_tokens": len(ids) if ids else None,
            "prompt_truncated": lesson.get("prompt_truncated"),
            "status": "prompt_tokens_unavailable" if not ids else "compared" if candidates else "no_frozen_samples",
            "sample_comparisons": [],
        }
        if ids:
            for index, tokens in candidates:
                common = 0
                for left, right in zip(ids, tokens):
                    if left != right:
                        break
                    common += 1
                start = max(0, common - 4)
                evidence["sample_comparisons"].append({
                    "sample_index": index,
                    "sample_tokens": len(tokens),
                    "common_prefix_tokens": common,
                    "prompt_is_sample_prefix": common == len(ids),
                    "sample_ends_before_prompt": common == len(tokens) < len(ids),
                    "boundary_start_index": start,
                    "prompt_boundary_ids": ids[start:common + 5],
                    "sample_boundary_ids": tokens[start:common + 5],
                })
        observations.append(evidence)
    return {
        "basis": "recorded_prompt_token_ids_vs_frozen_samples",
        "limitations": "Each comparison starts at one frozen sample's beginning. Samples may be chunked, shortened or repeated; no whole-row token sequence is reconstructed. Prompt truncation applies to generation. Prepared samples may not be consumed in a diagnostic round. Agreement is not a learning result or an acceptance gate; mismatches may be intentional.",
        "lessons": observations,
    }
