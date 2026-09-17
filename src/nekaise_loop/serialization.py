"""Explicit single-turn student serialization; no model or tensor imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def checkpoint_eos_ids(checkpoint, tokenizer):
    path = Path(checkpoint) / "generation_config.json"
    eos = json.loads(path.read_text()).get("eos_token_id") if path.is_file() else None
    if eos is None:
        eos = tokenizer.eos_token_id
    return [eos] if isinstance(eos, int) else list(eos)


def chat_prompt(tokenizer, prompt):
    template = tokenizer.get_chat_template()
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    return rendered, {"format": "chat_template", "version": 1,
        "enable_thinking": False,
        "template_sha256": hashlib.sha256(template.encode()).hexdigest()}


def student_prompt(tokenizer, prompt, format="raw_text", *, max_tokens=None):
    if format == "raw_text":
        return prompt, True, {"format": "raw_text", "version": 1}
    if format != "chat_template":
        raise ValueError(f"Unknown student format: {format}")
    rendered, evidence = chat_prompt(tokenizer, prompt)
    length = len(tokenizer.encode(rendered, add_special_tokens=False))
    if max_tokens is not None and length > max_tokens:
        raise ValueError(f"Chat prompt has {length} tokens, exceeding max_seq_len={max_tokens}; truncation would change its native response boundary")
    # The native template owns BOS and role tokens. Never add them a second time.
    return rendered, False, evidence


def chat_training(row, tokenizer, eos_ids):
    """Teach the exact generation prefix, response and native assistant terminator.

    A completed-message template need not reproduce a no-thinking generation
    prefill. Use the actual generation prefix and derive only the closing suffix
    from the native template. Post-termination whitespace is not a target.
    """
    prompt, response = row["training_prompt"], row["training_response"]
    rendered, evidence = chat_prompt(tokenizer, prompt)
    expected = row.get("training_template_sha256")
    if expected and expected != evidence["template_sha256"]:
        raise ValueError("Historical chat template changed; re-author this lesson explicitly instead of reformatting replay")
    messages = [{"role": "user", "content": prompt}]
    header = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    complete = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": ""}],
        tokenize=False, add_generation_prompt=False)
    if not complete.startswith(header):
        raise ValueError("Chat template cannot expose a single-turn assistant closing suffix")
    closing = complete[len(header):]
    # Validate the actual response too. Explicit empty reasoning preserves literal
    # response content in templates that otherwise parse embedded thinking tags.
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant",
        "content": response, "reasoning_content": ""}], tokenize=False, add_generation_prompt=False)
    if full != header + response + closing:
        raise ValueError("Chat template rewrites the assistant response; use an explicit raw-text recipe for this content")
    closing_ids = tokenizer.encode(closing, add_special_tokens=False)
    end = next((i for i, token in enumerate(closing_ids) if token in eos_ids), None)
    if end is None:
        raise ValueError("Native assistant closing suffix has no effective generation EOS")
    closing_ids = closing_ids[:end + 1]
    prefix_ids = tokenizer.encode(rendered, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    ids = prefix_ids + response_ids + closing_ids
    text = rendered + response + tokenizer.decode(closing_ids, skip_special_tokens=False)
    return ids, {**row, "text": text, "serialization": {**evidence,
        "prompt_text": rendered, "prompt_token_ids": prefix_ids,
        "assistant_closing_text": closing, "assistant_closing_ids": closing_ids,
        "response_tokens": len(response_ids), "eos_token_ids": eos_ids}}
