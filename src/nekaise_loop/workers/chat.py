"""One-shot, CPU-only interactive inference. Never imported by the API."""
from __future__ import annotations

import ctypes
import fcntl
import json
import os
import signal
import sys
import subprocess
import time


_EVENT_OUTPUT = None

def emit(kind, **values):
    print(json.dumps({'type': kind, **values}, ensure_ascii=False), file=_EVENT_OUTPUT or sys.stdout, flush=True)


def render_conversation(tokenizer, messages, max_tokens):
    rendered = tokenizer.apply_chat_template(
        [{**m, **({'reasoning_content': ''} if m['role'] == 'assistant' else {})} for m in messages],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    if len(ids) > max_tokens:
        raise ValueError(f'Conversation has {len(ids)} tokens; limit is {max_tokens}. Start a new chat or shorten the message.')
    return ids


def main():
    global _EVENT_OUTPUT
    _EVENT_OUTPUT = os.fdopen(os.dup(1), 'w')
    os.dup2(2, 1)  # Third-party stdout must not enter the NDJSON event channel.
    # Set before importing any ML package. FP32 preserves the checkpoint's weights.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    parent = int(os.environ['NEKAISE_CHAT_PARENT_PID'])
    if ctypes.CDLL(None).prctl(1, signal.SIGTERM) != 0:
        raise RuntimeError('Cannot establish chat process ownership')
    if os.getppid() != parent:
        return
    os.nice(10)
    subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    signal.alarm(180)
    payload = json.loads(sys.stdin.readline())
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
    from nekaise_loop.artifacts import verify_checkpoint

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    snapshot = payload['snapshot']
    verify_checkpoint(snapshot, require_optimizer=False)
    checkpoint = snapshot['checkpoint']
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    ids = render_conversation(tokenizer, payload['messages'], payload['max_prompt_tokens'])
    loaded_at = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True,
        trust_remote_code=False, dtype=torch.float32, attn_implementation='sdpa').to('cpu').eval()
    # File-backed CPU tensors are copied before releasing the reader lock.
    # This prevents future page faults from depending on retired checkpoint bytes.
    with torch.no_grad():
        for param in model.parameters():
            param.data = param.data.clone()
        for buffer in model.buffers():
            buffer.data = buffer.data.clone()
    fcntl.flock(payload['checkpoint_lock_fd'], fcntl.LOCK_UN)
    fcntl.flock(payload['source_lock_fd'], fcntl.LOCK_UN)
    emit('ready', prompt_tokens=len(ids), load_seconds=time.monotonic()-loaded_at)

    class Streamer(TextStreamer):
        def on_finalized_text(self, text, stream_end=False):
            if text:
                emit('delta', text=text)

    tokens = torch.tensor([ids], dtype=torch.long, device='cpu')
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(input_ids=tokens, attention_mask=torch.ones_like(tokens),
            max_new_tokens=payload['max_new_tokens'], do_sample=False,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            use_cache=True, streamer=Streamer(tokenizer, skip_prompt=True, skip_special_tokens=True))
    answer = output[0, len(ids):].tolist()
    text = tokenizer.decode(answer, skip_special_tokens=True)
    if not text.strip():
        raise ValueError('The model returned an empty reply. Try rephrasing the message.')
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos or [tokenizer.eos_token_id]
    emit('done', text=text, tokens=len(answer), seconds=time.monotonic()-started,
         stop_reason='eos' if answer[-1] in eos else 'length', device='cpu')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        emit('error', message=str(exc)[:400])
        sys.exit(1)
