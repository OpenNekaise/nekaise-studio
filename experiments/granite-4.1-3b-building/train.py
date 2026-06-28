#!/usr/bin/env python3
"""train.py — TRAIN recipe for the building pack (MUTABLE: edit to change HOW the model learns).

SFT granite-4.1-3b on the senior-building-engineer Q&A distilled from Opus 4.8 (built by
build_data.py), then evaluate on the FIXED, verifiable ontology scorer over the HELD-OUT
building — i.e. does learning to reason like a building engineer on the training buildings
generalize to an unseen one? Saves a checkpoint and prints the METRIC line the loop reads.

    set -a; source /home/zengp/Code/KebAgent/.env; set +a   # only needed if a method calls a teacher
    python experiments/granite-4.1-3b-building/train.py

Methods: sft (default) | dpo | grpo. Stages hand off via outputs/<stage>/. The eval is the
fixed pack — NEVER edit packs/building/scorer.py.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
OUT_DIR = EXP_DIR / "outputs"
sys.path.insert(0, str(REPO / "lib"))
sys.path.insert(0, str(REPO / "dashboard"))
sys.path.insert(0, str(REPO / "packs" / "building"))

import datakit  # noqa: E402
from pack import load as load_pack  # noqa: E402

try:
    from runlog import RunLogger, trainer_callback  # noqa: E402
except Exception:
    RunLogger = None

# ==================================== KNOBS ====================================
PACK            = "building"
METHOD          = "grpo"       # "sft" | "dpo" | "grpo"
DATASET         = "auto"       # "auto" = data/LATEST (the distilled Q&A) | a dataset_id
# WINNING 2-STAGE RECIPE (building_judge 0.540, +0.18 over base): first SFT (METHOD="sft",
# INIT_FROM=None, STAGE="sft") on the grounded v3 demos -> outputs/sft; then this GRPO stage.
INIT_FROM       = "outputs/sft"  # GRPO continues run-C's grounded SFT adapter
STAGE           = "grpo"       # checkpoint saved to outputs/<STAGE>/  (this is best.json's winner)

BASE_MODEL      = "unsloth/granite-4.1-3b"
OPEN_BOOK       = True    # eval: put the held-out building's ontology in the prompt (as nekaise-edge serves it)
MAX_SEQ_LEN     = 16384   # wide enough to hold the held-out ontology (~10k tok) + question for open-book eval
LOAD_IN_4BIT    = True
TIME_BUDGET_MIN = 40           # sweet-spot search to ~80 steps
EVAL_N          = 8            # synthetic building_acc is only a sanity check (real metric = eval_judge); keep it cheap
MAX_NEW_TOKENS  = 320

LORA = dict(r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])

TRAIN = dict(per_device_train_batch_size=1, gradient_accumulation_steps=8,
             warmup_steps=5, max_steps=40, learning_rate=2e-4,
             optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="linear",
             logging_steps=5, seed=3407)  # 40 best (run 4); 70 (run 5) overfit type/count and forgot base conns

GRPO = dict(num_generations=6, max_prompt_length=2432, max_completion_length=400,
            lr=8e-6, max_steps=60, grad_accum=1, beta=0.02, temperature=1.0,
            slice_chars=8000, save_steps=20)  # ~60 steps is the sweet spot; MORE overfits the 3 training buildings

# Matches the student persona the distilled data was written for.
SYSTEM_PROMPT = (
    "You are a knowledgeable building engineer assistant. Answer the question accurately and "
    "specifically, grounding your answer in the building's equipment, sensors, topology, and "
    "semantic model (Brick / ASHRAE 223P). Name the relevant entities and explain your reasoning."
)

# --- env overrides: select a stage without editing the committed knobs, e.g.
#     NEKAISE_METHOD=cpt NEKAISE_STAGE=cpt NEKAISE_INIT_FROM=none python train.py
import os as _os  # noqa: E402
METHOD = _os.environ.get("NEKAISE_METHOD", METHOD)
STAGE = _os.environ.get("NEKAISE_STAGE", STAGE)
if "NEKAISE_INIT_FROM" in _os.environ:
    _v = _os.environ["NEKAISE_INIT_FROM"]
    INIT_FROM = None if _v.lower() in ("", "none") else _v

# Continued pretraining (next-token) on the cleaned HVAC corpus (build_cpt_data.py).
# LoRA on the 7 linear projections only -- light domain-adaptive pretraining over ~2.8M tokens.
# (Training embed_tokens/lm_head with a lower LR is the heavier Unsloth CPT config; add later if
# perplexity barely moves.) Followed by the existing SFT stage to restore instruction-following.
CPT = dict(seq_len=2048, packing=True, epochs=3, warmup_steps=10,
           per_device_train_batch_size=2, grad_accum=8,
           weight_decay=0.01, logging_steps=1, save_steps=100000,
           # SPEED: load at a CPT-sized context (not MAX_SEQ_LEN=16384) so Unsloth doesn't reserve
           # huge buffers and offload gradients -- that offloading was the ~5 min/step tax.
           load_seq_len=2048, load_in_4bit=False,
           # FULL-PARAMETER toggle. False = LoRA on the 7 linears (cheap, 124MB adapter).
           # True  = train all 3.43B params (bf16 + 8-bit Adam + grad-checkpointing ~ 25GB / 47GB).
           full_finetuning=True, lr=5e-5, lr_full=1e-5)
# ===============================================================================


def resolve_dataset(pack) -> list[dict]:
    if DATASET == "render":
        rows = []
        for ex in pack.load_split("train", None):
            rows.append({"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": ex["answer"]},
            ]})
        return rows
    d = datakit.latest_dir(EXP_DIR) if DATASET == "auto" else (EXP_DIR / "data" / DATASET)
    if not d or not (d / "data.jsonl").exists():
        raise FileNotFoundError(
            f"DATASET={DATASET!r} not found. Run build_data.py first (it distills the Q&A).")
    print(f"[train] dataset: {datakit.provenance(d).get('dataset_id', d.name)} "
          f"({len(datakit.read_dir(d))} rows) from {d}")
    return datakit.read_dir(d)


USE_VLLM = False               # vLLM not installed here (torch 2.10); GRPO uses HF generate (slower)
GPU_UTIL = 0.55


def load_model(init_from, seq_len=MAX_SEQ_LEN, four_bit=LOAD_IN_4BIT, full_ft=False):
    from pathlib import Path as _P
    from unsloth import FastLanguageModel
    src = BASE_MODEL if init_from is None else str(EXP_DIR / init_from)
    vllm_kw = (dict(fast_inference=True, max_lora_rank=LORA["r"], gpu_memory_utilization=GPU_UTIL)
               if USE_VLLM else {})
    model, tok = FastLanguageModel.from_pretrained(
        model_name=src, max_seq_length=seq_len, load_in_4bit=(four_bit and not full_ft),
        full_finetuning=full_ft, dtype=None, **vllm_kw)
    if full_ft:
        return model, tok  # all params trainable; no LoRA adapter
    # Attach a fresh LoRA unless we're continuing an existing adapter checkpoint (which already has
    # one — a second get_peft_model would error). A merged model (no adapter_config.json) gets a
    # fresh adapter, which is what vLLM-GRPO-from-SFT needs.
    has_adapter = init_from is not None and (_P(src) / "adapter_config.json").exists()
    if not has_adapter:
        model = FastLanguageModel.get_peft_model(
            model, use_gradient_checkpointing="unsloth", random_state=TRAIN["seed"], **LORA)
    return model, tok


def time_budget_callback():
    from transformers import TrainerCallback
    mins = float(_os.environ.get("NEKAISE_BUDGET_MIN", TIME_BUDGET_MIN))

    class _TB(TrainerCallback):
        deadline = time.time() + mins * 60
        def on_step_end(self, args, state, control, **kw):
            if time.time() > self.deadline:
                control.should_training_stop = True
            return control
    return _TB()


def holdout_corpus(pack, max_chars=300000):
    """Full text data-in-hand for the held-out building (retrieval source). Shared: lib/corpus.py."""
    import prepare  # packs/building/prepare.py (fixed data locator)
    from corpus import building_corpus
    text, _ = building_corpus(prepare.DATA / pack.HOLDOUT_BUILDING, max_chars)
    return text


def evaluate(model, tok, pack, n):
    from unsloth import FastLanguageModel
    from corpus import retrieve
    from collections import Counter
    FastLanguageModel.for_inference(model)
    full = holdout_corpus(pack) if OPEN_BOOK else None
    test = pack.load_split("test", n)
    correct = 0
    by, hit = Counter(), Counter()
    for ex in test:
        kind = ex["answer"].split(":")[0]
        by[kind] += 1
        if full is None:
            user = ex["question"]
        else:
            ctx = retrieve(full, ex["question"], ignore=[pack.HOLDOUT_BUILDING])
            user = ("Use only the building's data below to answer.\n\n"
                    f"--- BUILDING DATA ---\n{ctx}\n--- END BUILDING DATA ---\n\n"
                    f"Question: {ex['question']}")
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        pred = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        ok = pack.is_correct(pred, ex["answer"])
        correct += ok
        hit[kind] += int(ok)
    print(f"[eval] by kind: {dict((k, f'{hit[k]}/{by[k]}') for k in sorted(by))}")
    return correct / len(test)


def save_checkpoint(model, tok, stage, metric, value):
    dest = OUT_DIR / stage
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dest))
    tok.save_pretrained(str(dest))
    best_path = OUT_DIR / "best.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else None
    if best is None or value > best.get("value", float("-inf")):
        best_path.write_text(json.dumps(
            {"stage": stage, "metric": metric, "value": round(value, 4),
             "path": f"outputs/{stage}"}, indent=2))
    print(f"[train] saved checkpoint -> {dest}")


def run_sft(model, tok, rows, callbacks):
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    ds = Dataset.from_list([
        {"text": tok.apply_chat_template(r["messages"], tokenize=False)} for r in rows])
    SFTTrainer(
        model=model, processing_class=tok, train_dataset=ds, callbacks=callbacks,
        args=SFTConfig(dataset_text_field="text", max_length=MAX_SEQ_LEN,
                       output_dir=str(OUT_DIR / "_trainer"), report_to="none", **TRAIN),
    ).train()


# --- anchor-recall reward over realistic training-building Q&A (verifiable, on-policy RL) -------
import re as _re, json as _json, os as _os


def _norm(s):
    s = s.lower().replace("_", " ")
    s = _re.sub(r"[^a-z0-9./:\- ]+", " ", s)
    return _re.sub(r"\s+", " ", s).strip()


def _anchor_in(anchor, text):
    a = anchor.strip(); na = _norm(a); nt = _norm(text)
    if not na:
        return False
    if _re.fullmatch(r"-?\d+(\.\d+)?", a):
        return _re.search(rf"(?<![\d.]){_re.escape(a)}(?![\d.])", text) is not None
    if "/" in a or a.endswith((".txt", ".ttl", ".csv", ".md", ".pdf", ".png")):
        base = _norm(a.rsplit("/", 1)[-1]); return na in nt or (len(base) >= 4 and base in nt)
    if na in nt:
        return True
    toks = [t for t in na.split() if len(t) >= 2]
    return len(toks) >= 2 and toks[0] in nt and sum(t in nt for t in toks) >= max(2, len(toks) - 1)


def realistic_grpo_dataset(tok):
    """Training-building realistic Q&A -> {prompt (sys+slice+q), anchors (in-slice, JSON)}."""
    from datasets import Dataset
    from corpus import building_corpus, retrieve
    import prepare
    holdout = _os.environ.get("NEKAISE_HOLDOUT") or prepare.default_holdout()
    qa_dir = EXP_DIR / "data" / "realistic_qa"
    rows = []
    for f in sorted(qa_dir.glob("*.json")):
        b = f.stem
        if b == holdout:
            continue
        corpus, _ = building_corpus(prepare.DATA / b, max_chars=300000)
        for it in _json.loads(f.read_text()):
            if not it.get("anchors"):
                continue
            sl = retrieve(corpus, it["question"], max_chars=GRPO["slice_chars"], ignore=[b])
            in_slice = [a for a in it["anchors"] if _anchor_in(a, sl)]
            if len(in_slice) < 2:
                continue
            user = ("Use only the building's data below to answer.\n\n"
                    f"--- BUILDING DATA ---\n{sl}\n--- END BUILDING DATA ---\n\nQuestion: {it['question']}")
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True)
            rows.append({"prompt": prompt, "anchors": _json.dumps(in_slice, ensure_ascii=False)})
    return Dataset.from_list(rows)


def run_grpo(model, tok, pack, callbacks):
    from trl import GRPOTrainer, GRPOConfig
    ds = realistic_grpo_dataset(tok)
    print(f"[grpo] realistic anchor-reward dataset: {len(ds)} questions", flush=True)

    def reward_fn(completions, anchors, **kw):
        out = []
        for c, anc in zip(completions, anchors):
            al = _json.loads(anc)
            r = sum(_anchor_in(a, c) for a in al) / max(1, len(al))
            if len(c) > 1700:                     # discourage verbosity that truncates at the 400-tok eval cap
                r *= 0.9
            out.append(float(r))
        return out

    g = GRPO
    cfg = GRPOConfig(
        num_generations=g["num_generations"], max_prompt_length=g["max_prompt_length"],
        max_completion_length=g["max_completion_length"],
        per_device_train_batch_size=g["num_generations"], gradient_accumulation_steps=g["grad_accum"],
        learning_rate=g["lr"], max_steps=g["max_steps"], warmup_steps=5, optim="adamw_8bit",
        weight_decay=0.01, lr_scheduler_type="constant", logging_steps=5, seed=TRAIN["seed"],
        beta=g.get("beta", 0.04), temperature=g.get("temperature", 0.9),
        use_vllm=USE_VLLM, save_steps=g.get("save_steps", 40), save_total_limit=4,
        output_dir=str(OUT_DIR / "_trainer"), report_to="none")
    GRPOTrainer(model=model, processing_class=tok, train_dataset=ds, reward_funcs=[reward_fn],
                callbacks=callbacks, args=cfg).train()


def _cpt_dir() -> Path:
    return EXP_DIR / "data" / "cpt"


def cpt_train_texts() -> list[str]:
    p = _cpt_dir() / "train.jsonl"
    if not p.exists():
        raise FileNotFoundError("CPT data missing -- run build_cpt_data.py first.")
    return [json.loads(l)["text"] for l in p.read_text().splitlines() if l.strip()]


def cpt_heldout_texts() -> list[str]:
    p = _cpt_dir() / "heldout.jsonl"
    return [json.loads(l)["text"] for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def eval_perplexity(model, tok, texts, seq_len=2048, max_docs=40):
    """Mean per-token perplexity over held-out HVAC docs -- the ceiling metric for CPT."""
    import math
    import torch
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    nll, ntok = 0.0, 0
    with torch.no_grad():
        for t in texts[:max_docs]:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=seq_len).input_ids.to(model.device)
            if ids.shape[1] < 2:
                continue
            loss = model(ids, labels=ids).loss.item()
            n = ids.shape[1] - 1
            nll += loss * n
            ntok += n
    FastLanguageModel.for_training(model)
    return math.exp(nll / max(1, ntok))


def run_cpt(model, tok, callbacks):
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    c = CPT
    eos = tok.eos_token or ""
    texts = cpt_train_texts()
    ds = Dataset.from_list([{"text": t + eos} for t in texts])
    lr = c["lr_full"] if c["full_finetuning"] else c["lr"]
    mode = "FULL-param" if c["full_finetuning"] else "LoRA"
    print(f"[cpt] {mode} | {len(ds)} docs, ~{sum(len(t) for t in texts)//4/1e6:.2f}M tokens (rough), "
          f"{c['epochs']} epoch(s), lr={lr}", flush=True)
    SFTTrainer(
        model=model, processing_class=tok, train_dataset=ds, callbacks=callbacks,
        args=SFTConfig(dataset_text_field="text", max_length=c["seq_len"], packing=c["packing"],
                       per_device_train_batch_size=c["per_device_train_batch_size"],
                       gradient_accumulation_steps=c["grad_accum"], num_train_epochs=c["epochs"],
                       warmup_steps=c["warmup_steps"], learning_rate=lr, optim="adamw_8bit",
                       weight_decay=c["weight_decay"], lr_scheduler_type="linear",
                       gradient_checkpointing=c["full_finetuning"],
                       logging_steps=c["logging_steps"], seed=TRAIN["seed"], save_steps=c["save_steps"],
                       output_dir=str(OUT_DIR / "_trainer"), report_to="none"),
    ).train()


def main() -> None:
    pack = load_pack(PACK)
    log = RunLogger(EXP_DIR.name, BASE_MODEL, PACK, f"{PACK}_acc") if RunLogger else None
    callbacks = [time_budget_callback()] + ([trainer_callback(log)] if log else [])

    if METHOD == "cpt":
        model, tok = load_model(INIT_FROM, seq_len=CPT["load_seq_len"],
                                four_bit=CPT["load_in_4bit"], full_ft=CPT["full_finetuning"])
    else:
        model, tok = load_model(INIT_FROM)

    t0 = time.time()
    if METHOD == "cpt":
        held = cpt_heldout_texts()
        ppl_before = eval_perplexity(model, tok, held) if held else float("nan")
        run_cpt(model, tok, callbacks)
        ppl_after = eval_perplexity(model, tok, held) if held else float("nan")
        dest = OUT_DIR / STAGE
        dest.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(dest))
        tok.save_pretrained(str(dest))
        train_min = (time.time() - t0) / 60
        if log:
            log.update(metric="cpt_ppl", before=round(ppl_before, 3), after=round(ppl_after, 3),
                       delta=round(ppl_after - ppl_before, 3), status="done", ended=time.time())
        print(f"[cpt] saved adapter -> {dest}")
        print(f"METRIC cpt_ppl_before={ppl_before:.3f} cpt_ppl_after={ppl_after:.3f} "
              f"drop={ppl_before - ppl_after:.3f} stage={STAGE} train_min={train_min:.1f}")
        return

    if METHOD == "sft":
        run_sft(model, tok, resolve_dataset(pack), callbacks)
    elif METHOD == "grpo":
        run_grpo(model, tok, pack, callbacks)
    else:
        raise ValueError(f"unknown METHOD={METHOD!r} (use cpt|sft|grpo)")
    train_min = (time.time() - t0) / 60

    acc = evaluate(model, tok, pack, EVAL_N)
    save_checkpoint(model, tok, STAGE, f"{PACK}_acc", acc)
    if log:
        log.log_step(TRAIN["max_steps"], eval_acc=acc)
        log.finish(after=acc)
    print(f"METRIC {PACK}_acc={acc:.4f} n={EVAL_N} method={METHOD} "
          f"dataset={DATASET} stage={STAGE} train_min={train_min:.1f}")


if __name__ == "__main__":
    main()
