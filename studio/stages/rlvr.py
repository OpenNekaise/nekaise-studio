"""RLVR stage — TRL GRPOTrainer + gym verifier, JustRL configuration VERBATIM.
Card row 3 (SPEC.md §1 + Appendix A). The frozen values below ARE the appendix table;
tests assert the two never drift apart.

    python -m studio.stages.rlvr --config configs/rlvr.yaml [--seed N] [--dry-run]

Agent-movable: task-set composition + verifier binding (`data:` section). ALL
hyperparameters are frozen (start = JustRL published values, no search); clip-higher is
the only stabilizer. Rollouts via vLLM (Unsloth fast_inference shared-weights mode) —
transformers `.generate()` is banned on this path (R4).
"""
from __future__ import annotations

import argparse

from studio.stages import _common
from studio.stages._common import REPO

FROZEN = {
    # JustRL (arXiv:2512.16649 Table 2) → TRL GRPOConfig; SPEC.md Appendix A
    "algo": "grpo",
    "learning_rate": 1.0e-6,
    "schedule": "constant",
    "num_generations": 8,            # rollout N (group size)
    "temperature": 1.0,              # fixed, no adaptive scheduling
    "epsilon": 0.2,                  # clip ratio [0.8, 1.28] → low
    "epsilon_high": 0.28,            #                        → high (clip-higher, DAPO)
    "beta": 0.0,                     # NO KL loss (TRL default is nonzero — must be 0.0)
    "entropy_regularization": False,  # none
    "effective_prompts_per_step": 256,
    "reward": "binary_verifier",     # gym verifier, binary/simple — shaping = spec change
    "stabilizer": "clip_higher_only",
    "rollout_engine": "vllm",        # unsloth fast_inference / TRL colocate
    "peft": {"kind": "lora", "r": 32, "alpha": 64, "dropout": 0.0},
}


def load_tasks(cfg: dict):
    """Task-set composition — THE movable knob. Optional trainable-band filter (R6)."""
    from gym import tasks as gym_tasks
    from gym.tasks import sampler
    rows = []
    for spec in cfg["data"]["tasks"]:
        part = gym_tasks.load(spec["set"], split=spec.get("split", "dev"),
                              n=int(spec.get("n", 0)))
        if spec.get("trainable_filter"):
            part = sampler.filter_trainable(part, spec["set"])
        rows += part
    if not rows:
        raise SystemExit("task-set composition resolved to zero tasks")
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "rlvr.yaml"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = _common.load_config(args.config)
    _common.assert_frozen(cfg, FROZEN, "rlvr")
    if args.dry_run:
        print("rlvr config OK (frozen section matches the card / SPEC Appendix A)")
        return

    run = cfg["run"]
    seed = args.seed if args.seed is not None else int(run.get("seed", 3407))
    exp_dir = REPO / "experiments" / run["experiment"]
    budget = _common.Budget(run["max_minutes"])
    rows = load_tasks(cfg)
    print(f"[rlvr] {len(rows)} tasks, group={FROZEN['num_generations']}, seed={seed}")

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_name=str(REPO / run["init_from"]) if not str(run["init_from"]).startswith(
            ("unsloth/", "Qwen/", "ibm-granite/")) else run["init_from"],
        max_seq_length=int(run.get("max_seq_len", 2048)),
        fast_inference=True,                       # vLLM shared-weights rollout (R4)
        max_lora_rank=FROZEN["peft"]["r"], dtype=None)
    if hasattr(tok, "tokenizer"):
        tok = tok.tokenizer
    model = FastLanguageModel.get_peft_model(
        model, r=FROZEN["peft"]["r"], lora_alpha=FROZEN["peft"]["alpha"],
        lora_dropout=FROZEN["peft"]["dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=seed)

    def as_prompt(t):
        if t.mode == "complete":
            return t.prompt
        msgs = ([{"role": "system", "content": t.system}] if t.system else []) + \
            [{"role": "user", "content": t.prompt}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    ds = Dataset.from_list([{"prompt": as_prompt(t), "task_id": t.id} for t in rows])

    # Batch GEOMETRY scales to hardware; effective prompts/step is the frozen semantic.
    per_dev = int(run.get("per_device_completions", 8))
    grad_accum = max(1, FROZEN["effective_prompts_per_step"]
                     * FROZEN["num_generations"] // per_dev)
    grpo_args = GRPOConfig(
        learning_rate=FROZEN["learning_rate"], lr_scheduler_type=FROZEN["schedule"],
        num_generations=FROZEN["num_generations"], temperature=FROZEN["temperature"],
        epsilon=FROZEN["epsilon"], epsilon_high=FROZEN["epsilon_high"],
        beta=FROZEN["beta"],
        per_device_train_batch_size=per_dev, gradient_accumulation_steps=grad_accum,
        max_prompt_length=int(run.get("max_prompt_length", 1024)),
        max_completion_length=int(run.get("max_completion_length", 256)),
        max_steps=int(run.get("max_steps", 1000)), logging_steps=1, seed=seed,
        output_dir=str(exp_dir / "outputs" / "_trainer"), report_to="none")
    GRPOTrainer(model=model, processing_class=tok, args=grpo_args, train_dataset=ds,
                reward_funcs=[_common.make_reward(rows)],
                callbacks=[budget.callback()]).train()

    stage = run.get("stage", "rlvr")
    _common.finish_stage(exp_dir=exp_dir, stage=stage, metric=run.get("metric", "reward"),
                         value=None, cfg=cfg, budget=budget, model=model, tok=tok,
                         log_fields={"variable": "rlvr: config run — eval via the runner"})


if __name__ == "__main__":
    main()
