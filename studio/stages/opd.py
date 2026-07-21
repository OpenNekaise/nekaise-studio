"""OPD stage — on-policy distillation via TRL GKDTrainer, sampled-token variant.
Card row 4 (SPEC.md §1).

    python -m studio.stages.opd --config configs/opd.yaml [--seed N] [--dry-run]

Agent-movable: TEACHER CHOICE only (own post-RL checkpoint / privileged-information
self-distill). Divergence settings are frozen; no top-k variants. Same loop type as RLVR
(GRPO-style): conceptually the verifier reward swapped for the teacher log-ratio —
GKDTrainer is TRL's packaging of exactly that. Teacher scoring uses the HF forward pass
(GKD default) for now; `prompt_logprobs` acceleration via vLLM is a roadmap item (R4).
"""
from __future__ import annotations

import argparse

from studio.stages import _common
from studio.stages._common import REPO

FROZEN = {
    "algo": "gkd",
    "lmbda": 1.0,                   # fully on-policy: student samples every step
    "beta": 0.5,                    # generalized JSD interpolation — frozen, no variants
    "seq_kd": False,
    "top_k": None,                  # no top-k tricks
    "learning_rate": 1.0e-6,
    "schedule": "constant",
    "temperature": 1.0,
    "peft": {"kind": "lora", "r": 32, "alpha": 64, "dropout": 0.0},
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "opd.yaml"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = _common.load_config(args.config)
    _common.assert_frozen(cfg, FROZEN, "opd")
    if args.dry_run:
        print("opd config OK (frozen section matches the card)")
        return

    run = cfg["run"]
    seed = args.seed if args.seed is not None else int(run.get("seed", 3407))
    exp_dir = REPO / "experiments" / run["experiment"]
    budget = _common.Budget(run["max_minutes"])

    from gym import tasks as gym_tasks
    rows = []
    for spec in cfg["data"]["tasks"]:
        rows += gym_tasks.load(spec["set"], split=spec.get("split", "dev"),
                               n=int(spec.get("n", 0)))
    teacher = cfg["data"]["teacher"]          # THE knob: post-RL self / privileged-info
    print(f"[opd] student={run['init_from']} teacher={teacher} on {len(rows)} prompts")

    import trainkit
    from datasets import Dataset
    from transformers import AutoModelForCausalLM
    from trl import GKDConfig, GKDTrainer

    model, tok = trainkit.load_model(
        run["init_from"], max_seq_len=int(run.get("max_seq_len", 2048)),
        lora=dict(r=FROZEN["peft"]["r"], lora_alpha=FROZEN["peft"]["alpha"],
                  lora_dropout=FROZEN["peft"]["dropout"],
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]),
        seed=seed)
    teacher_path = str(REPO / teacher) if (REPO / str(teacher)).exists() else str(teacher)
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_path, dtype="auto")

    def msgs(t):
        sys_part = [{"role": "system", "content": t.system}] if t.system else []
        return {"messages": sys_part + [{"role": "user", "content": t.prompt},
                                        {"role": "assistant", "content": t.ref_solution}],
                "task_id": t.id}

    ds = Dataset.from_list([msgs(t) for t in rows])
    gkd_args = GKDConfig(
        lmbda=FROZEN["lmbda"], beta=FROZEN["beta"], seq_kd=FROZEN["seq_kd"],
        temperature=FROZEN["temperature"], learning_rate=FROZEN["learning_rate"],
        lr_scheduler_type=FROZEN["schedule"],
        per_device_train_batch_size=int(run.get("per_device_batch", 2)),
        gradient_accumulation_steps=int(run.get("grad_accum", 8)),
        max_steps=int(run.get("max_steps", 500)), logging_steps=1, seed=seed,
        output_dir=str(exp_dir / "outputs" / "_trainer"), report_to="none")
    GKDTrainer(model=model, teacher_model=teacher_model, processing_class=tok,
               args=gkd_args, train_dataset=ds,
               callbacks=[budget.callback()]).train()

    stage = run.get("stage", "opd")
    _common.finish_stage(exp_dir=exp_dir, stage=stage, metric=run.get("metric", "kd_loss"),
                         value=None, cfg=cfg, budget=budget, model=model, tok=tok,
                         log_fields={"variable": f"opd: teacher={teacher}"})


if __name__ == "__main__":
    main()
