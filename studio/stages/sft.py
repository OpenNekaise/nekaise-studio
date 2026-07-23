"""SFT stage — TRL SFTTrainer, plain cross-entropy on judge-gated teacher data.
Card row 2 (SPEC.md §1).

    python -m studio.stages.sft --config configs/sft.yaml [--seed N] [--dry-run]

Agent-movable: the `data:` section (synthesis strategy / filter threshold / mix live in
the data-production recipe; here: which dataset + mix ratios). The loss function is
frozen — no variants, ever.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from studio.stages import _common
from studio.stages._common import REPO

FROZEN = {
    "loss": "cross_entropy",        # plain CE via TRL SFTTrainer — no variants
    "optimizer": "adamw_8bit",
    "schedule": "cosine",
    "max_seq_len": 2048,
    "learning_rate": 2.0e-4,
    "peft": {"kind": "lora", "r": 16, "alpha": 32, "dropout": 0.0},
}

METRIC_DEFAULT = "task_acc_dev"


def load_rows(cfg: dict, exp_dir: Path) -> list[dict]:
    """[{'messages': [...]}] from datakit LATEST ('auto') or explicit jsonl paths,
    with integer mix weights — the data recipe."""
    import datakit
    rows: list[dict] = []
    for src in cfg["data"]["sources"]:
        weight = int(src.get("weight", 1))
        if src.get("dataset") == "auto":
            d = datakit.latest_dir(exp_dir)
            if not d:
                raise SystemExit("data source 'auto' but no dataset — run build_data.py first")
            part = datakit.read_dir(d)
        else:
            p = _common.active_workspace().resolve_input(src["jsonl"])
            part = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        rows += part * max(1, weight)
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "sft.yaml"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = _common.load_config(args.config)
    _common.assert_frozen(cfg, FROZEN, "sft")
    if args.dry_run:
        print("sft config OK (frozen section matches the card)")
        return

    import datakit
    import runlog
    import trainkit
    from studio.tools import runmeta

    run = cfg["run"]
    seed = args.seed if args.seed is not None else int(run.get("seed", 3407))
    exp_dir = _common.experiment_dir(run["experiment"])
    budget = _common.Budget(run["max_minutes"])
    rows = load_rows(cfg, exp_dir)
    dataset_dir = datakit.latest_dir(exp_dir)
    dataset_provenance = datakit.latest_provenance(exp_dir) if dataset_dir else {}
    source_ref = run.get("init_from") or run["base_model"]
    config_sha256 = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    logger = runlog.RunLogger(
        run["experiment"], model=source_ref, pack="sft_data",
        metric=run.get("metric", METRIC_DEFAULT), stage=run.get("stage", "sft"),
        kind=run.get("kind", "experiment"), seed=seed,
        dataset_id=dataset_provenance.get("dataset_id"), config_sha256=config_sha256,
        code=runmeta.git_state(REPO), environment=runmeta.environment(),
        metadata={"dataset_stats": dataset_provenance.get("stats", {}),
                  "rows": len(rows)},
    )
    print(f"[sft] {source_ref} on {len(rows)} demos, seed={seed}")

    try:
        model, tok = trainkit.load_model(
            logger.store.resolve_model_ref(source_ref), max_seq_len=FROZEN["max_seq_len"],
            lora=dict(r=FROZEN["peft"]["r"], lora_alpha=FROZEN["peft"]["alpha"],
                      lora_dropout=FROZEN["peft"]["dropout"],
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"]),
            seed=seed)
        logger.update(model_revision=getattr(model.config, "_commit_hash", None))
        trainkit.run_sft(
            model, tok, rows, max_seq_len=FROZEN["max_seq_len"],
            train_args=dict(per_device_train_batch_size=int(run.get("per_device_batch", 4)),
                            gradient_accumulation_steps=int(run.get("grad_accum", 4)),
                            num_train_epochs=float(cfg["data"].get("epochs", 2)),
                            learning_rate=FROZEN["learning_rate"], optim=FROZEN["optimizer"],
                            lr_scheduler_type=FROZEN["schedule"], logging_steps=5,
                            save_strategy="no", seed=seed),
            out_dir=logger.dir / "scratch",
            callbacks=[budget.callback(), runlog.trainer_callback(logger)],
        )

        stage = run.get("stage", "sft")
        artifact = _common.finish_stage(
            exp_dir=exp_dir, stage=stage, metric=run.get("metric", METRIC_DEFAULT),
            value=None, cfg=cfg, budget=budget, run_logger=logger, model=model, tok=tok,
            log_fields={"dataset_id": dataset_provenance.get("dataset_id")},
        )
        logger.trained(checkpoint=artifact, minutes=budget.minutes,
                       timeboxed=budget.exceeded)
        print(f"RUN_ID {logger.run_id}")
        print(f"CHECKPOINT {logger.store.resolve_stored_path(artifact['path'])}")
    except BaseException as exc:
        logger.update(minutes=budget.minutes)
        logger.fail(exc)
        raise


if __name__ == "__main__":
    main()
