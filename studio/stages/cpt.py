"""CPT stage — BF16 full-parameter training + WSD schedule. Card row 1 (SPEC.md §1).

    python -m studio.stages.cpt --config configs/cpt.yaml [--seed N] [--dry-run]

Agent-movable: the `data:` section (domain/replay mix, annealing subset). Everything in
FROZEN is spec; assert_frozen refuses drift. This process only trains and saves a
checkpoint. Evaluation is a separate gym command after this process exits and releases
its model and VRAM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from studio.stages import _common
from studio.stages._common import REPO

FROZEN = {
    "optimizer": "adamw_8bit",
    "schedule": "wsd",              # warmup–stable–decay; shape is spec
    "warmup_ratio": 0.03,
    "decay_ratio": 0.10,
    "max_seq_len": 2048,
    "learning_rate": 2.0e-5,        # conservative full-parameter CPT card value
    "precision": "bf16",
    "finetuning": "full",
}

def resolve_dataset_dir(exp_dir: Path, dataset_id: str) -> tuple[Path, dict]:
    """Resolve an explicit immutable dataset object and verify its identity."""
    import datakit
    directory = datakit.data_root(exp_dir) / "objects" / dataset_id
    if not directory.is_dir():
        raise SystemExit(f"no dataset object {dataset_id!r} under {exp_dir}")
    provenance = datakit.provenance(directory)
    if provenance.get("dataset_id") != dataset_id:
        raise SystemExit(f"dataset identity drift at {directory}")
    return directory, provenance


def load_data_paths(cfg: dict, exp_dir: Path,
                    dataset_id: str | None = None) -> tuple[list[Path], int, list[dict]]:
    """Resolve weighted immutable JSONL sources without reading corpus text."""
    import datakit
    paths: list[Path] = []
    content_tokens = 0
    provenances: list[dict] = []
    for src in cfg["data"]["sources"]:
        weight = max(1, int(src.get("weight", 1)))
        if src.get("dataset") == "auto":
            if dataset_id:
                d, provenance = resolve_dataset_dir(exp_dir, dataset_id)
            else:
                d = datakit.latest_dir(exp_dir)
                if not d:
                    raise SystemExit(
                        "data source 'auto' but no dataset — run build_data.py first")
                provenance = datakit.latest_provenance(exp_dir)
            path = datakit.data_file(d)
            tokens = int((provenance.get("stats") or {}).get("content_tokens") or 0)
            if tokens <= 0:
                raise SystemExit(f"dataset {d} has no content_tokens provenance")
            provenances.append(provenance)
        else:
            path = _common.active_workspace().resolve_input(src["jsonl"])
            tokens = int(src.get("content_tokens") or 0)
            if tokens <= 0:
                raise SystemExit(
                    f"external JSONL source {path} requires content_tokens metadata")
            provenances.append({"external_jsonl": str(path), "content_tokens": tokens})
        paths.extend([path] * weight)
        content_tokens += tokens * weight
    return paths, content_tokens, provenances


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "cpt.yaml"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dataset-id", default=None,
                    help="train this immutable dataset object instead of LATEST")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = _common.load_config(args.config)
    _common.assert_frozen(cfg, FROZEN, "cpt")
    if args.dry_run:
        print("cpt config OK (frozen section matches the card)")
        return

    import math

    import trainkit
    import datakit
    import runlog
    from studio.tools import runmeta

    run = cfg["run"]
    seed = args.seed if args.seed is not None else int(run.get("seed", 3407))
    exp_dir = _common.experiment_dir(run["experiment"])
    budget = _common.Budget(run["max_minutes"])

    data_paths, content_tokens, source_provenances = load_data_paths(
        cfg, exp_dir, dataset_id=args.dataset_id)
    if args.dataset_id:
        dataset_dir, dataset_provenance = resolve_dataset_dir(exp_dir, args.dataset_id)
    else:
        dataset_dir = datakit.latest_dir(exp_dir)
        if dataset_dir is None:
            raise SystemExit(f"no dataset for experiment {run['experiment']!r} — "
                             "run its build_data.py first")
        dataset_provenance = datakit.latest_provenance(exp_dir)
    print(f"[cpt] full BF16 {run['base_model']} on "
          f"{content_tokens / 1e6:.3f}M content tokens, seed={seed}")

    fingerprint_files = [
        "SPEC.md", "studio/stages/cpt.py", "studio/stages/_common.py",
        "lib/trainkit.py", "studio/tools/runmeta.py", "lib/runlog.py",
        "requirements.txt",
    ]
    recipe_rel = f"experiments/{run['experiment']}/build_data.py"
    if (REPO / recipe_rel).is_file():
        fingerprint_files.append(recipe_rel)
    config_abs = Path(args.config).resolve()
    try:
        fingerprint_files.append(str(config_abs.relative_to(REPO)))
        fingerprints = runmeta.file_fingerprints(REPO, fingerprint_files)
    except ValueError:                       # config lives in an external workspace
        fingerprints = runmeta.file_fingerprints(REPO, fingerprint_files)
        fingerprints[str(config_abs)] = runmeta.sha256_file(config_abs)
    metric = (run.get("metric")
              or ("operational_smoke" if run.get("kind") == "plumbing_smoke"
                  else "corpus_transfer_macro_dev"))
    config_sha256 = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    logger = runlog.RunLogger(
        run["experiment"], model=run["base_model"], pack="nekaise-corpus",
        metric=metric, stage=run.get("stage", "cpt"),
        kind=run.get("kind", "experiment"), seed=seed,
        dataset_id=dataset_provenance["dataset_id"], config_sha256=config_sha256,
        code=runmeta.git_state(REPO), environment=runmeta.environment(),
        metadata={
            "dataset_stats": dataset_provenance.get("stats", {}),
            "dataset_spec": dataset_provenance.get("spec", {}),
            "file_fingerprints": fingerprints,
        },
    )

    try:
        model, tok = trainkit.load_model(
            run.get("init_from") or run["base_model"],
            max_seq_len=FROZEN["max_seq_len"],
            load_in_4bit=False,
            full_finetuning=True,
            lora=None,
            seed=seed)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        if trainable != total:
            raise RuntimeError(
                f"full CPT invariant failed: trainable={trainable:,}, total={total:,}")
        print(f"[cpt] trainable parameters: {trainable:,}/{total:,} (100.00%)")
        model_revision = getattr(model.config, "_commit_hash", None)
        logger.update(trainable_parameters=trainable, total_parameters=total,
                      model_revision=model_revision)

        bs, ga = int(run.get("per_device_batch", 2)), int(run.get("grad_accum", 8))
        epochs = float(cfg["data"].get("epochs", 1))
        tokens_per_step = FROZEN["max_seq_len"] * bs * ga
        total_steps = max(1, math.ceil(content_tokens * epochs / tokens_per_step))
        train_args = dict(
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=ga,
            num_train_epochs=epochs,
            learning_rate=FROZEN["learning_rate"],
            optim=FROZEN["optimizer"],
            bf16=True,
            fp16=False,
            logging_steps=1,
            save_strategy="no",
            seed=seed,
            dataset_num_proc=int(run.get("preprocess_workers", 4)),
            **_common.wsd_kwargs(
                total_steps, FROZEN["warmup_ratio"], FROZEN["decay_ratio"]),
        )
        trainkit.run_cpt(
            model, tok, data_paths, max_seq_len=FROZEN["max_seq_len"],
            train_args=train_args, out_dir=logger.dir / "scratch",
            cache_dir=datakit.data_root(exp_dir) / "_hf_cache",
            callbacks=[budget.callback(), runlog.trainer_callback(logger)])

        stage = run.get("stage", "cpt")
        artifact = _common.finish_stage(
            exp_dir=exp_dir,
            stage=stage,
            metric=metric,
            value=None,
            cfg=cfg,
            budget=budget,
            run_logger=logger,
            model=model,
            tok=tok,
            log_fields={
                "dataset_id": dataset_provenance["dataset_id"],
                "dataset_stats": dataset_provenance.get("stats", {}),
                "file_fingerprints": fingerprints,
                "model_revision": model_revision,
            },
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
