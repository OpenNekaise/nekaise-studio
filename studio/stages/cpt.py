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


def _save_kwargs(save_steps: int) -> dict:
    """Operational crash safety (run.save_steps). 0 keeps the historical no-save
    behaviour; N>0 writes a full Trainer checkpoint (model+optimizer+scheduler+state)
    every N steps under the run's scratch dir, keeping the last two. Multi-day ladder
    runs need this; it does not touch the frozen card."""
    if save_steps <= 0:
        return {"save_strategy": "no"}
    return {"save_strategy": "steps", "save_steps": save_steps,
            "save_total_limit": 2, "save_only_model": False}


RESUME_STRICT_FILES = ("SPEC.md", "studio/stages/cpt.py", "studio/stages/_common.py",
                       "studio/tools/runmeta.py", "lib/trainkit.py", "lib/runlog.py",
                       "requirements.txt")
RESUME_STRICT_PACKAGES = ("unsloth", "unsloth_zoo", "trl", "transformers", "torch",
                          "datasets", "accelerate", "bitsandbytes")
CHECKPOINT_REQUIRED = ("trainer_state.json", "optimizer.pt", "scheduler.pt")


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _valid_safetensors(path: Path) -> bool:
    """The file is exactly 8 + header_len + max(data_offsets) bytes and its JSON header
    parses: a truncated or over-long shard fails without importing torch."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            header_len = int.from_bytes(handle.read(8), "little")
            if header_len <= 0 or header_len > size - 8:
                return False
            header = json.loads(handle.read(header_len))
        end = 0
        for name, info in header.items():
            if name == "__metadata__":
                continue
            end = max(end, int((info.get("data_offsets") or [0, 0])[1]))
        return size == 8 + header_len + end
    except (OSError, ValueError, TypeError, AttributeError, OverflowError):
        return False


def _valid_torch_zip(path: Path) -> bool:
    """torch.save writes a zip archive; a torn write loses the central directory or
    corrupts an entry, both of which testzip() reports."""
    import zipfile
    try:
        with zipfile.ZipFile(path) as archive:
            return bool(archive.namelist()) and archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def _complete_checkpoint(directory: Path) -> bool:
    """A Trainer checkpoint is usable only if every state file it needs is present,
    structurally intact (readable zip archives, safetensors whose size matches the
    header), every weight shard named by the safetensors index exists and validates,
    the single-process RNG file is present, and trainer_state.json (written last by the
    Trainer) parses to the step named in the directory."""
    if not directory.is_dir():
        return False
    if not _nonempty(directory / "trainer_state.json"):
        return False
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        if not _valid_torch_zip(directory / name):
            return False
    index = directory / "model.safetensors.index.json"
    loose_shards = list(directory.glob("model-*-of-*.safetensors"))
    if index.is_file():
        try:
            shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
        except (OSError, json.JSONDecodeError, AttributeError):
            return False
        if not shards or not all(_valid_safetensors(directory / shard) for shard in shards):
            return False
    elif loose_shards:
        return False                     # shards without their index: orphaned save
    elif (directory / "model.safetensors").is_file():
        if not _valid_safetensors(directory / "model.safetensors"):
            return False
    elif (directory / "pytorch_model.bin").is_file():
        if not _valid_torch_zip(directory / "pytorch_model.bin"):
            return False
    else:
        return False
    try:
        state = json.loads((directory / "trainer_state.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    try:
        step = int(directory.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return False
    return int(state.get("global_step", -1)) == step


def _last_complete_checkpoint(trainer_dir: Path) -> Path | None:
    candidates = []
    for child in trainer_dir.glob("checkpoint-*"):
        try:
            candidates.append((int(child.name.rsplit("-", 1)[1]), child))
        except (IndexError, ValueError):
            continue
    for _step, child in sorted(candidates, reverse=True):
        if _complete_checkpoint(child):
            return child
        print(f"[cpt] skipping incomplete checkpoint {child}")
    return None


def _prior_process_alive(prior: dict) -> bool:
    """True if the interrupted run's recorded trainer process is still running on this
    host (a resume would then start a second trainer on the same GPU)."""
    import os
    import socket
    pid = prior.get("pid")
    if not pid or prior.get("host") != socket.gethostname():
        return False
    try:
        cmdline = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except (OSError, ValueError):
        return False
    return b"studio.stages.cpt" in cmdline or b"studio/stages/cpt.py" in cmdline


def _resume_checkpoint(logger, prior_run_id: str, *, dataset_id: str,
                       config_sha256: str, seed: int, experiment: str,
                       fingerprints: dict, environment: dict) -> tuple[str, str | None]:
    """Locate the last COMPLETE Trainer checkpoint of an interrupted run and assert
    that this invocation is the same experiment: same dataset, same semantic config,
    same seed, same experiment, unchanged frozen-card code, identical training
    package versions, and a prior run that did not finish. Every check is strict: a
    missing value on either side refuses the resume (never fail-open). Returns the
    checkpoint path and the prior run's recorded model revision, which main() checks
    against the loaded model before training."""
    prior = logger.store.get_run(prior_run_id)
    problems = []
    if not prior.get("dataset_id") or prior.get("dataset_id") != dataset_id:
        problems.append(f"dataset {prior.get('dataset_id')} != {dataset_id}")
    if prior.get("seed") is None or int(prior.get("seed")) != int(seed):
        problems.append(f"seed {prior.get('seed')} != {seed}")
    if not prior.get("config_sha256") or prior.get("config_sha256") != config_sha256:
        problems.append("config sha256 missing or differs from the interrupted run")
    if prior.get("experiment") != experiment:
        problems.append(f"experiment {prior.get('experiment')} != {experiment}")
    if prior.get("status") in {"trained", "evaluating", "succeeded"}:
        problems.append(f"prior run finished training (status {prior.get('status')})")
    if _prior_process_alive(prior):
        problems.append(f"prior run process pid={prior.get('pid')} is still alive on this host")
    prior_fp = (prior.get("metadata") or {}).get("file_fingerprints") or {}
    for name in RESUME_STRICT_FILES:
        if not prior_fp.get(name):
            problems.append(f"{name} fingerprint missing on the interrupted run")
        elif not fingerprints.get(name):
            problems.append(f"{name} fingerprint missing on this invocation")
        elif prior_fp[name] != fingerprints[name]:
            problems.append(f"{name} changed since the interrupted run")
    prior_env = prior.get("environment") or {}
    for package in RESUME_STRICT_PACKAGES:
        if not prior_env.get(package) or not environment.get(package):
            problems.append(f"{package} version missing on one side")
        elif prior_env[package] != environment[package]:
            problems.append(f"{package} {prior_env[package]} != {environment[package]}")
    if problems:
        raise SystemExit("refusing to resume " + prior_run_id + ": " + "; ".join(problems))
    trainer_dir = logger.store.run_dir(prior["experiment"], prior_run_id) / "scratch" / "_trainer"
    last = _last_complete_checkpoint(trainer_dir) if trainer_dir.is_dir() else None
    if last is None:
        raise SystemExit(f"no complete Trainer checkpoint under {trainer_dir}")
    prior_revision = (prior.get("model_revision")
                      or (prior.get("metadata") or {}).get("model_revision"))
    return str(last), prior_revision


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "cpt.yaml"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dataset-id", default=None,
                    help="train this immutable dataset object instead of LATEST")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume-from", default=None, metavar="RUN_ID",
                    help="continue an interrupted run from its last Trainer checkpoint "
                         "(same dataset, config and seed are asserted); this run gets "
                         "its own run id with parent_run_id set")
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
        init_ref = run.get("init_from") or run["base_model"]
        init_source = logger.store.resolve_model_ref(str(init_ref))
        logger.update(init_from=str(init_ref), init_source=init_source)
        if init_source != init_ref:
            print(f"[cpt] init_from {init_ref} -> {init_source}")
        resume_checkpoint = None
        prior_revision = None
        if args.resume_from:
            resume_checkpoint, prior_revision = _resume_checkpoint(
                logger, args.resume_from, dataset_id=dataset_provenance["dataset_id"],
                config_sha256=config_sha256, seed=seed, experiment=run["experiment"],
                fingerprints=fingerprints, environment=runmeta.environment())
            logger.update(parent_run_id=args.resume_from,
                          resume_checkpoint=resume_checkpoint)
            print(f"[cpt] resuming {args.resume_from} from {resume_checkpoint}")
        model, tok = trainkit.load_model(
            init_source,
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
        if args.resume_from and (prior_revision or model_revision) \
                and prior_revision != model_revision:
            raise SystemExit(
                f"refusing to resume {args.resume_from}: model revision "
                f"{prior_revision} != {model_revision} (base weights or tokenizer changed)")
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
            seed=seed,
            **_save_kwargs(int(run.get("save_steps", 0) or 0)),
            dataset_num_proc=int(run.get("preprocess_workers", 4)),
            **_common.wsd_kwargs(
                total_steps, FROZEN["warmup_ratio"], FROZEN["decay_ratio"]),
        )
        trainkit.run_cpt(
            model, tok, data_paths, max_seq_len=FROZEN["max_seq_len"],
            train_args=train_args, out_dir=logger.dir / "scratch",
            cache_dir=datakit.data_root(exp_dir) / "_hf_cache",
            callbacks=[budget.callback(), runlog.trainer_callback(logger)],
            resume_from_checkpoint=resume_checkpoint)

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
