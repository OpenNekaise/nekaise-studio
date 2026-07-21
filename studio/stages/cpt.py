"""CPT stage — Unsloth standard training + WSD schedule. Card row 1 (SPEC.md §1).

    python -m studio.stages.cpt --config configs/cpt.yaml [--seed N] [--dry-run]

Agent-movable: the `data:` section (domain/replay mix, annealing subset). Everything in
FROZEN is spec; assert_frozen refuses drift. Eval = corpus_probes dev absorption via the
gym runner (vLLM) in a fresh subprocess (clean VRAM).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from studio.stages import _common
from studio.stages._common import REPO

FROZEN = {
    "optimizer": "adamw_8bit",
    "schedule": "wsd",              # warmup–stable–decay; shape is spec
    "warmup_ratio": 0.03,
    "decay_ratio": 0.10,
    "max_seq_len": 2048,
    "learning_rate": 1.0e-4,        # validated in agentic-cpt run 1; change = spec change
    "peft": {"kind": "lora", "r": 32, "alpha": 64, "dropout": 0.0},
}

METRIC = "corpus_probes_dev"


def load_texts(cfg: dict, exp_dir: Path) -> list[str]:
    """Data recipe: weighted sources — {'dataset': 'auto'} (datakit LATEST) or
    {'jsonl': path, 'field': 'text'}; integer weight = repetition count."""
    import datakit
    texts: list[str] = []
    for src in cfg["data"]["sources"]:
        weight = int(src.get("weight", 1))
        if src.get("dataset") == "auto":
            d = datakit.latest_dir(exp_dir)
            if not d:
                raise SystemExit("data source 'auto' but no dataset — run build_data.py first")
            rows = [r["text"] for r in datakit.read_dir(d)]
        else:
            import json
            p = Path(src["jsonl"])
            rows = [json.loads(l)[src.get("field", "text")]
                    for l in p.read_text().splitlines() if l.strip()]
        texts += rows * max(1, weight)
    return texts


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "cpt.yaml"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = _common.load_config(args.config)
    _common.assert_frozen(cfg, FROZEN, "cpt")
    if args.dry_run:
        print("cpt config OK (frozen section matches the card)")
        return

    import math

    import trainkit

    run = cfg["run"]
    seed = args.seed if args.seed is not None else int(run.get("seed", 3407))
    exp_dir = REPO / "experiments" / run["experiment"]
    budget = _common.Budget(run["max_minutes"])

    texts = load_texts(cfg, exp_dir)
    print(f"[cpt] {run['base_model']} on {len(texts)} docs "
          f"(~{sum(map(len, texts)) // 4 / 1e6:.1f}M tok), seed={seed}")

    model, tok = trainkit.load_model(
        run.get("init_from") or run["base_model"], max_seq_len=FROZEN["max_seq_len"],
        lora=dict(r=FROZEN["peft"]["r"], lora_alpha=FROZEN["peft"]["alpha"],
                  lora_dropout=FROZEN["peft"]["dropout"],
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]),
        seed=seed)

    bs, ga = int(run.get("per_device_batch", 2)), int(run.get("grad_accum", 8))
    epochs = float(cfg["data"].get("epochs", 1))
    total_steps = max(1, math.ceil(len(texts) / (bs * ga)) * int(math.ceil(epochs)))
    train_args = dict(per_device_train_batch_size=bs, gradient_accumulation_steps=ga,
                      num_train_epochs=epochs, learning_rate=FROZEN["learning_rate"],
                      optim=FROZEN["optimizer"], logging_steps=5, seed=seed,
                      **_common.wsd_kwargs(total_steps, FROZEN["warmup_ratio"],
                                           FROZEN["decay_ratio"]))
    trainkit.run_cpt(model, tok, texts, max_seq_len=FROZEN["max_seq_len"],
                     train_args=train_args, out_dir=exp_dir / "outputs",
                     callbacks=[budget.callback()])

    stage = run.get("stage", "cpt")
    trainkit.save_checkpoint(model, tok, exp_dir / "outputs" / stage,
                             {"stage": stage, "config": cfg, "seed": seed})
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools" / "eval_probes.py"),
         "--checkpoint", str(exp_dir / "outputs" / stage), "--split", "dev",
         "--exp", run["experiment"]], capture_output=True, text=True)
    sys.stdout.write(proc.stdout[-2000:])
    m = re.search(rf"METRIC {METRIC}=([0-9.]+)", proc.stdout)
    value = float(m.group(1)) if m else None
    _common.finish_stage(exp_dir=exp_dir, stage=stage, metric=METRIC, value=value,
                         cfg=cfg, budget=budget, model=model, tok=tok)


if __name__ == "__main__":
    main()
