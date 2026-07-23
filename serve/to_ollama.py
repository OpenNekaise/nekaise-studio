#!/usr/bin/env python3
"""Export a winning checkpoint to GGUF and load it into Ollama — the Studio → Edge handoff.

    # default: export the experiment's best metric pointer
    python serve/to_ollama.py --exp cpt --metric corpus_probes_dev --name nekaise-1b

    # or point at an exact run (preferred for reproducibility)
    python serve/to_ollama.py --run-id <run_id> \
        --name nekaise-1b --quant q4_k_m

Merges the LoRA adapter into the base, writes a quantized GGUF (Unsloth does it in one
call), writes a Modelfile, and runs `ollama create`. Then point nekaise-edge at it with
OLLAMA_MODEL=<name>.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lib"))
from runstore import RunStore  # noqa: E402
from workspace import Workspace  # noqa: E402

MODELFILE = "FROM ./{gguf}\nPARAMETER temperature 0\n"
ACTIVE_WORKSPACE = Workspace.resolve(REPO).apply_environment()


def resolve_model_dir(exp: str, model: Path | None, run_id: str | None,
                      metric: str) -> tuple[str | None, Path]:
    """Explicit path/run wins; otherwise resolve the immutable best pointer."""
    if model:
        return None, model
    store = RunStore(REPO)
    if run_id:
        return run_id, store.resolve_checkpoint(run_id)
    try:
        return store.resolve_pointer(exp, f"best:{metric}")
    except KeyError as exc:
        sys.exit(f"{exc}; evaluate and decide a run first, or pass --run-id/--model")


def export(model_dir: Path, out: Path, name: str, quant: str) -> None:
    from unsloth import FastLanguageModel  # heavy; imported only when actually exporting

    out.mkdir(parents=True, exist_ok=True)

    # Unsloth merges the adapter into the base and writes a quantized GGUF in one call.
    model, tok = FastLanguageModel.from_pretrained(model_name=str(model_dir), load_in_4bit=False)
    model.save_pretrained_gguf(str(out), tok, quantization_method=quant)

    gguf = next(out.glob("*.gguf"))
    (out / "Modelfile").write_text(MODELFILE.format(gguf=gguf.name))
    subprocess.run(["ollama", "create", name, "-f", str(out / "Modelfile")], check=True)
    print(f"created ollama model: {name}  (run: ollama run {name})")
    print(f"point nekaise-edge at it:  OLLAMA_MODEL={name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default="cpt", help="experiment folder name")
    ap.add_argument("--model", type=Path, help="explicit checkpoint dir (overrides --exp best)")
    ap.add_argument("--run-id", help="immutable checkpoint producer run")
    ap.add_argument("--metric", default="corpus_probes_dev")
    ap.add_argument("--name", required=True, help="ollama model name to create")
    ap.add_argument("--quant", default="q4_k_m")
    args = ap.parse_args()
    run_id, model_dir = resolve_model_dir(args.exp, args.model, args.run_id, args.metric)
    export_id = run_id or "explicit"
    out = ACTIVE_WORKSPACE.store_dir / "exports" / export_id / args.quant
    export(model_dir, out, args.name, args.quant)


if __name__ == "__main__":
    main()
