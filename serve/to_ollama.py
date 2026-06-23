#!/usr/bin/env python3
"""Export a winning checkpoint to GGUF and load it into Ollama — the Studio → Edge handoff.

    # default: export the experiment's BEST checkpoint (from outputs/best.json)
    python serve/to_ollama.py --exp granite-4.1-3b-gsm8k --name nekaise-granite-3b

    # or point at a specific stage/dir
    python serve/to_ollama.py --model experiments/granite-4.1-3b-gsm8k/outputs/grpo \
        --name nekaise-granite-3b --quant q4_k_m

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
MODELFILE = "FROM ./{gguf}\nPARAMETER temperature 0\n"


def resolve_model_dir(exp: str, model: Path | None) -> Path:
    """An explicit --model wins; otherwise read the experiment's outputs/best.json."""
    if model:
        return model
    best = REPO / "experiments" / exp / "outputs" / "best.json"
    if not best.exists():
        sys.exit(f"no {best} — run train.py first (it records the best stage), or pass --model")
    info = json.loads(best.read_text())
    d = REPO / "experiments" / exp / info["path"]
    print(f"best checkpoint: stage={info['stage']} {info['metric']}={info['value']} -> {d}")
    return d


def export(model_dir: Path, name: str, quant: str) -> None:
    from unsloth import FastLanguageModel  # heavy; imported only when actually exporting

    out = model_dir / "gguf"
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
    ap.add_argument("--exp", default="granite-4.1-3b-gsm8k", help="experiment folder name")
    ap.add_argument("--model", type=Path, help="explicit checkpoint dir (overrides --exp best)")
    ap.add_argument("--name", required=True, help="ollama model name to create")
    ap.add_argument("--quant", default="q4_k_m")
    args = ap.parse_args()
    export(resolve_model_dir(args.exp, args.model), args.name, args.quant)


if __name__ == "__main__":
    main()
