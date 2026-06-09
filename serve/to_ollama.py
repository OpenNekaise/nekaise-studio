#!/usr/bin/env python3
"""Export a fine-tuned model to GGUF and load it into Ollama — the Studio -> Edge handoff.

    python serve/to_ollama.py --model experiments/granite-4.1-3b-gsm8k/outputs \
        --name nekaise-granite-3b --quant q4_k_m

Merges the LoRA adapter into the base, converts to a quantized GGUF (Unsloth does this in
one call), writes a Modelfile, and runs `ollama create`. nekaise-edge then pulls/runs the
resulting Ollama model.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

MODELFILE = """FROM ./model.gguf
PARAMETER temperature 0
"""


def export(model_dir: Path, name: str, quant: str) -> None:
    out = model_dir / "gguf"
    out.mkdir(parents=True, exist_ok=True)

    # Unsloth merges the adapter and writes a quantized GGUF in one call:
    #   from unsloth import FastLanguageModel
    #   model, tok = FastLanguageModel.from_pretrained(model_dir)
    #   model.save_pretrained_gguf(out, tok, quantization_method=quant)
    # TODO: wire up the call above; for now this assembles the Modelfile + ollama create.

    (out / "Modelfile").write_text(MODELFILE)
    subprocess.run(["ollama", "create", name, "-f", str(out / "Modelfile")], check=True)
    print(f"created ollama model: {name}  (run: ollama run {name})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True, help="trained model/adapter dir")
    ap.add_argument("--name", required=True, help="ollama model name to create")
    ap.add_argument("--quant", default="q4_k_m")
    args = ap.parse_args()
    export(args.model, args.name, args.quant)


if __name__ == "__main__":
    main()
