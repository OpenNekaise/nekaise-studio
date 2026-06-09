#!/usr/bin/env python3
"""GSM8K data prep — FIXED. One-time download/cache of the dataset. Do NOT edit.

    python packs/gsm8k/prepare.py

Just warms the Hugging Face cache and reports split sizes so the first training run
doesn't pay the download mid-experiment.
"""
from datasets import load_dataset


def main() -> None:
    for split in ("train", "test"):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        print(f"gsm8k/{split}: {len(ds)} examples")


if __name__ == "__main__":
    main()
