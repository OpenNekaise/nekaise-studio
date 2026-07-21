#!/usr/bin/env python3
"""build_data.py — WHAT: the pure-CPT corpus (MUTABLE recipe).

No teacher, no QA pairs: this is PURE continued pretraining. Loads the deterministic
curated corpus slice (lib/corpusprep: core topics, English-dominant, domain-dense,
top-N docs), drops the probe pack's heldout docs (corpusprep.is_heldout_doc — the same
rule that makes those docs `transfer` probes), and caches raw-text rows via datakit.

Knobs to vary here: N_DOCS (corpus size), MIN_DOC_CHARS, later: doc mixing/repetition,
paraphrase augmentation (see experiments/granite-4.1-3b-building/augment_corpus.py).

    python experiments/agentic-cpt/build_data.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO = EXP_DIR.parents[1]
sys.path.insert(0, str(REPO / "lib"))
import corpusprep  # noqa: E402
import datakit  # noqa: E402

CORPUS = REPO / "nekaise_data" / "hvac_corpus"
N_DOCS = int(os.environ.get("NEKAISE_CPT_DOCS", 1200))


def main() -> None:
    docs = corpusprep.load_clean_docs(CORPUS, n_docs=N_DOCS)
    train = [d for d in docs if not corpusprep.is_heldout_doc(d["id"])]
    heldout = len(docs) - len(train)
    rows = [{"text": d["text"], "meta": {"doc_id": d["id"], "topic": d["topic"]}}
            for d in train]
    chars = sum(len(r["text"]) for r in rows)
    spec = {"kind": "cpt", "task": "pure_agentic_cpt", "source": "hvac_corpus",
            "n_docs": N_DOCS, "selector": "corpusprep.select_docs",
            "holdout_rule": "corpusprep.is_heldout_doc"}
    d = datakit.write(EXP_DIR, spec, rows,
                      stats={"train_docs": len(rows), "heldout_docs": heldout,
                             "train_chars": chars, "train_tokens_est": chars // 4})
    print(f"[build] {len(rows)} train docs ({chars/1e6:.1f}M chars ~{chars//4/1e6:.1f}M tok), "
          f"{heldout} heldout (probe-transfer) docs excluded -> {d}")


if __name__ == "__main__":
    main()
