"""gym.verifiers — pure verifier functions: the only definition of right and wrong.

Contract (R1): every verifier is

    verify(prompt: str, response: str, meta: dict) -> float   # in [0, 1]

- pure: no framework dependency, no global state, no I/O beyond what `meta` names
  (sparql_exec reads the ontology file meta points at; sandbox_tests spawns a sandboxed
  subprocess — both still a function of their inputs);
- `meta` carries the gold: each module documents the keys it needs;
- the SAME function is imported by the TRL reward wrapper, the SFT gate/filter, and the
  eval runner. Adapters stay thin; logic lives here only.

Hard-verifiable tier (SPEC §6 — grow this share every version): numeric_cloze,
final_number, numeric_tolerance, ontology_qa, sparql_exec, sandbox_tests, bench_qa.
Soft tier (falls back where hard verification is impossible): anchor_recall. The LLM
judge is advisory-only and lives outside gym entirely.
"""
from __future__ import annotations

from typing import Callable

from . import (anchor_recall, bench_qa, final_number, numeric_cloze,
               numeric_tolerance, ontology_qa, sandbox_tests, sparql_exec)

Verifier = Callable[[str, str, dict], float]

REGISTRY: dict[str, Verifier] = {
    "numeric_cloze": numeric_cloze.verify,
    "final_number": final_number.verify,
    "numeric_tolerance": numeric_tolerance.verify,
    "ontology_qa": ontology_qa.verify,
    "anchor_recall": anchor_recall.verify,
    "bench_qa": bench_qa.verify,
    "sparql_exec": sparql_exec.verify,
    "sandbox_tests": sandbox_tests.verify,
}


def get(name: str) -> Verifier:
    if name not in REGISTRY:
        raise KeyError(f"unknown verifier '{name}' (have: {sorted(REGISTRY)})")
    return REGISTRY[name]
