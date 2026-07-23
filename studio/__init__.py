"""studio — the training side: stages, tools, loop machinery.

Depends one-way on gym/ (tasks + verifiers + runner). Agent entry point:

    python -m studio.cli train <cpt|sft> --config configs/<stage>.yaml

Every stage validates its config's `frozen:` section against the algorithm card
(SPEC.md §1) and refuses to run on drift — the agent's movable knob is the `data:`
section only.
"""
