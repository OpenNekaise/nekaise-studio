"""studio — the training side: stages, tools, loop machinery.

Depends one-way on gym/ (tasks + verifiers + runner). Stage entry points:

    python -m studio.stages.<cpt|sft|rlvr|opd> --config configs/<stage>.yaml

Every stage validates its config's `frozen:` section against the algorithm card
(SPEC.md §1) and refuses to run on drift — the agent's movable knob is the `data:`
section only.
"""
