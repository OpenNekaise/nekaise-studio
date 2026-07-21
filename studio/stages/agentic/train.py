"""Agentic training (phase B) — rejection SFT now; short-horizon GRPO blocked on the
harness interface.

    python -m studio.stages.agentic.train --config configs/agentic.yaml [--dry-run]

Phase A (works today): collect.py wrote trajectories.jsonl → point configs/sft.yaml's
data.sources at it and run the sft stage (plain CE on solved trajectories — rejection
SFT is exactly that; no new trainer, per the card).

Phase B (short-horizon GRPO, outcome reward): needs multi-turn rollout through the agent
runtime. That requires opennekaise's headless programmable mode (one command = one task,
structured output) — the interface request is drafted in docs/opennekaise-headless-issue.md
and serves gym exams, RL rollout, and CI regression alike. Until it exists this entry
point validates the config and exits with status 2 (blocked), so nothing pretends to
train. No process rewards, no long-trajectory dense supervision — ever (card row 5).
"""
from __future__ import annotations

import argparse
import sys

from studio.stages import _common
from studio.stages._common import REPO
from studio.stages.agentic import FROZEN


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "agentic.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = _common.load_config(args.config)
    _common.assert_frozen(cfg, FROZEN, "agentic")
    if args.dry_run:
        print("agentic config OK (frozen section matches the card)")
        return

    print("agentic phase B (short-horizon GRPO) is BLOCKED on the opennekaise headless "
          "interface — see docs/opennekaise-headless-issue.md.\n"
          "Available today: python -m studio.stages.agentic.collect (trajectories), then "
          "the sft stage on the collected jsonl (rejection SFT).")
    sys.exit(2)


if __name__ == "__main__":
    main()
