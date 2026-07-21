# [DRAFT — file against OpenNekaise/opennekaise] Headless programmable mode

> Cross-repo dependency of nekaise-gym's agentic training stage (REFACTOR.md R11).
> Not yet filed; this draft is the interface definition the studio codes against.

## Ask

A headless, programmable entry point: **one command runs one task and returns structured
output**, no TUI, no interactive session.

```bash
opennekaise run-task \
  --workspace <dir>          # task files (the agent's world)
  --task-file <task.json>    # {"prompt": ..., "max_steps": N}
  --model <ollama-name|endpoint>
  --out <result.json>        # structured: see below
  --max-steps N --timeout-s S
```

`result.json` (exit 0 on completion, nonzero on infra failure — task failure is a result,
not an error):

```json
{
  "answer": "<final answer / produced artifact path>",
  "steps": [{"action": "...", "input": "...", "observation": "..."}],
  "n_steps": 7,
  "wall_s": 41.3,
  "truncated": false
}
```

## Why (three consumers, one interface)

1. **gym exams** — evaluate an agent (not just a chat model) on outcome-verified tasks:
   the runner shells out per task, verifies `answer` with the gym verifier.
2. **RL rollout** — short-horizon GRPO (5–15 steps, outcome reward) samples trajectories
   through the SAME runtime the product uses; no simulator drift.
3. **CI regression** — golden tasks per release; structured output diffs cleanly.

## Constraints from the training side

- Steps must be capped by the caller (`--max-steps`); the run must hard-stop and report
  `truncated: true` rather than hang.
- Deterministic workspace: everything the agent may touch lives under `--workspace`;
  no writes outside it.
- Token/latency budget per task (`--timeout-s`) enforced runtime-side.
- Model swappable per call (`--model`): candidate checkpoints are served via
  Ollama/vLLM and compared against the incumbent.

## Until this exists

nekaise-gym ships a toy in-process substrate (gym/tasks/toy_agentic.py +
studio/stages/agentic) that validates the trajectory → verifier → score chain; phase B
(short-horizon GRPO through the real runtime) is blocked on this interface
(studio/stages/agentic/train.py exits 2 with a pointer here).
