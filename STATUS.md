# STATUS — current phase and targets

> Durable procedure lives in `SPEC.md` (the constitution), `skills/`, and `AGENTS.md`;
> this page is the part that CHANGES. Last update: 2026-07-21.

## Phase: POST-REFACTOR SHAKEDOWN

The REFACTOR.md rebuild (R1–R12; deviations in docs/REFACTOR-NOTES.md) landed IN THIS
REPO — **nekaise-studio is the one canonical repository**. The transient nekaise-gym
repo is archived; containment direction is now constitutional (BOUNDARY.md Article 0:
studio contains gym, never the reverse). The algorithm card, ban list, null-hypothesis meta-rule, and
environment lock moved to **SPEC.md** — STATUS no longer carries them. Architecture:

- **gym/** — tasks + verifiers + runner: the single definition of correct. One-way
  dependency (studio → gym), enforced by tests.
- **studio/stages/** — `python -m studio.stages.<cpt|sft|rlvr|opd> --config
  configs/<stage>.yaml`; frozen sections asserted against the card.
- **studio/tools/** — explog (schema'd LOG), variance_check (noise band),
  crystallize_gate (≥2-experiment admission).

Mainline: **1B-class (Qwen3.5-0.8B → product name Nekaise-1B)**; 3B parked as an optional
verification line (re-verify any 3B conclusion on 1B before adopting — learnability gap).

## Best known (corpus_probes dev, absorption)

| model | base | best treatment |
|---|--:|---|
| Qwen3.5-0.8B | 0.1148 (transfer 0.0968) | **0.1318** (transfer 0.1075) — LoRA r32 CPT smoke (pre-refactor run 1, cosine) |

nekaise-bench stays DECOUPLED: milestone-only external referee (`tools/eval_bench.py`),
version-stamped, never keep/revert.

## Next levers (strict order — R5 before any new experiment)

1. **HUMAN: install + pin vLLM** (SPEC §4; requirements.txt placeholder). The runner and
   RLVR rollout paths are wired but unexercised until then.
2. **variance_check on the mainline CPT config** — 3 seeds, write the noise band, then
   re-audit the pre-refactor conclusion (0.1318 vs 0.1148) against it;
   `crystallize_gate --audit` marks affected local skills `unverified`.
3. **Card-migration CPT run** — `studio.stages.cpt` (WSD, frozen) replacing the cosine
   smoke run; re-baseline under the card. One migration run, then the only CPT knob is
   the data recipe.
4. **Data recipe: diverse forms** — paraphrase augmentation vs raw repetition on
   absorption probes (the Allen-Zhu lever; expected main gain).
5. **calibrate.py first pass** — model ladder over corpus_probes dev; bands feed the
   RLVR task sampler (20–80% window) when RLVR unlocks.
6. **Corpus capacity & task-set build-out** — the agent's free energy goes here, not to
   hyperparameter search (SPEC §5).

## Recently retired

- **STATUS-resident algorithm card** → moved to SPEC.md §1 (constitution layer) with the
  ban list, null-hypothesis rule, and environment lock (refactor R2, 2026-07-21).
- **HF `generate()` eval paths** → gym runner (vLLM/OpenAI-compatible); old versions in
  `attic/` (R4).
- **packs/ as logic** → thin shims over gym; verifier truth lives in gym/verifiers only
  (R1).
- **2B / granite-4.1-3b levers** (1B refocus, 2026-07-21) — parked, not abandoned.
