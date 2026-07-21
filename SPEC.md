# SPEC.md — the constitution

> **Immutable bootloader layer.** This file defines what the agent may and may not do when
> training models in this repo. The `run-experiment` skill's adjudication logic references
> this file explicitly; every keep/revert decision is bound by it. Changing ANYTHING here is
> a human-reviewed spec change, never a loop move. Last human revision: 2026-07-21.

Development and experiments are executed by an agent; an agent's behavior is determined by
readable files. Consensus that is not written down will be re-invented or violated by the
next agent instance. Hence this constitution. Its sibling **BOUNDARY.md** defines what
gym and studio each are and fixes the containment direction (Article 0: studio contains
gym, never the reverse) — equally binding.

## 1. Algorithm card

One stage = the most boring algorithm that works, run longer than anyone. Rationale: the
experiment loop is agent-run, and complex recipes (multi-stage, curricula, dynamic
sampling) create a high-dimensional search space in which a one-change-per-run loop
oscillates without converging. A minimal recipe means few knobs, clean attribution per
run, and LOG conclusions that accumulate. Minimalism here is a structural requirement,
not an aesthetic.

| stage | implementation | agent-movable knob (the ONLY one) | frozen |
|---|---|---|---|
| CPT | Unsloth standard training + WSD schedule (warmup–stable–decay) | data recipe (domain/replay mix, annealing subset) | optimizer, schedule shape, sequence length |
| SFT | TRL `SFTTrainer`, plain cross-entropy on judge-gated teacher data | data recipe (synthesis strategy, filter threshold, mix ratio) | loss function — no variants |
| RLVR | TRL `GRPOTrainer` + gym verifier, JustRL hyperparameters verbatim (Appendix A) | task-set composition + verifier | ALL hyperparameters; clip-higher is the only stabilizer |
| OPD | TRL `GKDTrainer`, sampled-token variant (fully on-policy) | teacher choice (own post-RL checkpoint / privileged-info self-distill) | divergence settings; no top-k variants |
| Agentic | trajectory-level rejection SFT → short-horizon GRPO (outcome reward) | task-set composition | 5–15 step cap; no process rewards; no dense supervision over long trajectories |

Engineering note: RLVR and OPD are the same training-loop *type* — OPD swaps the verifier
reward for a teacher log-ratio. Five stages, three loop types (pretrain-style, SFT-style,
GRPO-style). Stage entry points are `python -m studio.stages.<stage> --config
configs/<stage>.yaml`; each stage's `frozen:` config section is asserted in code against
this card and refuses to run on drift.

## 2. Ban list (as binding as the card)

Default-banned from the loop. Any of these enters ONLY when the dashboard shows the
specific pathology it treats, and with explicit human approval:

- curriculum learning
- dynamic sampling / difficulty filtering during training (JustRL shows unnecessary; the
  *calibration ladder* that filters the task POOL by measured pass rate is data recipe,
  not banned)
- length penalties — sole exception: if the overlong-response share measurably spikes, add
  JustRL's two-stage length schedule
- PPO + critic (GRPO removed the value network; don't invite it back)
- process reward models
- MCTS / search augmentation
- DPO stacking
- model merging

Each of these is something the literature claims necessary and the JustRL line of evidence
shows is a pseudo-need.

## 3. Null-hypothesis meta-rule

> **Before any algorithm change is declared effective, it must beat the control: the same
> plain baseline trained for the same additional compute.** If both rise equally, the
> change is judged ineffective and reverted.

"Adding X gained 2 points" only counts if "no X, equal extra steps" does not also gain
2 points. The control run is logged like any run. Data-recipe changes (`build_data.py` /
the `data:` section) compete at equal wall-clock as usual — no control needed. Under a
frozen card, algorithm changes are rare by design, so the doubled cost of the control is
acceptable precisely because it is seldom paid. The keep/revert additionally requires the
effect size to exceed the measured noise band (see `studio/tools/variance_check.py`).

## 4. Environment lock

The training-stack version trio is PINNED (see `requirements.txt`) and validated together:

    unsloth==2026.6.9   trl==0.24.0   transformers==5.5.0

vLLM joins the lock the moment it is installed (rollout engine, see R4): pin the exact
version next to the trio. **The agent has no authority to change dependency versions.**
Installing, bumping, or removing any dependency is a human-approved operation: bump the
trio together, re-run the gsm8k bootstrap baseline to revalidate, then commit the bump.
(`uv` is available for lockfile generation once `pyproject.toml` grows a `[project]`
table — until then `requirements.txt` pins are the lock; see docs/REFACTOR-NOTES.md.)

## 5. Explicitly not doing

- **No veRL/OpenRLHF migration.** Triggers that reopen the question: (a) multi-GPU /
  multi-node training; (b) multi-turn agentic RL requiring online harness rollout; (c)
  throughput still hard-bottlenecked after vLLM colocate is fully enabled. Even then,
  migrating only the agentic stage is on the table — not the whole stack.
- **No technique from the ban list** without dashboard-shown pathology + human approval.
- **No hyperparameter-search experiments at the current corpus scale (~3.5M tokens)** —
  conclusions don't extrapolate. The agent's free energy goes to corpus capacity and
  task-set build-out.

## 6. Where complexity IS allowed (asymmetry principle)

Complexity saved on algorithms transfers to three places where it compounds:

1. **Verifiers** (`gym/verifiers/`) — extend from anchor-recall toward the hard-verifiable
   tier: executable SPARQL (result-set F1), numeric tolerance, sandboxed unit tests. The
   judge's share of reward trends DOWN with every version.
2. **Data recipes** — the `data:`/`build_data.py` search space may be arbitrarily rich;
   the data mix is the one dimension this philosophy encourages iterating.
3. **Measurement** — the dashboard tracks entropy, response-length distribution, grad
   norm, per-difficulty-band pass rate (OPD: overlap ratio), and anchor-density
   distribution alongside reward. Keep/revert reads mechanism metrics, not just the
   score. Defense against reward hacking is measurement, never algorithm patches.

## 7. Terminology

- `gym/runner/` is the **runner** (batch generation + evaluation orchestration).
- The word **harness** is reserved for the agent runtime (opennekaise / nekaise-edge).
  Do not use it for anything inside `gym/` or `studio/`.
- Container identities and the gym/studio boundary live in **BOUNDARY.md** (Article 0:
  containment direction; the examination-hall / workshop definitions).

---

## Appendix A — JustRL → TRL hyperparameter mapping (RLVR stage)

Source: JustRL paper (arXiv:2512.16649, Table 2), veRL implementation. These are the
FROZEN values for `studio/stages/rlvr.py`. Start from these; do not search.

| JustRL (veRL) | value | TRL `GRPOConfig` field | note |
|---|---|---|---|
| learning rate | 1e-6, constant | `learning_rate=1e-6`, `lr_scheduler_type="constant"` | |
| train batch size | 256 prompts | `per_device_train_batch_size × gradient_accumulation_steps × world_size = 256 / num_generations` … | semantic target: 256 prompts/step effective; batch GEOMETRY scales to hardware, effective size does not |
| PPO mini/micro batch | 64 / 1 per GPU | gradient accumulation | geometry, not semantics |
| rollout N (group size) | 8 | `num_generations=8` | |
| temperature | 1.0, fixed | `temperature=1.0` | no adaptive scheduling |
| clip ratio range | [0.8, 1.28] | `epsilon=0.2`, `epsilon_high=0.28` | clip-higher (DAPO-style), the ONLY stabilizer |
| KL loss | none | `beta=0.0` | TRL's default beta is nonzero — MUST set 0.0 explicitly |
| entropy regularization | none | (TRL default: none) | do not add |
| max prompt length | 1k | `max_prompt_length` | task-window parameter, not a stability parameter — set per task set |
| max response length | 15k (math CoT) | `max_completion_length` | ditto; our short-answer tasks use far less |
| reward | binary outcome, rule-based verifier | gym verifier via reward wrapper | keep binary/simple; reward shaping is a spec change |
| advantage estimator | GRPO (group-normalized) | TRL default GRPO loss | keep defaults; `scale_rewards` at TRL default |

Deviations legend: batch geometry and max lengths are hardware/task-shaped and may differ
from the paper's literal values at CONSTANT semantics (effective prompts/step, enough
window for the task). Everything in the stability column (lr, schedule, clip, KL, entropy,
temperature, group size) is copied verbatim and frozen.
