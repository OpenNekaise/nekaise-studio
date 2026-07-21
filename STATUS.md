# STATUS — current phase and targets

> Durable procedure lives in `skills/` and `AGENTS.md`; this page is the part that CHANGES.
> Update it when the phase, metric, or priorities shift. Last update: 2026-07-21.

## Phase: AGENTIC-CPT (pure) — 1B-class refocus

The reform phase: the studio's identity is **agentic LLM training** — the agent runs the
loop; CPT/distill/RL are tools. All effort concentrates on the **1B-class base
(Qwen3.5-0.8B)** until the algorithm card below is proven end-to-end. Larger bases
(Qwen3.5-2B, granite-4.1-3b) are parked: a minimal recipe is scale-invariant (JustRL's
recipe held across bases with one set of hyperparameters), so the loop we prove small is
the loop we scale later — unchanged.

Current workload: **pure CPT** (no teacher, no QA pairs) on the curated corpus slice,
measured by the studio-owned `corpus_probes` pack.

- Active experiment: `experiments/agentic-cpt/`
- Loop metric: `python tools/eval_probes.py --checkpoint <stage> --split dev`
  (= absorption-probe accuracy; `transfer` reported as diagnostic; frozen split milestone-only)
- **nekaise-bench is DECOUPLED**: milestone-only external referee, never keep/revert
  (`tools/eval_bench.py`; results record the bench dataset version — v4 at reform time —
  and are only comparable within one version)
- Campaigns: `python tools/campaign.py <spec.json>` (idempotent resume)

## Algorithm card (frozen spec)

The loop is run by an agent, and an agent's worst enemy is a recipe with many knobs.
Complex recipes (multi-stage, curricula, dynamic sampling) mean a high-dimensional search
space in which a one-change-per-run loop oscillates without converging; a minimal recipe
means few knobs, clean attribution per run, and `LOG.md` conclusions that accumulate.
JustRL-style minimalism is an aesthetic choice for a human researcher; for an agentic
studio it is a structural requirement.

One stage = the most boring algorithm that works, run longer than anyone. Per stage the
**one movable knob** is what the loop may vary; everything in **frozen** is spec — changing
it is a human spec-review, not a loop move.

| stage | algorithm | the one movable knob | frozen |
|---|---|---|---|
| CPT | plain next-token + WSD schedule (warmup–stable–decay) | data recipe (domain/replay mix, annealing subset) | optimizer, schedule shape, seq len |
| SFT | plain cross-entropy on judge-gated teacher data | data recipe (synthesis strategy, filter threshold, mix ratio) | loss function — no variants |
| RLVR | GRPO, JustRL config verbatim: single stage, fixed hyperparameters, binary/simple verifiable reward, clip-higher as the only stabilizer | task-set composition + verifier | all hyperparameters (start from published JustRL values, no search) |
| OPD | sampled-token on-policy distillation inside the SAME GRPO loop (reward = log q − log p) | teacher choice (own post-RL checkpoint / privileged-information self-distill) | no top-k, no extra divergence variants |
| Agentic | trajectory-level rejection SFT → short-horizon GRPO (outcome reward, 5–15-step tasks) | task-set composition | no process rewards, no dense supervision over long trajectories |

Engineering dividend: RLVR and OPD share one training loop — OPD is GRPO with the verifier
swapped for a teacher log-ratio (a reward-function config change, not new code). Five
stages, three loop types total (pretrain-style, SFT-style, GRPO-style): the minimalism
cashes out at the code layer too.

## Ban list (as binding as the card)

Default-banned from the loop unless the dashboard shows the specific pathology each one
treats: **curriculum learning; dynamic sampling / difficulty filtering** (JustRL shows
unnecessary); **length penalties** (sole exception: if overlong-response share measurably
spikes, add JustRL's two-stage length schedule); **PPO + critic** (GRPO removed the value
network — don't invite it back); **process reward models; MCTS / search augmentation; DPO
stacking; model merging**. Each is something the literature claims necessary and the
JustRL line of evidence shows is a pseudo-need.

## Meta-rule: beat the null hypothesis

Any **algorithm** change must beat the control "the same plain baseline trained N more
steps at equal compute" before it is kept — see `skills/run-experiment.md` §Rules for the
binding wording. JustRL's whole story is that complex tricks never beat this control.
Data-recipe changes (`build_data.py`) compete at equal wall-clock as before. Under the
frozen card, algorithm changes should be rare — so the doubled cost of the control run is
acceptable exactly because it is seldom paid.

## Where complexity IS allowed (asymmetry principle)

The complexity budget saved on algorithms transfers entirely to three places where
complexity compounds (algorithm-side complexity doesn't):

1. **Verifiers** — extend from anchor-recall toward the hard-verifiable tier (executable
   SPARQL, numeric tolerance, sandboxed tests); judge share trends down over time.
2. **Data recipes** — `build_data.py`'s search space may be arbitrarily rich; the data mix
   is the one dimension this philosophy encourages iterating.
3. **Measurement** — dashboard tracks entropy, response-length distribution, grad norm, and
   per-difficulty-band pass rate (OPD: overlap ratio) alongside reward; keep/revert reads
   mechanism metrics, not just the score. Anchor-recall GRPO keeps a reward-hacking monitor
   (anchor-stuffing detection): defense by measurement, not by patching the algorithm.

## Best known (corpus_probes dev, absorption)

| model | base | best treatment |
|---|--:|---|
| Qwen3.5-0.8B | 0.1148 (transfer 0.0968) | **0.1318** (transfer 0.1075) — run 1: LoRA r32 CPT smoke, 1200-doc slice, lr 1e-4 cosine |

Historical note: all pre-reform numbers in `docs/RESULTS.md` were measured on nekaise-bench
**v1** (677 items) as the loop metric. The bench has since been rebuilt (v4: 311 items,
hardened to 27B≈0.45) and decoupled; those numbers are milestone history, not comparable
baselines. Milestone re-baselines on bench v4 happen at the next phase gate.

## Next levers (one change per run)

1. **Card migration run** — re-run the current best CPT treatment under the card's frozen
   spec (WSD schedule replacing run 1's cosine; frozen optimizer/seq-len). One migration
   run to re-baseline; from then on the only CPT knob is the data recipe.
2. **Data recipe: diverse forms** — paraphrase augmentation (`augment_corpus.py` pattern)
   vs raw repetition, judged by absorption probes. The Allen-Zhu lever; expected main gain.
3. **Data recipe: mix & annealing** — domain/replay ratio, annealing subset, corpus slice
   size (N_DOCS 1200 → more/denser), topic reweighting (probes are topic-tagged).
4. **Corpus capacity & task-set build-out** — at ~3.5M tokens, later-stage (RLVR/agentic)
   conclusions would be noise; the agent's free energy goes to corpus production and to
   building the verifiable task sets those stages will need, not to algorithm search.
5. **Milestone gate** — when a treatment holds on frozen probes, run the bench (v4)
   milestone: base vs best checkpoint, paired flips, `--milestone`.

## Recently retired

- **2B / granite-4.1-3b levers** (1B refocus, 2026-07-21) — parked, not abandoned: prove
  the frozen card on the 1B-class base first; the card is scale-invariant by design, so
  scaling back up later changes the card not at all.
- **nekaise-bench as loop metric** (decoupling reform, 2026-07-20) — superseded by
  `packs/corpus_probes`; bench is milestone-only now. Rationale: a hardened external exam
  is the right *referee* and the wrong *training-loop signal* (sub-4B floors ≈0.05–0.16 on
  bench v4; probe accuracy sits in the sensitive range and moves densely under CPT).
- Local skill `verify-holdout-before-judge` — superseded by the in-code guard
  (`prepare.require_holdout_matches_exam`, commit f0cbab6) + doctor check.
