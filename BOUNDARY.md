# BOUNDARY.md — what gym is, what studio is, and which contains which

> Constitution layer, next to SPEC.md. Referenced by AGENTS.md and the skills.

## Article 0: containment direction

**studio contains gym. Never the reverse.** Stated both ways because direction-ambiguous
verbs caused a real incident: *studio ⊃ gym; gym ⊄ studio-container roles; gym never
contains studio.* "Extract the gym" means carving a component OUT OF studio — not
renaming or upgrading studio into gym.

Why this article exists: the 2026-07-21 refactor executed the extraction backwards at the
REPOSITORY level — the whole project (training code, skills, experiments) was committed
and pushed into a repo named `nekaise-gym`. Package-level separation was correct
throughout (isolation tests green); the inversion lived at repo-identity level, where no
test looks. See docs/REFACTOR-NOTES.md D10. The judgment rule: **a repository's name must
equal its contents, and gym's contents never include training code.**

## gym = the examination hall

Three jobs only: author tasks, grade, run exams.

    gym/tasks/       tasks: prompt + gold reference + difficulty_band + dev/frozen splits
    gym/verifiers/   grading: pure functions verify(prompt, response, meta) -> float —
                     the project's ONLY definition of right and wrong
    gym/runner/      exams: pull any model (vllm serve / Ollama / frontier API) through
                     the tasks and score it

It NEVER contains: training code, hyperparameters, experiment logs, model weights, or
anything with "train" as the verb. Health check: *could someone who has never heard of
studio use this to evaluate GPT?* If yes, it is clean.

## studio = the workshop

One job: train a model until it passes the examination hall.

    SPEC.md / BOUNDARY.md   constitution: algorithm card, ban list, null-hypothesis
                            rule, environment lock, this boundary
    configs/                five frozen stage configs (cpt/sft/rlvr/opd/agentic)
    studio/stages/          five training entry points; frozen-section drift refuses
    lib/                    training scaffolding, checkpoint provenance
    skills/                 the agent's operating manual
    experiments/            the ledger + archived history (incl. legacy DPO code —
                            archived status, ban-listed for new work)

It NEVER contains: task definitions or grading logic — it does `import gym` like a
third-party library. Health check: *would this code change if you swapped the model or
the training framework?* If yes, it is studio's.

## Future split

When gym graduates to its own repository, that repo contains ONLY the `gym/` package +
its tests + a README — no studio history, no training code, no skills.

## Instruction-writing norm

Tests cannot catch identity-level errors — 73 green tests coexisted with an inverted repo
topology. The only defense is at the instruction layer: when writing about container
relationships, never rely on direction-ambiguous verbs ("extract", "migrate", "set up");
state the relation bidirectionally ("A contains B; B does not contain A") or with ⊂.
