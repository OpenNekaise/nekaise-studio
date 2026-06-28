# Skill: crystallize-skill

You have just finished an experiment or investigation. Before moving on, decide whether it
produced a **reusable, validated lesson** worth saving as a skill — and if so, write it.

This is the **mutation** half of how Nekaise Studio improves itself. The studio gets better not
when a model finishes training, but when a *validated finding* becomes a skill the next run reads.
Skipping this step is how the same wrong turns get re-discovered forever.

## When to crystallize

Save a finding as a skill only if ALL of these hold:

- **Validated, not a hunch.** It was confirmed by the eval harness — a `METRIC` move, a
  `building_judge` / `domain_quiz` result, a measured speedup or memory number — not by how
  plausible it sounds.
- **Reusable beyond this run.** It will apply to future datasets and experiments, not just this one.
- **Not already captured.** No existing skill says it. If one says something weaker, vaguer, or now
  wrong, **update that skill** instead of adding a new one.

Do NOT crystallize:

- **Unvalidated ideas** → they go in the experiment's `LOG.md` as hypotheses, not in a skill.
- **Building-specific facts** (a tag value, a setpoint, a file path) → that is *data*; it belongs
  in `nekaise_data/`, never in a skill.
- **One-off results** ("run 7 scored 0.54") → a log entry, not a lesson.

## Where it goes

- New emergent skills go in **`skills/local/<name>.md`** — git-ignored, this machine's, written
  freely by you. They are not pushed; everyone's local skills differ.
- A local skill earns its way into the shared **`skills/`** kernel only by **promotion**: it must
  survive the eval harness and a human-reviewed PR. Never edit the core `skills/*.md`, `packs/*`,
  or `lib/*` to encode a fresh finding — propose it.

## How to write it

Same format as the core skills: `# Skill: <name>`, then tight, imperative instructions to a future
agent. Include the **evidence** that validated it (which metric moved, by how much, on what) so a
later `prune-skills` pass can re-judge it. Keep it short — a skill is a lever, not an essay.

At the end of every `run-experiment` cycle, ask: *"what reusable, validated lesson did this
produce?"* If the answer is real, crystallize it before you forget — then consider whether
`prune-skills` is due.
