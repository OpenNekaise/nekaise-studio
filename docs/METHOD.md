# Method: driving the loop toward teacher parity

A playbook for using Nekaise Studio on **any** pack. It's general — no dataset-specific numbers.
Your run-by-run results live in each experiment's `LOG.md` (git-ignored, private to you).

## The goal is teacher parity, not a fixed score

The bar is not "reach X% accuracy." It is: **a small model, under the same harness, reaches the
accuracy of a frontier teacher.** You are closing a *gap*, so you must measure both ends of it.

## 1. Diagnose before you optimize

Two measurements come *before* any fine-tuning, and they save you from optimizing the wrong thing:

- **Measure the teacher (the goal line).** Run a frontier model over the same input the student
  will get, scored by the pack's fixed referee. That score is the ceiling you're chasing.
- **Measure the untrained base (attribution).** Does fine-tuning even help? Modern open instruct
  models (e.g. Granite, Qwen, Llama) already ship strong tool-use / reading ability — the base may
  already be good. If your fine-tune ≈ the base, your data/recipe isn't the lever; if your fine-tune
  is *below* base, your data is actively hurting. Never assume SFT helps — prove it against base.

## 2. Open-book = the data in hand + retrieval

- Give the model the building's **real data** — control cards, tables, notes, and the ontology.
  The ontology is **one source, not the center**; most buildings don't have a clean one.
- A small model can't hold a whole building. **Retrieve the relevant slice per question**
  (`lib/corpus.retrieve`). This fits the context window, keeps training cheap, and matches how the
  model is actually deployed (retrieve, don't dump the whole building).
- **Train and eval in the same retrieved-context format** — fairness, and the read-then-answer
  skill transfers.

## 3. Distill the teacher, task-aligned

- Build training data by having the **teacher answer the referee's own questions** over the
  retrieved context, and **keep only the answers the scorer marks correct** (rejection sampling).
  The targets are concise and verifiably right.
- This teaches *read-then-answer in the eval's shape*. Do **not** train on verbose reasoning prose
  for an exact-match metric — it teaches the wrong thing.

## 4. Pitfalls (each one cost a run to learn)

1. **Closed-book eval caps the score.** If the answer needs data the model can't see, no amount of
   training helps. Give the model the data at inference.
2. **Heavy SFT on narrow / verbose prose → catastrophic forgetting** of the base's reading skill.
   Keep SFT light and task-aligned; check every run against the base bar.
3. **Full-corpus-in-every-example training is compute-bound.** Long sequences → few steps fit the
   time box → undertrained. Retrieval (short slices) fixes it.
4. **Don't center the ontology.** It's one representation of heterogeneous real data.
5. **Some metrics are context-capped** (e.g. counting *all* instances of a class from a slice —
   even the teacher can't). Route those to an aggregation / judge track; don't fight them with
   retrieval.

## The cadence

Propose **one** change → rebuild data if needed → train (time-boxed) → score on the fixed referee
→ keep if it beats the best, revert otherwise → log it. **Compare only within the same harness** —
when you change the harness (closed-book → open-book → retrieval), past numbers are not comparable;
re-measure base and teacher on the new harness.
