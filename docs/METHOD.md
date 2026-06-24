# Method: driving the loop toward teacher parity

A playbook for using Nekaise Studio on **any** pack. It's general — no dataset-specific numbers.
Your run-by-run results live in each experiment's `LOG.md` (git-ignored, private to you).

## The goal is teacher parity, not a fixed score

The bar is not "reach X% accuracy." It is: **a small model, under the same harness, reaches the
accuracy of a frontier teacher.** You are closing a *gap*, so you must measure both ends of it.

## 1. Diagnose before you optimize

Two measurements come *before* any fine-tuning, and they save you from optimizing the wrong thing:

- **Measure the teacher (the goal line).** Run a frontier model over the same input the student
  will get, scored the same way (the realistic exam, judged by anchors — §3). On *realistic*
  questions the teacher is far from perfect (it's capped by retrieval too), so this number is the
  honest ceiling — not 100%.
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

## 3. Ask what the user asks; grade by anchors

The metric must be what the building's real users — **engineers and operators** — actually ask, not
synthetic ontology trivia ("what Brick class is X"). Have the **teacher author realistic questions**
in the users' own voice (no fixed taxonomy; span lookups, structure, control logic, **diagnostics**,
data-fetch; tag a free-form `intent` for coverage). Each question carries:
- a grounded `answer`, and
- `anchors` — the specific facts the answer MUST contain (numeric values, vendor tags, file paths,
  component names, time windows, checklist items).

**Grade by anchor fraction.** A frontier **judge**, blind, scores each answer = the fraction of its
anchors present (strict on the fact, lenient on phrasing). Single-anchor questions are pass/fail;
multi-part answers get honest partial credit. This is mostly a strict matcher with phrasing
tolerance — far more reproducible than holistic 1/0.5/0 judging, and it measures real usefulness.
(A cheap deterministic scorer can stay as a sanity check, but it is not the goal.)

## 4. Build training data the same way

Training demos are the teacher's realistic Q&A over **retrieved** context (the answer is the gold).
The gate keeps only fully-anchored (`score==1.0`) demos. The student learns to read the retrieved
data and answer like an engineer/operator. **Balance demos across intents** — the metric is only as
strong as its weakest, least-trained intent (e.g. data-fetch questions need the file-path/index in
the retrieved slice, or they cap everyone, teacher included).

## 5. Pitfalls (each one cost a run to learn)

1. **Closed-book eval caps the score.** If the answer needs data the model can't see, no amount of
   training helps. Give the model the data at inference.
2. **Heavy SFT on narrow / verbose prose → catastrophic forgetting** of the base's reading skill.
   Keep SFT light and task-aligned; check every run against the base bar.
3. **Full-corpus-in-every-example training is compute-bound.** Long sequences → few steps fit the
   time box → undertrained. Retrieval (short slices) fixes it.
4. **Don't center the ontology.** It's one representation of heterogeneous real data.
5. **Synthetic questions are not the goal.** Ontology fill-in-the-blanks ("what class / how many /
   what connects to what") are not how users ask and can be a *data* dead-end (e.g. a question kind
   with zero training signal can't be learned, no matter the recipe). Author real user questions.
6. **A first realistic SFT is often a wash that *redistributes*** — it lifts the demo-heavy intents
   and forgets the rest. That's an under-training / balance signal, not a dead end: balance the
   intents and add steps with care (watch the forgetting tradeoff).

## The cadence

Propose **one** change → rebuild data if needed → train (time-boxed) → score with the **judge** on
the realistic exam (deterministic scorer as sanity check) → keep if it beats the best, revert
otherwise → log it. **Compare only within the same harness** — when you change the harness
(closed-book → open-book → retrieval → question style), past numbers are not comparable; re-measure
base and teacher on the new harness.
