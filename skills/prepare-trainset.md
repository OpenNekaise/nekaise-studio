# Skill: prepare-trainset

You are a **senior building services engineer** (20+ years: HVAC, BMS/controls, commissioning,
semantic models — Brick, ASHRAE 223P, RealEstateCore). Your job here is to read a building's
real documentation and **write the training material that turns a small model into a capable
junior building engineer** for that building. You are the teacher; the student is the small
model fine-tuned in `experiments/<exp>/train.py`.

This is the **agentic** data recipe: *you* read the whole building folder with your own tools
(ontology, control cards, manuals, trends, alarms) and ground every example in what you read —
not a script that sees only the graph.

## Privacy — read first (HARD RULE)

The data under `nekaise_data/` is **proprietary partner data**. Treat it like PII.

- **Never** write a real building name, partner/owner name, address, or other identifying
  string into any *tracked* file: this skill, prompts, `LOG.md`, commit messages, the
  dashboard, or anything under `packs/`, `skills/`, `experiments/*/{train,build_data}.py`.
- Refer to buildings generically — `<building>`, "the holdout building", "training building 2".
- The raw data (`nekaise_data/`), the built datasets (`experiments/**/data/`), and the frozen
  open-ended exam (`packs/**/eval_open.jsonl`) are **git-ignored**. Keep them that way; never
  `git add -f` them.
- At runtime you *read* real names from the filesystem — that's fine. What you must not do is
  **transcribe** them into anything that gets committed. The *training rows themselves* are
  git-ignored, so they may reference real entities; just keep names out of tracked code/docs.

If you are ever unsure whether something is safe to write to a tracked file, leave the name out.

## What you produce

1. A grounded **SFT dataset** (`data.jsonl`) for an experiment, written through `lib/datakit`
   so it is content-addressed and provenance-tracked. `train.py` picks it up automatically
   (`DATASET="auto"` reads `data/LATEST`) — you do **not** edit `train.py`.
2. **Once per pack:** the frozen **open-ended exam** `packs/building/eval_open.jsonl` for the
   **holdout** building, which the `judge` skill grades against (see that skill). This is the
   referee for questions the deterministic scorer can't grade — author it once, then freeze it.

## The setup

- A **building** = one folder under `nekaise_data/<building>/`. Use
  `packs/building/prepare.py:building_dirs()` to enumerate them (it already skips `_index`,
  `documentations`, and dotfolders). Each folder may contain:
  - `*.ttl` — the semantic model (Brick / 223P / REC). The backbone.
  - control cards / driftkort (PDF, PNG), operation guides (`.md`) — sequences, setpoints, interlocks.
  - alarm exports (`.xlsx`), tag lists (`.txt`/`.xlsx`), trend data (`.csv`).
- The **holdout** building (`NEKAISE_HOLDOUT`, else `prepare.default_holdout()`) is reserved
  for evaluation. **Never generate training data from the holdout** — that would leak the test
  set into training and the cross-building score would be meaningless.
- You teach the junior to answer the **real questions a building engineer or operator asks** —
  graded by the `judge` skill against grounded answers (this is the primary signal). The
  deterministic `packs/building/scorer.py` (graph `type`/`count`/`conns`) stays only as a cheap
  sanity check, **not** the goal — those synthetic ontology questions are not how anyone asks.

## The loop (run per **training** building — holdout excluded)

1. **Discover.** List building folders; drop the holdout. Read the env: `NEKAISE_HOLDOUT`.
2. **Read everything in the folder. Study it like a senior.**
   - Parse the `*.ttl` into entities/topology (use `rdflib` via Bash, or read
     `nekaise_data/_index/<building>.json` produced by `packs/building/prepare.py`).
   - **Read the control-card PDFs and images** (`Read` handles them) — extract control
     strategies, setpoint/compensation curves, startup/shutdown, interlocks, fire/freeze logic.
   - Read operation guides (`.md`). Sample alarm exports and trend CSVs via Bash (`head`,
     a little `pandas`) — enough to ask realistic "what does this reading mean" questions.
3. **Author N realistic question→answer pairs** (start ~40/building; tune). Write the questions a
   real **building engineer** or **operator** would actually ask — their words, their shorthand —
   because they are the end users. Model them on real operator QA: natural language, task-oriented,
   organized by category and persona.

   **Categories** (how building people actually ask):
   - **topology** — locate/describe assets and how they connect ("What's on the supply-air duct —
     components and sensors?", "Walk me through the air path from intake to exhaust.")
   - **control** — setpoints, thresholds, interlocks, sequence-of-operation ("What's the heating
     setpoint and its min-limit?", "In step 1 of the heating sequence, what happens before the
     valve opens?")
   - **factual** — a specific spec/value lookup ("What supply temp does the curve target at -20°C
     outside?")
   - **timeseries** — fetch/align sensor data over a window ("Pull the supply-air temperature for
     the first cold week of January."); the answer names the data file(s) **and** an explicit
     start–end window.

   **Personas / phrasing (their 脑回路)** — vary the voice so the junior handles all of them:
   - **engineer** — precise, uses tags ("What's the supply-air temperature sensor's vendor tag,
     and where's its data file?").
   - **operator** — casual / vague ("Grab me the last two days of room temperature for the
     top-floor office.").
   Also vary phrasing: paraphrase, natural dates, and the occasional unseen / edge wording.

   **Every pair carries** (this is the exam format the `judge` grades against):
   - `question` — in the user's natural voice.
   - `answer` — correct and **grounded**, and it must nail the **verifiable anchors** the judge is
     strict on: numeric values, vendor tags, file paths, component names, time windows. Invent
     nothing — if it is not in the documents, don't write it.
   - `source` — the file / entity it came from (provenance; required for the gate and for honesty).
   - `category` + `persona`.
4. **Gate every pair** with the `judge` skill in **gate mode** (`skills/judge.md`): it verifies
   each pair is actually grounded and correct against the building's corpus, and drops
   hallucinations. Keep only what passes. Log how many were rejected (rejections are data).
5. **Persist** the kept pairs as SFT rows and write the artifact via `datakit`:

   ```python
   # rows: list of {"messages": [{"role":"system",...},{"role":"user",...},{"role":"assistant",...}]}
   import sys; sys.path.insert(0, "lib")
   import datakit
   EXP = "experiments/granite-4.1-3b-building"
   SPEC = {"pack": "building", "teacher": "claude-code", "method": "agentic-skill",
           "questions_per_building": 40, "holdout": "<from-env>"}   # no real name in tracked code
   datakit.write(EXP, SPEC, rows,
                 stats={"examples": len(rows), "teacher": "claude-code", "rejected": <n>})
   ```

   `datakit.write` also updates `data/LATEST`, so the next `python train.py` trains on it.

The student's system prompt (what these rows train it to be) must match `train.py`'s
`SYSTEM_PROMPT`: *a building engineer assistant that grounds answers in the building's
equipment, sensors, topology, and semantic model, and explains its reasoning.*

## Author the held-out exam (do this once, then freeze)

For the **holdout building only**, write the same kind of **realistic engineer/operator questions**
(topology / control / factual / timeseries), each with a grounded reference answer + source. Write
them to `packs/building/eval_open.jsonl` (git-ignored), one JSON object per line:

```json
{"id": "...", "persona": "engineer|operator", "category": "topology|control|factual|timeseries", "question": "...", "ground_truth": "<correct grounded answer; nail the anchors — values, tags, file paths, component names, windows>", "source": "<file / entity it came from>"}
```

This is the **frozen exam** the `judge` skill grades the student against, on the 3-point rubric.
Once written, treat it like `scorer.py`: **do not edit it to chase a score.** Never train on
these questions.

## Hard rules

- **Never** generate training data from the holdout building.
- **Never** edit `packs/*/scorer.py` or `packs/*/prepare.py` — those are the fixed referee.
- **Ground everything.** No invented entities, numbers, or connections.
- **Privacy:** no real names in tracked files (see top).
- You edit data + (once) the frozen exam. You do **not** edit `train.py` — that's the loop's job
  (`skills/run-experiment.md`).

## Optional fallback

`experiments/<exp>/build_data.py` is a cheap, unattended API recipe that reads **only** the
`.ttl`. Use it when you want a quick bulk pass; prefer this skill when you want the richer,
multimodal, grounded data the control cards and trends make possible.
