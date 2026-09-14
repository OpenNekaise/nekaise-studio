# CoAPT in Nekaise Studio

Co-Adaptive Pretraining and Tuning creates teaching material in response to the current
student's actual behavior. The teacher has full educational authority; the worker executes
and records its decisions. The separate orchestrator repairs runtime failures.

## Teacher authority and memory

The teacher can inspect all teaching history in this workspace, across every campaign and
continuation: source passages, prompts, actual attempts, corrections, prior questions and
answers, grades, training metrics, dataset/checkpoint metadata, notes and operational logs.
The archive tool provides pagination and search with no recency cutoff. Latest teaching
notes are injected as a starting point; older notes and full artifacts remain accessible.

The teacher chooses sources, topics, lesson count, order, language, difficulty, teaching
style, CPT/QA composition, the exact prompt shown to the student, raw readings, historical
review examples, token shares, passes over the dataset, evaluation design and next actions.
Configured lesson counts, question counts, source prefix and mixture are suggestions.
Explicit operator lifecycle/resource budgets remain execution constraints.

Set curriculum train_epochs=0 to preserve weights for this round while still preparing
lessons and conducting assessment. Positive train_epochs enables training. The separate
config train_steps=0 means automatic update counting from tokens and passes; it does not
mean zero training. A positive train_steps overrides the update count of enabled training.
The worker executes the structured fields; notes explain those decisions.

The teacher may use its expertise, combine documents and create its own explanations or
exercises. It records sources it actually uses; teacher-authored tasks may have none.
Source eligibility and content hashes establish provenance, not teaching correctness.
There is no second teacher gate, exact-quotation admission test or fixed score threshold.
Only the teacher decides whether its corrected lesson should enter training.

## Student-conditioned teaching

1. The teacher consults history and chooses a curriculum, reading/review material and mix.
2. The student answers the teacher's exact prompts using its current checkpoint.
3. The teacher inspects those real attempts, creates corrected teaching text and decides
   which examples to use. Errors/evidence fields remain observations; explanations intended
   for learning must be in training_text itself. The teacher supplies that exact sequence,
   including any task or context needed; the worker does not rebuild or reformat it.
4. The worker freezes exact tokenized samples, lineage, source snapshots and realized token
   shares. It trains the current student with the recorded optimizer recipe. Diagnostic-only
   rounds preserve the parent checkpoint and produce no invented training metrics.
5. The teacher chooses an online assessment, with references/rubrics recorded before the
   student answers. It grades actual answers and decides which concepts need practice.
6. The teacher writes durable student notes and the next teaching strategy, then chooses
   continue, pause or complete. The next round starts from that retained checkpoint.

CPT material is usually self-contained prose. QA material includes a task, context as
needed and its answer, in the teacher's chosen plain-text format. They use the same
full-token causal-LM trainer: all supplied tokens contribute to loss. Teacher prompts can
provide source context or request closed-book answers. Deliberate duplicate rows are
preserved; the worker does not remove the teacher's chosen repetitions. Data is
student-conditioned at round boundaries; this is not per-update on-policy distillation.

## Scope and provenance

The teacher's diagnostic scores guide education; changing questions do not establish a
comparable benchmark trend. Independent benchmarks remain outside this loop. Agentic
trajectory SFT, student tool execution and probability-based OPD remain future work.
Teaching authority does not fabricate training paths the worker has not implemented.

This document is maintained here and is included in every teacher request. It adapts the
method description from the legacy studio, archived in `../nekaise-studio-bak/docs/COAPT.md`; the upstream project is not a runtime
dependency. Its independent referee, keep/revert experiment protocol, fixed frontier and
human approval rules are not this project's teaching contract. An upstream snapshot is
retained in [reference/COAPT-upstream-2026-09-14.md](reference/COAPT-upstream-2026-09-14.md).
