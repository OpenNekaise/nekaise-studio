# CoAPT Mid-training in Nekaise Studio

Co-Adaptive Pretraining and Tuning creates teaching material in response to the current
student's actual behavior. The teacher has full educational authority; the worker executes
and records its decisions. The separate orchestrator repairs runtime failures.

## Project terminology and current scope

**CoAPT Mid-training = CPT + SFT** in this project: continued pretraining and supervised
fine-tuning form one path, combining domain-text learning and teacher-corrected QA
supervision. Unless explicitly qualified, project discussions, documentation and teacher
instructions use training or CoAPT to mean CoAPT Mid-training. This is the current
implementation and development scope.

**CoAPT Post-training = RL + OPD** is the name reserved for future reinforcement learning
and on-policy distillation. It is planned, not an available training mode. Concrete
implementation choices are deferred while we focus on Mid-training. These are project
naming conventions, not universal definitions of the underlying methods.

CPT and SFT remain useful only for recipe composition and material provenance: `cpt`
labels teaching prose, and `sft` labels QA supervision. Both use the same trainer. Recipes
may describe how much of each to use; the teacher chooses their composition through the
lessons. The existing `token_mix` controls teacher/corpus/replay shares. Neither naming
convention introduces a new ratio setting, a mandatory nonzero share, or a fixed sequence.
The teacher may choose either material type, both, or reading/review/diagnostic-only rounds.

## Teacher authority and memory

The teacher can inspect all teaching history in this workspace, across every campaign and
continuation: source passages, prompts, actual attempts, corrections, prior questions and
answers, grades, training metrics, dataset/checkpoint metadata, notes and operational logs.
The archive tool provides pagination and search with no recency cutoff. Latest teaching
notes are injected as a starting point; older notes and full artifacts remain accessible.

The teacher chooses sources, topics, lesson count, order, language, difficulty, teaching
style, the recipe's CPT/SFT composition, the exact prompt shown to the student, raw
readings, historical review examples, token shares, passes over the dataset, evaluation
design and next actions.
Configured lesson counts, question counts, source prefix and mixture are suggestions.
Explicit operator lifecycle/resource budgets remain execution constraints.

Set curriculum train_epochs=0 to preserve weights for this round while still preparing
lessons and conducting assessment. Positive train_epochs enables training. The separate
config train_steps=0 means automatic update counting from tokens and passes; it does not
mean zero training. A positive train_steps overrides the update count of enabled training.
The worker executes the structured fields; notes explain those decisions.

Teacher-selected token shares normally sum to 1. With `train_epochs=0`, the teacher may
instead set all three shares to 0: lesson text and student observations remain recorded,
but no training target tokens are prepared or consumed. A diagnostic round may also keep
a sum of 1 to prepare samples without weight updates. Positive training passes require
shares summing to 1. The campaign's initial suggested shares still sum to 1.

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
   for learning must be in training_text or the selected chat_response training_response. The teacher supplies
   the content and selects the documented serialization; unrelated fields are not targets.
4. The worker freezes exact tokenized samples, lineage, source snapshots and realized token
   shares. It trains the current student with the recorded optimizer recipe. Diagnostic-only
   rounds preserve the parent checkpoint and produce no invented training metrics.
5. The teacher chooses an online assessment, with references/rubrics recorded before the
   student answers. It grades actual answers and decides which concepts need practice.
6. The teacher writes durable student notes and the next teaching strategy, then chooses
   continue, pause or complete. A teacher pause suspends teaching and wakes the separate
   orchestrator for operational review, preserving the teacher's reasons and notes.
   It does not require routine user approval; only an explicit user pause/stop holds
   automatic execution indefinitely. The next round starts from that retained checkpoint.

Within the Mid-training recipe, CPT material is usually self-contained prose. SFT material
includes a task, context as needed and its answer, in the teacher's chosen raw-text or native single-turn chat
format. The unified full-token causal-LM trainer learns from both prompts and answers;
this is full-sequence QA supervision. Teacher prompts can provide source context or
request closed-book answers. Deliberate duplicate rows are
preserved; the worker does not remove the teacher's chosen repetitions. Data is
student-conditioned at round boundaries; this is not per-update on-policy distillation.

## Student text interface and response boundaries

Campaign `student_format` declares the inference interface. Historical Base campaigns
use `raw_text`: the worker tokenizes exact `student_prompt` with the checkpoint tokenizer's
special-token handling and continues it. Fresh SFT campaigns may use `chat_template`:
`student_prompt` is the exact content of one user message, wrapped by the checkpoint's
native template with `enable_thinking=false` and its assistant generation prefix. There
is no added system message, reference, rubric or hidden teaching text. Lesson and evaluation
`student_format=campaign` use this default; explicit `raw_text` or `chat_template` overrides
are available for deliberate interface probes. Only newly generated tokens are returned.
Chat prompts exceeding `max_seq_len` fail explicitly because truncation could remove the
assistant boundary. Raw prompts retain their recorded truncation behavior.

The Base and SFT checkpoints are distinct starting points. A fresh SFT campaign resets
optimizer state and does not become a descendant of Base weights. Checkpoint comparison
uses the reference round's saved interface, defaulting to raw text for older artifacts;
an explicit per-lesson interface override applies to both sides. It compares checkpoints
AND interfaces, not weights alone. Each side records actual prompt IDs and serialization.
Do not attribute a difference solely to SFT training or treat it as proof of the earlier
Base incident's cause. Initial context may identify a user-requested diagnostic comparison;
read `campaign_context_artifact` through the archive alongside the complete teaching history.

You own the relationship between student_prompt and training_text. Full-sequence
training learns task wording, directives, delimiters, answers and the terminating EOS;
it does not mask the prompt or automatically extract an assistant response. If you
want to teach continuation from a particular response boundary, you can include that
same prefix in training_text and end student_prompt at the intended start of generation.
For example, text serialized as `Task: ...\n\nAnswer: ...` and a bare question present
different contexts. Whitespace and punctuation at the boundary can also change token
IDs; a character prefix is not necessarily an exact prefix of the frozen token sequence.
Inspect saved prompt_token_ids, frozen samples and raw generation evidence when that
distinction matters. Choose any format, or deliberately contrast formats; no template
or prefix match is mandatory for ordinary full-text tokenization, and the worker does
not rewrite your choices.

Revision `training_tokenization` defaults to `full_text`, preserving ordinary whole-text
tokenization and historical replay. You may instead choose `prompt_prefix` when
`training_text` starts with the exact nonempty `student_prompt`. This optional mode
tokenizes the prompt with generation's special-token handling, then the unchanged
remaining suffix without additional special tokens, and appends EOS as usual. It
prevents BPE merges across that chosen boundary without inserting a template, rewriting
whitespace, masking prompt loss or suppressing generated EOS. The explicit prefix
requirement is a serialization constraint only for this option, not a teaching gate.
Frozen rows record the mode and prompt; replay preserves the original choice. Chunking
and mixture allocation still apply, so inspect actual samples and prompt truncation.
Token alignment makes the intended continuation trainable; it does not guarantee recall,
grounding or recovery of earlier learned behavior. Exact prompts, text, mode, exposure
and assessment remain your choices.

For native chat supervision, choose `training_tokenization=chat_response`, supply only
the exact assistant continuation in `training_response`, and set unused `training_text`
to empty. The worker encodes the same native no-thinking generation prefix, then your
response separately, then the template's assistant terminator through its first effective
EOS. This preserves the generation boundary even when a completed-message template omits
the empty thinking prefill or BPE would merge across the response boundary. It adds no
second BOS or legacy EOS after the chat terminator; whitespace after termination is not
a target. Empty responses are an explicit teaching choice, not inferred from an empty
attempt. Native templates must support a literal single-turn assistant continuation;
unsupported serialization fails rather than silently rewriting text. Raw `full_text` and
`prompt_prefix` remain available for prose or deliberate contrasts, including in chat
campaigns. All three modes supervise the full sequence, including prompts and role tokens;
this is not assistant-only loss or multi-turn/tool trajectory training.

Frozen chat rows preserve user content, rendered prefix, prefix IDs, template hash, response,
and closing IDs. Replay preserves the chosen mode and content. A saved chat template hash
must still match, or the teacher must explicitly re-author that lesson. Replay is tokenized
with the current checkpoint tokenizer, so raw replay can differ across checkpoints.
Actual encoded IDs determine BOS handling: this pinned SFT fast tokenizer inserts BOS 0
for raw `add_special_tokens=True` through its post-processor despite its
`add_bos_token=false` attribute. Native chat uses `add_special_tokens=False` because its
template owns BOS. Frozen IDs, not tokenizer flags or a claim of cross-lineage token
identity, define actual training. Corpus prose remains raw text.

Supplying an answer fragment in `student_prompt` does not make that fragment
conditioning-only during training. All tokenization modes supervise its tokens too.
For example, if a lowercase-answer task's prompt is extended with `Co` and its training
text completes `Cobalt`, full-sequence loss also teaches `Co` immediately after the
original, unextended prompt. That can conflict with teaching lowercase `cobalt` there;
`prompt_prefix` does not isolate supervision to the remaining `balt`. An auxiliary
wrong-prefix diagnostic need not become a training target or a required normal input.
You choose whether and how to teach such branches consistently with the intended task.

Reflection receives `prompt_training_prefixes`, also saved in the freeze artifact:
character-prefix agreement and per-sample comparisons of the actual recorded prompt
IDs with frozen training IDs, including common-prefix lengths and boundary tokens.
This is descriptive evidence for your next choices, not a request to align all tasks.
Missing prompt IDs or absent samples leave agreement unknown. Each sample is compared
separately because chunking and mixture allocation may shorten, omit or repeat material;
a sample comparison does not reconstruct the complete row. Prompt truncation and
diagnostic rounds without training must also be considered. Whitespace in your exact
prompt is preserved, but adding whitespace does not guarantee token-prefix agreement.

Selecting replay references does not guarantee consumption of every selected example
or its answer. The replay token budget can omit rows or truncate a sample inside its
prompt. Training passes repeat the frozen samples; they do not rotate omitted rows
into that budget. Inspect row IDs and actual target spans in the frozen samples and
recorded training consumption before attributing preservation to replay. You choose
the references, mixture, material and exposure; no minimum share or coverage gate is
imposed. Actual before/after responses establish what was preserved.

For empty, copied or unrelated responses, separate observed termination/continuation
from a demonstrated subject-matter misconception. Matching FP32 forward audits establish
execution consistency for those tokens, not learning or general instruction following.
You may compare a faithful trained prefix with a fresh task using a comparable response
boundary, or choose another teaching approach. Recall of a stored row is distinct from
transfer to a new task. Exact prompts, whether to train, exposure, assessment and next
actions remain your decisions; neither prefix agreement nor a correct answer is an
operational acceptance gate. Report suspected runtime discrepancies to the orchestrator.

## Scope and provenance

### Learning work and same-question observations

The curriculum receives `previous_learning_work` for the latest completed round in
its own continuation lineage; all older history remains available. Reflection receives
the current `learning_work`, excluding its still-running stage. These observations
separate recorded target exposure and optimizer updates from per-stage walltime and
checkpoint storage. Work across retries is identified separately from the retained
training artifact. The inner training timer excludes model loading, generation and
saving. Stage sums exclude unfinished attempts and between-round operational reviews;
neither timer measures GPU utilization. Prepared prompt/continuation coverage is not
consumed exposure, and continuation targets include termination tokens.

Use this evidence to choose useful work per round, material, passes and assessment.
A small dataset and one pass may produce only one short update despite expensive
authoring and checkpoint I/O. More passes are an available experiment, not a minimum
or an assurance of learning. When errors persist, decide what bounded observation
could separate recall, transfer, dose, retention and presentation. You choose when
prerequisite exercises should reconnect to the campaign's subject. No fixed curriculum,
token quantity, repetition schedule, mixture or learning-score gate follows from these
measurements.

Online evaluation items may set `compare_before=true`. The worker freezes the round's
verified input/output checkpoint identities with the question and reference, then runs
that same exact prompt on both checkpoints, with the same configured interface and
greedy decoding. Only selected questions receive additional inference. Both processes
share one stage execution budget and retain their own logs and generation evidence.
The unfinished round already protects its input and output from checkpoint retirement.
Checkpoint checks for inference verify model/tokenizer files and manifest identity;
they do not read optimizer contents. Unrecorded missing files still fail verification,
and training/resumption checks continue to verify optimizer bytes too.

Declare optional `dimensions` as named `{name, rubric}` criteria with the question,
before collecting answers. Grading returns `{name, score, feedback}` for each declared
dimension. Names and rubrics are yours: distinguish factual/numeric meaning, requested
procedures, output form or other aspects when useful. The overall grade is your judgment,
not an enforced average. Equivalent but unreduced fractions and incorrect numeric values
are different observations; strict formatting alone does not prove missing knowledge.

Paired answers are graded together with shuffled opaque IDs and checkpoint labels
removed. This reduces explicit ordering cues; it does not isolate grading from all
teacher memory. Original IDs, checkpoint provenance, both answers and both grades are
restored in the saved evidence. A per-item score difference is recorded only when
actual prompt tokens, generation settings and runtime match. This is online diagnostic
feedback, not a general capability score or acceptance decision. Questions can still
change across rounds. If no update occurred, one actual generation and grade are reused
with an explicit no-update label; this is not a second observation or measured gain.
Existing historical scoring remains available for exact before/after target likelihoods.

The teacher's diagnostic scores guide education; changing questions do not establish a
comparable benchmark trend. Independent benchmarks remain outside this loop. CoAPT
Post-training (RL + OPD), student tool execution and role-aware trajectory supervision
remain future work; the implemented Mid-training SFT component is text QA supervision.
Teaching authority does not fabricate training paths the worker has not implemented.

This document is maintained here and is included in every teacher request. It adapts the
method description from the legacy studio, archived in `../nekaise-studio-bak/docs/COAPT.md`; the upstream project is not a runtime
dependency. Its independent referee, keep/revert experiment protocol, fixed frontier and
human approval rules are not this project's teaching contract. An upstream snapshot is
retained in [reference/COAPT-upstream-2026-09-14.md](reference/COAPT-upstream-2026-09-14.md).


Historical checkpoint availability follows orchestrator retention decisions: full state, inference
weights only, or records only. Archive round queries include `checkpoint_retention`; a missing
receipt means no deliberate retirement. Preserve learning from all recorded lessons and metrics,
but choose historical inference targets whose weights remain available. An unavailable model is
not a missing learning record, and no checkpoint is silently recreated or reset after retirement.
## Batched execution and useful work per round

Student attempts and online answers execute in bounded same-checkpoint FP32 greedy
batches. The execution row and token-position limits do not choose lesson count or
training dose. The teacher retains full authority over tasks, content, coverage,
mixture, repetition, assessment and focused diagnostic rounds. Operator
`workload_guidance` expresses a preference about useful work and feedback granularity.

A curriculum may record `work_plan`: estimated causal targets per pass, material
strategy and dose rationale. Estimates are distinct from frozen preparation and
actual optimizer metrics. The teacher sees available/prepared/unused/repeated
target occurrences per stream, actual update sizes and completed inference batch
work at reflection and next selection. These values do not impose an acceptance
gate or minimum volume. More repetitions do not establish new material coverage.

The current recipe anchors the first available positive-share stream in
teacher/corpus/replay order. Non-anchor streams fill their declared ratio; selected
material can be partly unused or repeated. Each pass flushes its final partial
update. A larger reading selection alone need not increase training when a small
teacher anchor and the chosen shares still bound prepared targets.

Paired online questions use matching prompt groups on both checkpoints. Shared
batch context, exact unpadded prompts, decoding and runtime are recorded for
comparability; padding tokens after a response terminates are never student output.
The teacher still receives actual observations and decides what they establish.


## Delegated material authors

The operator requires expansion through registered **Material Authors** in every positive-training round (authorized 2026-09-18; `expansion_policy=required_v1`). Authors follow the teacher's plan, corrected seeds and selected source snapshots. They may be remote APIs or locally served models. Author concurrency and request/output limits are execution budgets, not curriculum quotas. The teacher chooses authors/jobs and their content. Positive training must consume teacher-selected, edited or preauthorized expanded material; `train_epochs=0` diagnostics may omit expansion. Author failures retain ordinary waiting/recovery; the host never silently changes a recipe to a diagnostic. This operator requirement supersedes older optional-delegation guidance.

After seed revision, the worker persists independent author jobs and candidate artifacts. The operator-selected `material_review_policy` controls how the teacher authorizes them:

- `trusted_author_v1` implements the 2026-09-22 decision to completely trust generated teaching content. Curriculum jobs approve complete structurally valid batches in advance at the planned mix/passes. The material_select stage records exact accepted job IDs, manifest identity and explicit preauthorization without a Teacher call, content review, pruning or edits. Primary seeds retain their revision decisions. Online evaluation and reflection assess the student using coverage and observed answers, without re-auditing generated teaching text. Exact material remains available for teaching use. The teacher chooses subsequent tasks, authors and dose; zero-pass diagnostics remain possible.
- `teacher_review_v1` preserves the historical workflow and is the compatibility default when the field is absent. A Teacher call chooses review scope, exact IDs/batches, edits, omissions and final shares/passes. Rejecting every candidate requires explicitly choosing zero passes.

Trust does not weaken completion/schema/provenance/budget checks or label synthetic material as observed student work. A positive-training package with no expanded targets still fails for orchestrator recovery. Requested jobs execute even when the planned passes are zero. Exact duplicate candidates are not silently removed; prepared occurrences, exact content variants and actual exposure stay distinct. Changing the operator's trust policy requires a new operator instruction, not an automatic review or score gate.

Auxiliary synthetic material records provider/model, source/seed/job provenance and teacher selection. A variant without a student attempt is explicitly unobserved, never an empty student answer or an invented correction. Accepted variants join the existing teacher material stream with CPT/SFT labels and origin metadata. The student tokenizer applies its own native serialization. API reasoning fields are not copied into training. All selected material is visible to online assessment, reflection and later replay.

Report distinguishes requested and completed jobs, reservations/retry usage, prepared targets by origin and actual optimizer work. These quantities do not establish learning. Failed jobs are preserved for orchestrator recovery; they are not silently omitted. Full teaching and operational history stays available even when large request bodies are supplied through immutable files rather than inline context.


## Training production efficiency

The operator wants more useful measured training exposure and **Tokens trained / Teacher tokens**.
`learning_work.teacher_efficiency` divides actual optimizer target tokens across all passes and
retry attempts by primary Teacher input + output, counting cache/reasoning once. Author API
and orchestrator work remain separate. Different tokenizers are involved; this measures token
production efficiency, not monetary cost or learning gain. The next curriculum gets complete
round usage; reflection gets a provisional snapshot excluding its own call. Missing/pending
usage produces no complete ratio; `reported_ratio` is explicitly partial. A zero or unknown
denominator has no ratio.

Use informative seeds, substantial useful delegated variation, and the configured teacher authorization workflow.
Read efficiency alongside prepared targets per pass, passes, source allocation and observed
learning. Repetition raises exposure without distinct coverage; do not pad text, repeat solely
for the metric or remove necessary assessment. No efficiency threshold or independent
benchmark score controls teaching or checkpoint decisions.

Studio plots completed-round ratios with a descriptive smoothed curve. Its headline is summed
lineage trained tokens / summed reported Teacher tokens, never a mean of round ratios. Missing
or active calls make it provisional. Incomplete rounds are omitted from the scatter with their
count shown. Existing token totals retain their meaning. Historical artifacts stay immutable;
`legacy_optional` is a compatibility policy, not authorization to downgrade the live policy.

## Teaching experiments

Use Curriculum.experiment to record a teaching hypothesis, intended intervention,
budget basis, observation plan and evidence that would make you reconsider it.
Define or reuse an immutable strategy version; earlier experiments are available
through the read-only experiments, experiment and strategy archive operations.
A null plan is valid when you are not proposing an explicit experiment.
Reflection.experiment_review records your separate later judgment, including
uncertain, null and negative findings; it never rewrites the pre-training plan.
You choose the teaching method, dose, assessment and next action. These records
impose no fixed curriculum, test panel or score gate. The observation plan does
not freeze questions before training: online questions are still frozen before
answers. Sequential rounds and different starting states do not establish a
matched-start A/B comparison. Independent benchmark evidence stays outside this
teaching record. See docs/TEACHING_EXPERIMENTS.md for the executable contract.
