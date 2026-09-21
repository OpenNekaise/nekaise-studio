# nekaise-studio

**AI training AI, in a continuous research loop.**

Nekaise Studio is an [OpenNekaise](https://github.com/OpenNekaise) project for autonomous
model development. AI agents design curricula, create training data, train a student model,
and evaluate its behavior to decide what to try next. Each iteration carries forward the
student's checkpoint and the accumulated teaching history: the results of one experiment
shape the next.

## CoAPT Mid-training

[CoAPT](docs/COAPT.md), Co-Adaptive Pretraining and Tuning, adapts training data to the
current student. In this project, **CoAPT Mid-training** is the unified name for
**CPT + SFT**: learning from domain text and teacher-corrected question-and-answer
examples in the same training loop. CPT/SFT distinctions belong in recipes and material
composition, rather than separate training modes.

**CoAPT Post-training** is our collective name for **RL + OPD**, reserved for future
work. Current development and unqualified references to training mean CoAPT Mid-training.
The names do not require fixed proportions or a fixed sequence of training phases.

```text
Student attempts → Teacher corrections → Mid-training → Teacher assessment → Next lessons
```

The teacher has full educational authority and access to the complete teaching history.
It chooses the lessons, writes corrections, decides what enters training and what needs
review, and assesses the student's understanding. It can draw on the corpus and its own
knowledge to explain a concept or construct an exercise.

The teacher chooses each round's size, mixing corrected prose and question-and-answer text
with source material and earlier lessons. Each round continues from the retained student;
its actual answers guide the next curriculum. Independent benchmarks stay outside the loop.

The loop can run continuously, preserving checkpoints and teaching history across pauses
and retries. A separate orchestrator handles execution failures and recovery. It receives
host validation results, decides whether to retry, continue, wait or pause, and records
what it investigated or changed and why it chose that action. Explicit user holds are
separate from agent pauses. When idle execution has no scheduled review, the supervisor
wakes the orchestrator; teacher pauses also take this path. The dashboard separates
three workspaces: **Report** for current execution and orchestrator decisions, **Studio**
for teaching and student work, and **Previous runs** for searchable history and continuation
lineage. **Studio is the homepage**, with teacher-token, trained-token, teacher-assessment
and independent-evaluation charts visible together, followed by the current iteration and loss. Lessons & assessments
and Activity have their own sections. **Experiments** browses pre-training hypotheses,
strategy versions, later Teacher conclusions and measured evidence across all runs;
see [teaching experiments](docs/TEACHING_EXPERIMENTS.md). Session time and token consumption follow the
selected continuation lineage; loss and activity describe the selected run. Historical
views have an explicit return to the current run.
The orchestrator also organizes previous runs and training logs. It gives retention reasons,
keeps archived runs accessible, and moves selected redundant raw logs to recoverable trash
with summaries. The **Report** tab shows current status, agent decisions, host validation and recorded execution outcomes. The orchestrator reads the same report archive before deciding. Routine recovery needs no user approval: operational pauses schedule another automatic review, and repeated repair failures cool down before retrying. Explicit user pause/stop remains available. Teaching records and checkpoint lineage remain; the orchestrator decides which historical weights and optimizer states to retain, preserving findings and explicit deletion receipts.

## Start

Clone the repository alongside `nekaise-corpus`, then open it in Codex or Claude Code and
let the agent read [AGENTS.md](AGENTS.md). To start the dashboard yourself:

```bash
uv venv --python python3 .venv
uv pip install --python .venv/bin/python -r requirements.lock
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/nekaise-loop serve
```

Open **http://127.0.0.1:8765**, create a campaign, and start it. Training requires a GPU
environment with PyTorch and Transformers, a locally cached student model, eligible corpus
text, and an authenticated Codex or Claude Code CLI. Set `NEKAISE_MODEL_PYTHON` to the
Python executable in your training environment.

See [deployment](deploy/README.md) for service setup, [architecture](ARCHITECTURE.md) for
implementation details, and [validation](docs/VALIDATION.md) for what has been tested.

## License

[MIT](LICENSE). Corpus documents retain their original licenses.

### Student starting points

Historical campaigns use MiniCPM5-1B-Base with raw-text continuation. A fresh campaign can
select a pinned MiniCPM5-1B-SFT snapshot with `student_format=chat_template` and a fresh
optimizer. Native single-turn no-thinking prompts and `chat_response` teaching preserve
exact generation-prefix tokens and assistant termination; full-sequence CoAPT Mid-training
loss is unchanged. Base checkpoints remain available for teacher-led comparisons with
each side's interface recorded. See [student serialization](docs/COAPT.md#student-text-interface-and-response-boundaries).

## Parallel material authors

The primary teacher can delegate plan/seed expansion to configurable **Material Authors**.
Remote APIs and compatible locally served models share a bounded async scheduler with
per-author and shared-resource limits. The teacher selects exact candidates before
freezing; provenance, retry usage and unobserved synthetic examples remain explicit.
Configure the registry in `workspace/material-authors.json` and credentials in the ignored
root `.env`. See [Material Authors](docs/MATERIAL_AUTHORS.md) for setup, budgets, local
server limits, archival access and extension contracts.
Claude Code can also provide expanded material through `transport=claude_code` using
existing CLI authentication and a pinned Opus model. Its owned processes and author
usage remain separate from primary Teacher calls; Teacher selection still controls
the training package.


### Required expansion and efficiency (2026-09-18)

Positive training requires Material Author expansion and teacher-selected expanded targets.
Diagnostic rounds may preserve weights without expansion. Teacher content/review authority
remains intact. The default `required_v1` policy supersedes earlier optional-delegation advice;
legacy artifacts stay readable. Studio displays measured training tokens / primary Teacher
tokens, with per-round observations and a smoothed trend. The same usage feedback reaches
curriculum and reflection. See [the contract](docs/COAPT.md#training-production-efficiency).
