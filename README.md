# Nekaise Studio

A local workspace for watching and operating a small-model learning loop. The teacher
selects corpus lessons, observes the student, revises its attempts, and creates a fresh
evaluation after training. Evaluation gaps guide the next dataset.

The dashboard shows the actual source passages, student generations, teacher corrections,
word-level changes, teacher lessons, token shares, training loss, online evaluation answers, learning
priorities, event history, and checkpoint lineage. It contains no simulated live data.

This is the active implementation in `OpenNekaise/nekaise-studio`. On this server it lives
at `~/Code/nekaise-studio`; the previous implementation is preserved at
`~/Code/nekaise-studio-bak` and in Git history. `~/Code/nekaise-studio-loop` is a compatibility
symlink for absolute paths in historical artifacts. New development and services use the
canonical `nekaise-studio` directory. The CLI remains `nekaise-loop`.

## Run

```bash
uv venv --python python3 .venv
uv pip install --python .venv/bin/python -r requirements.lock
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/nekaise-loop serve
```

Open **http://127.0.0.1:8765** for a local installation. Create a campaign, then start it.
The dashboard binds to loopback by default and rejects cross-origin changes. The loop worker runs independently;
closing a tab or restarting the web server does not stop training.

On this remote server, open **http://100.123.76.107:8766** from a device connected to
Tailscale. The dashboard runs as `nekaise-loop-dashboard.service` in the user service
manager and binds specifically to the Tailscale address. Port 8765 on that interface
belongs to another application. See [deploy/README.md](deploy/README.md) for operation
and setup on another server.

The control plane needs Python 3.11+. GPU work uses a separate Python selected by
`NEKAISE_MODEL_PYTHON`. On this machine it defaults to
`~/miniconda3/envs/nekaise-studio/bin/python`, with PyTorch 2.10.0 and Transformers 5.5.0.
Elsewhere, set that variable to your ML environment. The model worker uses local files
only and will not silently download weights or fall back to CPU training.

The initial recipe uses cached `openbmb/MiniCPM5-1B-Base` weights and the eligible,
cleaned corpus in `../nekaise-corpus`. An existing Claude Code or Codex CLI login powers
the teacher; no OpenAI API key is required. The default teacher is Codex `gpt-5.6-terra`; the separate recovery orchestrator is
Codex `gpt-6-astra`. Each role also supports Claude Code, with its own model setting.
The selected agent CLI must be on the server's PATH. Override executable locations with
`NEKAISE_CLAUDE` or `NEKAISE_CODEX` if needed.

## Operate through the CLI

```bash
.venv/bin/nekaise-loop doctor
.venv/bin/nekaise-loop create --name "Building energy" --config configs/example.json
.venv/bin/nekaise-loop start <campaign_id>
.venv/bin/nekaise-loop status <campaign_id>
.venv/bin/nekaise-loop pause <campaign_id>
.venv/bin/nekaise-loop resume <campaign_id>
.venv/bin/nekaise-loop stop <campaign_id>
.venv/bin/nekaise-loop continue <campaign_id> --rounds -1
.venv/bin/nekaise-loop shutdown-worker
```

`rounds=-1` (the default) continues until paused/stopped, the teacher finishes the campaign,
or a dependency becomes unavailable. Values 1–100 run a finite campaign. A lightweight supervisor persists
between rounds and wakes a separate worker or recovery agent as needed. No agent call
is needed just to schedule the next round.

Teacher quota/rate limits enter **waiting**, preserving the unfinished stage. Automatic
retries back off from 30 minutes (rate limits: 60 seconds), up to 6 hours, or respect an
explicit retry-after delay. **Resume now** retries immediately. The default local teacher
call cap is `-1` (unlimited); a positive cap waits for explicit Resume, which renews it.

Other failures enter **recovering** and wake the orchestrator after the worker exits.
It can inspect logs, repair local code, or propose recipe changes. Decisions and repair
logs are retained. Code changes require passing integration tests and a fresh continuation
campaign with the latest verified checkpoint and teaching context. Three unsuccessful
repair attempts for the same failed stage leave a visible wait for the operator.
Orchestrator quota waits without consuming this repair allowance. `auto_recover=false`
disables this automation. OpenCode is an extension, not an implemented adapter.

`pause` finishes the current stage; `stop` cancels its owned subprocess. Both cancel
automatic recovery. `resume` retries the unfinished stage; after a code change or campaign
completion it creates a linked continuation. `continue --rounds -1` explicitly starts
unlimited learning from the latest verified checkpoint. Completed artifacts remain intact.
Training retries start from the round's parent weights **and saved optimizer state**;
there is no mid-stage resume. Every retry has its own logs, metrics, and output path.

One worker owns a workspace at a time. Set `NEKAISE_LOOP_WORKSPACE` or use the CLI's
`--workspace /path` option for another workspace. One source tree permits training readers
or an exclusive repair writer. Separate workspaces do not share a GPU scheduler.
For restart/reboot recovery, install the supervisor user service in [deploy/README.md](deploy/README.md).

## Teacher authority

The teacher has full educational authority and searchable access to **all teaching history**
in this workspace, including every campaign and continuation. Each request includes the
local [CoAPT handbook](docs/COAPT.md), latest teaching notes, and a read-only archive tool.
Pagination controls response size; it does not restrict accessible rounds or campaigns.
The teacher can also inspect teaching files and search the entire corpus.

The teacher can set `train_epochs=0` to prepare lessons and assess the student without
updating weights. In an enabled training round, `train_steps=0` automatically derives
updates from the prepared token budget and chosen passes.

It chooses sources and spans, lesson count/order/type/language/difficulty, exact student
prompts, exact training text, which corrections to use, raw readings, any historical review
lessons, token shares, dataset passes, assessment design and the next teaching action.
Configured counts, source prefix, mix and passes are starting suggestions. Source-backed,
combined-source and teacher-authored exercises are supported. No source-only knowledge
rule or second semantic gate overrides the teacher. Source hashes still verify provenance.

After each round, the teacher writes durable student notes and next-round instructions,
and chooses continue, pause or complete. Teacher pause uses the command queue. Assessment
can be deferred; a diagnostic-only round keeps the parent checkpoint and records no training
metrics. The worker executes decisions; the orchestrator handles runtime faults. Operator
pause/stop and explicit execution budgets retain priority.

## Short-round training recipe

- Teacher decisions are authoritative. No second grading call, quotation-match acceptance
  gate, or fixed score threshold overrides the teacher's curriculum decisions.
- Suggested token mixture: **60% teacher lessons, 20% source text, 20% review material**;
  the teacher can change it each round, including zero shares. Shares count
  causal targets, including EOS and every token in teacher-supplied training text. No assistant-only loss mask is implied.
  Missing streams redistribute their share; the first round normally uses 75%/25%/0%.
- Each round uses its teacher-selected rows once per pass by default (`train_epochs=1`,
  `train_steps=0`); the teacher chooses the passes and any deliberate duplicate rows. Other streams are trimmed/repeated to fill their token share. A positive
  `train_steps` is an explicit override and can repeat a small dataset; actual exposure is
  recorded in metrics. The dashboard shows prepared and consumed tokens separately.
- AdamW: peak LR **1e-5**, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01`, gradient
  clipping `1.0`. Gradients accumulate **2,048 target tokens** per update, with each example's
  loss weighted by its token count. The last partial update is flushed.
- One global **8,192-token warmup**, then constant LR. Completed rounds carry Adam moments,
  global counters and FP32 weights into the next round; GPU execution uses BF16 autocast.
  Recipe/training-code changes explicitly reset the optimizer in a continuation's first
  round. `inherit_optimizer=false` records that reset, then subsequent rounds inherit again.

These are conservative starting settings for continual small datasets, not measured
optimal hyperparameters. Teacher diagnostics guide learning; they do not establish
long-term model quality.

## What is implemented

- Durable `select → plan → draft → revise → gate → freeze → train → evaluate → answer →
  grade → adapt` rounds, with immutable artifacts and chained checkpoints.
- Independent supervisor/worker, durable command queue, pause/stop/resume, process timeouts, teacher
  call caps, and per-stage attempts. Python, prompt and teacher-handbook changes invalidate resuming
  completed stages; create a new campaign when changing the recipe implementation.
- Corpus snapshots validated against authoritative manifest hashes and eligibility.
  The teacher searches all eligible sources and chooses spans explicitly. Review references
  can address any historical teacher lesson, with its immutable artifact retained.
- Trusted teacher authoring and curriculum decisions. The historical `gate` stage now
  records the teacher’s inclusion/omission decision without a second call. Exact teacher
  training sequences use the full-token causal-LM trainer.
- Live per-step loss, learning rate, gradient norm, token throughput, and peak GPU memory.
- Online teacher-authored questions, frozen reference answers before student inference,
  rubric-based teacher grading, and teacher-authored memory and strategy for the next round.

Different rounds have different evaluations. Their scores are curriculum diagnostics,
not a comparable benchmark trend. Independent benchmark evaluation remains a separate,
deferred process. Agentic trajectory SFT and OPD are documented extension directions;
this version does not claim to implement them.

## Data and reproducibility

`workspace/` holds the SQLite database, immutable JSON artifacts, subprocess logs,
datasets, and checkpoints. It is ignored by Git. Corpus text and model outputs never
belong in source control. Source URLs, licenses, hashes, prompts, teacher usage, model
revision, runtime versions, and dataset/checkpoint lineage remain inspectable locally.

The worker pins the initial model to a local snapshot. Completed checkpoint files are
hashed and verified. Checkpoints are retained for inspection; there is no automatic
deletion policy in this version. Budget roughly **12 GB per 1B checkpoint** for FP32 weights plus Adam moments,
plus interrupted saves. Unlimited rounds still require enough disk space. Stop the worker before manually archiving old campaign directories.

## Validate

```bash
.venv/bin/python -m pytest
node --test dashboard/tests/lib.test.js
node --check dashboard/dist/app.js
```

Loop tests use explicit fake adapters in temporary directories. When the configured ML
runtime exists, a tiny CPU-only model also tests actual optimizer/weight continuity and
token-weighted loss; it makes no teacher calls and is not a student quality result. The optional
`.venv/bin/python scripts/smoke_campaign.py` runs **real** two-round training: one pass over prepared tokens
per round by default, with 2 lessons and 2 online questions suggested per round and a maximum of 14 teacher
calls. It requires GPU and provider access and leaves its records in the dashboard.
This validates the machinery, not model quality.

To use the existing Codex login for this check:
`scripts/smoke_campaign.py --teacher codex --teacher-model gpt-5.6-terra`
(run with `.venv/bin/python`). Provider limits are recorded as waiting; the bounded smoke disables automatic recovery
and exits for inspection. The loop never silently switches teachers. See [docs/VALIDATION.md](docs/VALIDATION.md) for the recorded
live run and verification limits.

Read [ARCHITECTURE.md](ARCHITECTURE.md) for file boundaries and lifecycle decisions,
[docs/EXTENDING.md](docs/EXTENDING.md) for adapters, and [IDEA.md](IDEA.md) for the design.
