# nekaise-studio

**AI training AI, in a continuous research loop.**

Nekaise Studio is an [OpenNekaise](https://github.com/OpenNekaise) project for autonomous
model development. AI agents design curricula, create training data, train a student model,
and evaluate its behavior to decide what to try next. Each iteration carries forward the
student's checkpoint and the accumulated teaching history: the results of one experiment
shape the next.

The first application is building-energy intelligence, grounded in
[nekaise-corpus](https://github.com/OpenNekaise/nekaise-corpus).

## CoAPT

[CoAPT](docs/COAPT.md), Co-Adaptive Pretraining and Tuning, adapts training data to the
current student.

```text
Student attempts → Teacher corrections → Train → Teacher assessment → Next lessons
```

The teacher has full educational authority and access to the complete teaching history.
It chooses the lessons, writes corrections, decides what enters training and what needs
review, and assesses the student's understanding. It can draw on the corpus and its own
knowledge to explain a concept or construct an exercise.

Training proceeds in small rounds, mixing corrected prose and question-and-answer text
with source material and earlier lessons. Each round continues from the retained student;
its actual answers guide the next curriculum. Independent benchmarks stay outside the loop.

The loop can run continuously, preserving checkpoints and teaching history across pauses
and retries. A separate orchestrator handles execution failures and recovery. The dashboard
shows the work as it happens: student attempts, teacher revisions, training loss and assessments.

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
