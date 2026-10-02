# nekaise-studio

**AI training AI, in a continuous research loop.**

An [OpenNekaise](https://github.com/OpenNekaise) workspace for developing **Kai**, our
student model. A Teacher and multiple Material Authors turn source material and the
student’s observed weaknesses into the next round of training.

## The loop

```text
Sources + student attempts → Teaching material → Training → Assessment → Next lessons
```

**CoAPT Mid-training** combines learning from domain text and instruction–response
examples. The Teacher chooses the curriculum, data mix and dose. Training carries
forward the model and optimizer; a separate orchestrator handles recovery.

Domain material comes from [nekaise-corpus](https://github.com/OpenNekaise/nekaise-corpus),
alongside a general-purpose curriculum, web sources and original generated material.
Independent evaluations from [nekaise-bench](https://github.com/OpenNekaise/nekaise-bench)
remain outside the teaching loop.

The dashboard brings together lessons, training progress, operational reports,
independent benchmark results and a conversation with the latest saved Kai model.

## Start

Clone alongside `nekaise-corpus`, then run:

```bash
uv venv --python python3 .venv
uv pip install --python .venv/bin/python -r requirements.lock
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/nekaise-loop serve
```

Open **http://127.0.0.1:8765**. Training requires a GPU environment with PyTorch and
Transformers, a locally cached student model, source data and configured Teacher/Author
providers. Set `NEKAISE_MODEL_PYTHON` to the training environment’s Python executable.

[CoAPT](docs/COAPT.md) · [Kai](docs/KAI.md) · [Architecture](ARCHITECTURE.md) ·
[Deployment](deploy/README.md) · [Agent instructions](AGENTS.md)

MIT licensed. Source material retains its original licenses.
