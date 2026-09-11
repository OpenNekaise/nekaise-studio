# nekaise-studio

_An [OpenNekaise](https://github.com/OpenNekaise) project._

nekaise-studio is an agent-operated training system for small language models. A larger
language model follows repository skills to diagnose the student, generate training data
from [nekaise-corpus](https://github.com/OpenNekaise/nekaise-corpus), run training, and
evaluate checkpoints.

## CoAPT

[CoAPT](docs/COAPT.md), Co-Adaptive Pretraining and Tuning, adapts training data to the
current student. Measurements select corpus passages; the student drafts continuations and
answers questions closed-book. The teacher corrects those outputs using only the source
passage. Student attempts supply on-policy contexts for teacher-corrected targets;
claim-level gates decide what enters training.

The method spans continued pretraining, or mid-training, and SFT. The CPT branch produces
corrected prose; the SFT branch produces question-and-answer pairs serialized as text into
the same causal language-model stream. Current campaigns focus on CPT. Agentic RL remains
future work.

Read [SPEC.md](SPEC.md) for training and decision rules, [BOUNDARY.md](BOUNDARY.md) for the
studio/gym boundary, and [STATUS.md](STATUS.md) for current work and evidence.

## Start

Clone the repository, open it in Claude Code or Codex, and let the agent read
[AGENTS.md](AGENTS.md) and [STATUS.md](STATUS.md).

```bash
git clone --depth 1 https://github.com/OpenNekaise/nekaise-studio.git
cd nekaise-studio
python tools/doctor.py
```

The doctor checks the GPU, pinned environments, configuration, corpus, and holdout. Fix
reported problems before training.

Keep generated work in an external workspace:

```bash
python -m studio.cli workspace init ../nekaise-user-workspace
python -m studio.cli --workspace ../nekaise-user-workspace integrity
```

Select it with `--workspace` or `NEKAISE_WORKSPACE`. The repository is the bootloader;
rounds, runs, checkpoints, and local skills live in the workspace. See
[docs/WORKSPACE.md](docs/WORKSPACE.md).

## License

`nekaise-studio` is MIT licensed. It is one project in the wider
[OpenNekaise ecosystem](https://github.com/orgs/OpenNekaise/repositories).
