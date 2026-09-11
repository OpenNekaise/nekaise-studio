# nekaise-studio

_An [OpenNekaise](https://github.com/OpenNekaise) project._

A model's mistakes can tell us what to teach it next.

nekaise-studio is an agent-operated studio for fine-tuning language models under 8B
parameters. A coding agent observes the student, writes its next lessons from the corpus,
trains it, and hands the result to a referee it cannot touch. The active work is
building-energy knowledge from [nekaise-corpus](https://github.com/OpenNekaise/nekaise-corpus).

## For this student

[CoAPT](docs/COAPT.md), Co-Adaptive Pretraining and Tuning, makes the curriculum answer to
one particular student. Its current attempts and mistakes decide which passages need
attention and how the teacher presents them. After training, the changed student selects
the next round's material.

The teacher is the coding agent, or a Codex model it calls. It corrects the student's drafts
into grounded prose and turns failed closed-book answers into question-and-answer lessons.
Every sentence must be supported by the source passage; the teacher's own knowledge cannot
fill a gap in the evidence.

## Judged from outside

The teacher never grades the exam. Frozen tasks and deterministic verifiers decide keep or
revert, and CoAPT only counts as working when it beats plain continued pretraining on the
same tokens. Every run, dataset, and checkpoint keeps an immutable identity, so a result can
be rebuilt and a decision traced back to its evidence.

The contract is [SPEC.md](SPEC.md). The Studio and gym boundary is [BOUNDARY.md](BOUNDARY.md).
The live state, evidence, and next step are in [STATUS.md](STATUS.md).

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
