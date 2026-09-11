# nekaise-studio

_An [OpenNekaise](https://github.com/OpenNekaise) project._

nekaise-studio is an agent-operated studio for fine-tuning small language models. A larger
language model reads what the small model gets wrong, writes the next round of training
data from the corpus, trains it, and hands the result to a referee it cannot touch. The
active work is building-energy knowledge from
[nekaise-corpus](https://github.com/OpenNekaise/nekaise-corpus).

## CoAPT

[CoAPT](docs/COAPT.md), Co-Adaptive Pretraining and Tuning, is the training loop. It spans
mid-training and post-training: continued pretraining on domain text, SFT on
question-and-answer pairs, and, in a future campaign, agentic RL, under one rule. The
training data is on-policy, generated from the current student's own outputs, with targets
corrected by a larger model that may use only the source passage. The teacher's own
knowledge cannot fill a gap in the evidence.

Each round, the student drafts over corpus passages and answers closed-book questions.
The teacher turns those drafts and answers into grounded text and Q&A pairs, the gated mix
is trained through a fixed recipe, and the changed student selects the next round's data.
The recipe never moves between rounds; only the data does.

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
