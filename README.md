# nekaise-studio

_An [OpenNekaise](https://github.com/OpenNekaise) project._

A model's mistakes can tell us what to teach it next.

Most training data is written for no one in particular. This studio writes it for one student, at one moment in its development, and then checks whether the lesson took.

Nekaise Studio is an agent-operated platform for fine-tuning language models under 8B parameters. A coding agent observes the student, builds its next lessons, runs training, and submits the result to an independent referee.

The active work is building-energy knowledge from [nekaise-corpus](https://github.com/OpenNekaise/nekaise-corpus). The aim is specialized models that run on-premises through [nekaise-edge](https://github.com/OpenNekaise/nekaise-edge), with a record of what they were taught and whether it helped.

## For this student

[CoAPT](docs/COAPT.md), Co-Adaptive Pretraining and Tuning, makes the curriculum answer to one particular student. Its current attempts and mistakes determine which source passages need attention and how the teacher presents them. After training, the changed student selects and shapes the next round's material.

The teacher is the coding agent, or a Codex model it calls. It corrects student drafts into grounded prose and can turn failed closed-book answers into question-and-answer lessons. Every claim must be supported by the source passage. The teacher's own knowledge cannot fill a gap in the evidence.

Material passes a grounding gate before entering training. The recipe stays fixed across rounds; nothing is tuned between them. The lessons change because the student changes.

## An exam outside the lesson

The teacher never grades the exam. An independent referee, with frozen tasks and deterministic verifiers the teacher cannot alter, supplies the measurements for keep or revert.

A gain must clear a measured noise band without breaching the transfer guardrail. Claims that CoAPT works must also beat an equal-compute raw-corpus control. More training alone is not evidence for better teaching. [SPEC.md](SPEC.md) binds these decisions; changing the contract requires human review.

A 1B-student pilot found a small gain over equal-token raw continued pretraining on a corpus-derived pool, while passing the transfer guardrail. [STATUS.md](STATUS.md) records the live evidence, its limits, and the next work.

Studio contains the gym; the gym never contains Studio. The gym holds tasks, verifiers, and the runner. It contains no training code. [BOUNDARY.md](BOUNDARY.md) keeps that separation explicit.

## A record that lasts

Runs, datasets, and checkpoints have permanent identities and immutable provenance. Source passages, student attempts, teacher revisions, gate verdicts, and token counts remain linked. A dataset can be rebuilt byte for byte. A decision can be traced back to its evidence.

The repository is a maintained bootloader. Its [skills](skills/) are the agent's operating procedures; mutable work lives in a user workspace. Validated findings can become local skills. Stale or contradicted advice is pruned. Promotion into the shared bootloader requires evaluation and human review.

## Start

Clone the repository, open it in Claude Code or Codex, and let the agent read [AGENTS.md](AGENTS.md) and [STATUS.md](STATUS.md).

```bash
git clone --depth 1 https://github.com/OpenNekaise/nekaise-studio.git
cd nekaise-studio
python tools/doctor.py
```

The doctor checks the GPU, pinned environments, configuration, corpus, and holdout. Fix reported problems before training.

Keep generated work in an external workspace:

```bash
python -m studio.cli workspace init ../nekaise-user-workspace
python -m studio.cli --workspace ../nekaise-user-workspace integrity
```

Select it with `--workspace` or `NEKAISE_WORKSPACE`. See [docs/WORKSPACE.md](docs/WORKSPACE.md) for workspace isolation and promotion.

## License

`nekaise-studio` is MIT licensed. It is one project in the wider
[OpenNekaise ecosystem](https://github.com/orgs/OpenNekaise/repositories).