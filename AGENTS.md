# Working on Nekaise Studio

Read `IDEA.md`, `ARCHITECTURE.md`, and `README.md`. The user-approved behavior is an online
teacher-evaluated training loop. Independent benchmarks stay outside the loop and must not
become a startup requirement or automatic acceptance gate. The teacher is trusted for
teaching and curriculum decisions. Keep the teacher and recovery orchestrator as separate
roles; quota exhaustion is waiting, and other interruptions wake the orchestrator by default.
The teacher has full access to all teaching history and decides sources, tasks, exact student
prompts and training text, review, token mix, assessment, notes and next teaching actions.
Do not reintroduce fixed curriculum rules, recency cutoffs or a second teaching gate.
`docs/COAPT.md` is included in teacher requests and is part of the execution fingerprint.

- Work in this repository. Treat `../nekaise-corpus` and `../nekaise-studio-bak` as read-only
  sources of data and architectural context; their live operations belong to those repos.
- Keep HTTP/UI, application service, worker, stages, persistence, and providers separate.
  Never import torch/transformers in the API process.
- Source snapshots, teacher outputs, datasets and checkpoints belong in ignored workspace
  storage. Never commit corpus text, credentials, generated training data or model weights.
- Record real observations. Do not label fixtures, invented metrics, or teacher predictions
  as student training results. Fake providers belong in tests, never the live provider list.
- Preserve stage artifacts and input fingerprints. Retry from an explicit stage boundary.
  Completed rounds carry optimizer state; do not claim mid-stage resume. Python/prompt
  changes require a fresh continuation campaign when
  the existing completed-stage fingerprints no longer match.
- Avoid editing Python or prompt files during a live campaign. The executing worker and
  its recorded source fingerprint must describe the same implementation.
- Use the command queue for start/pause/resume/stop. Only the worker owns model processes.
  Cancellation targets recorded process identities, never machine-wide name matches.
- Run the relevant Python integration tests after core changes. Pure dashboard helpers
  have Node tests and syntax checks. Do not launch real training for cosmetic changes.
- `scripts/smoke_campaign.py` incurs teacher calls and GPU work. Use it for explicit live
  validation, not as an automatic unit test. Inspect and preserve failed runs.
- Keep the UI practical and consistent with its Nordic typography, surfaces and palette.
  Main surfaces are lessons, revisions, loss, online diagnostics and activity, not marketing.
- Document extension contracts and concrete limitations. Agentic SFT/OPD remain planned
  until their real training and evaluation paths are implemented and validated.
