# Contributing to Nekaise Studio

The repo is a **bootloader**: a small, guarded kernel (skills, eval harness, plumbing) that
each user's agent extends locally. Contributions flow back through two doors.

## Door 1: promote a skill (the main path)

Your agent writes emergent skills into the active user workspace's `skills/local/` as it
works. A local skill enters the shared kernel like this:

1. **Validate.** The skill's advice must be confirmed by the eval harness — a kept run (or
   several) where following it beat the previous best. Building-specific facts never qualify;
   only transferable method does.
2. **Sanitize.** No real building / partner / address names, no personal paths. Run
   `python tools/privacy_check.py` — CI runs it too.
3. **Open a PR** that adds `skills/<name>.md` (canonical, driver-agnostic — no
   Claude-/Codex-specific instructions) plus a thin adapter `.claude/skills/<name>/SKILL.md`
   (frontmatter `name` + `description`, body pointing at the canonical file).
4. **Show the evidence** in the PR description: pack, metric before → after, number of runs,
   what the skill changes about the procedure.

`tests/test_repo_hygiene.py` enforces the mirror structure; reviewers judge the evidence.

## Door 2: platform changes

- **The referee is frozen.** PRs that change `packs/*/scorer.py`, `packs/*/prepare.py`,
  frozen exams, or how a metric is computed need a maintainer discussion *first* — the whole
  self-improving loop trusts these files.
- `lib/` is fixed plumbing: additive, backward-compatible changes only.
- Data recipes (`experiments/*/build_data.py`) are meant to evolve. Link improvements to
  exact run IDs, checkpoint digests, and typed before/after metrics.

## Dev setup

```bash
pip install -r requirements.txt
git config core.hooksPath tools/hooks   # pre-commit privacy guard
python tools/doctor.py                  # preflight
pytest                                  # must stay green (CPU-only, no GPU needed)
```

Use `python -m studio.cli workspace init ../nekaise-user-workspace` for normal
experimentation. That keeps local recipes, config overlays, run records, data, weights,
skills, and integrations out of the bootloader Git worktree. Promotion into a PR is then
an explicit copy of the minimal sanitized source change, not an accidental `git add`.

CI (`.github/workflows/ci.yml`) runs compile + tests + the privacy scan on every PR.

## Benchmark items

Independent building-energy QA questions belong in
[nekaise-bench](https://github.com/OpenNekaise/nekaise-bench), not here — that separation
(benchmark authored outside the training corpus) is what makes it a trustworthy third metric.
