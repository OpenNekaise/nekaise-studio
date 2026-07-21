# REFACTOR-NOTES — deviations from REFACTOR.md

> REFACTOR.md says: implement by the document's principles via the minimal-change path,
> and record deviations in the PR description. This repo was refactored locally (no PR
> yet), so deviations are recorded here instead. One entry per deviation, with rationale.

## D1. Environment lock uses requirements.txt pins, not `uv lock` (R2 §4)

`pyproject.toml` has no `[project]` table (the repo is skill-driven, not installed), so
`uv lock` has nothing to lock. The pinned trio lives in `requirements.txt`
(unsloth==2026.6.9, trl==0.24.0, transformers==5.5.0) and SPEC.md declares dependency
changes human-only. **vLLM is NOT yet installed** in the current environment — installing
and pinning it is a human action (SPEC §4); until then the vLLM code paths in
`gym/runner/` and `studio/stages/rlvr.py` are wired but unexercised. Growing a
`[project]` table + `uv lock` is a welcome follow-up.

## D2. `packs/` kept as thin re-export shims (R1)

The single source of truth for every verifier/task moved to `gym/`. The old
`packs/<name>/scorer.py` files remain as ~15-line shims that re-export the gym functions
under the legacy pack contract (`load_split / is_correct / reward / extract_answer`), so
existing experiment recipes and `lib/pack.py` keep working unchanged. This satisfies the
done-when (one verifier definition imported by reward, filter, and eval) — the shims
contain no logic.

## D3. Old HF-`generate()` eval paths moved to `attic/`, GPU paths unexercised (R3/R4)

`tools/eval_probes.py` and `tools/eval_bench.py` (checkpoint mode) were rewritten on
`gym/runner` (vLLM offline engine / OpenAI-compatible server; no `transformers.generate`).
The pre-refactor versions are preserved in `attic/`. The refactor session had no GPU
budget: vLLM/TRL training entry points are validated by CPU tests (config/frozen
assertions, import isolation, verifier contracts) but have not run a real training job
yet. First GPU smoke run should be the CPT card-migration run.

## D4. Privacy wordlist location (R10)

`.privacy-denylist` (git-ignored, repo root) already existed and is kept for
compatibility; the hook now ALSO reads `NEKAISE_DENYLIST` (an absolute path outside the
repo) so the wordlist can live entirely outside the working tree as R10 asks. Building
folder names under `nekaise_data/` are still auto-denied at runtime.

## D5. Bench harness import in gym (R1)

`gym/tasks/bench.py` imports the EXTERNAL nekaise-bench repo's own grading module (the
single grading truth for that benchmark). That is not a studio import — the one-way rule
(studio → gym, never gym → studio) holds; the isolation test enforces it against
`studio/`, `lib/`, `packs/`, `tools/`, and `experiments/`.

## D6. Difficulty calibration is task-sidecar, not task-inline (R6)

`gym/tools/calibrate.py` writes `calibration.json` + physical split files under
`gym/tasks/<task>/splits/` rather than mutating the task rows in place. Rationale: task
data files stay append-only and diff-able; the sampler joins calibration at load time.
Re-running calibration replaces the sidecar, never edits tasks.

## D7. Crystallize/prune gates are CLI checks, not hard interpreter locks (R9)

`studio/tools/crystallize_gate.py` refuses (exit 1) unless the finding is replicated in
≥2 distinct experiments' structured logs; the crystallize-skill instructs the agent to run
it and CI can enforce it, but a determined process could still write a file to
`skills/local/` without calling the gate. Full enforcement would need a commit hook on
`skills/local/` — noted as follow-up; the pre-commit hook currently guards privacy only.

## D9. Findings forced by the new guardrails (not in REFACTOR.md)

Two things the refactor's own done-when checks surfaced:

1. **numeric_cloze regex bug (real referee defect).** The R1 intake smoke test ("gold
   must pass its own verifier") failed on 13 committed probes: the number regex's
   comma-group branch matched "120" inside "1200" even with no comma present, so any
   4+-digit uncomma'd gold was unwinnable for EVERY model. Fixed (comma branch now
   requires a comma). Consequence: corpus_probes scores re-baseline slightly vs
   pre-refactor numbers; pre-refactor comparisons remain valid among themselves.
2. **The literal R4 grep has exactly one hit by design**: the vLLM offline engine's own
   `self.llm.generate(...)` in gym/runner/generate.py — that IS the sanctioned rollout
   engine, not an HF fallback. tests/test_no_hf_generate.py whitelists that single call
   site and bans everything else.

## D8. Legacy experiments untouched

`experiments/agentic-cpt/` and older recipe files still use the pre-refactor
`lib/trainkit` path (kept working via the shims). New work goes through
`studio.stages.*`; the legacy recipes migrate when their experiment is next touched.
`workspace/` scripts were never tracked, so `attic/` holds only previously tracked code.
