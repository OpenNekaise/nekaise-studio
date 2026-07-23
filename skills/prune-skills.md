# Skill: prune-skills

The **selection** half of how Nekaise Studio improves itself. `crystallize-skill` adds skills;
without pruning, the library accumulates duplicates, contradictions, stale advice, and lessons
overfit to a single run — and silently rots. Mutation without selection is not improvement.
Run this periodically, or whenever the active workspace's `skills/local/` has grown noticeably.

## What to do

**Run the evidence audit first (code-enforced, R9):**

    python -m studio.tools.crystallize_gate --audit

It re-checks every local skill's `finding: <slug>` tag against the experiments' structured
SQLite decision records. Skills whose evidence no longer clears the ≥2-experiment bar — or that
carry no tag at all — are DEMOTED to hypothesis (marked `status: unverified`, moved back
to LOG-level confidence), not silently kept. The null-hypothesis rule (SPEC.md §3) applies
retroactively: a skill whose gain would not have beaten "same baseline, more compute" is
overfit advice.

Then read every skill in the active workspace's `skills/local/` (and re-read the core
`skills/` for context). Look for:

- **Duplicates** — two skills making the same point → merge into the sharper one.
- **Contradictions** — a skill contradicted by a newer, better-validated finding → the newer
  evidence wins; rewrite or delete the loser. When the evidence is genuinely unclear, **keep and
  flag**, do not guess.
- **Stale / disproven** — a lesson a later experiment overturned → delete it.
- **Overfit** — advice that only held for one dataset or building → narrow its scope, or drop it.
- **Bloat** — an essay that buried its one actionable point → cut it down to the lever.

## Rules

- Prune operates on the active workspace's **`skills/local/`** (this user's emergent skills). The core `skills/*.md`,
  `packs/*`, and `lib/*` are maintainer-owned — propose changes to them as a **PR**, never edit in
  place.
- A skill survives only if its validating evidence still holds against the current eval harness.
  If you cannot point to the evidence, the skill is a hunch and should be marked unverified.
- Leave the library **leaner and non-contradictory** than you found it. Keep the evidence
  linkage in the corresponding run decisions.
