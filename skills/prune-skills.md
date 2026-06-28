# Skill: prune-skills

The **selection** half of how Nekaise Studio improves itself. `crystallize-skill` adds skills;
without pruning, the library accumulates duplicates, contradictions, stale advice, and lessons
overfit to a single run — and silently rots. Mutation without selection is not improvement.
Run this periodically, or whenever `skills/local/` has grown noticeably.

## What to do

Read every skill in `skills/local/` (and re-read the core `skills/` for context). Look for:

- **Duplicates** — two skills making the same point → merge into the sharper one.
- **Contradictions** — a skill contradicted by a newer, better-validated finding → the newer
  evidence wins; rewrite or delete the loser. When the evidence is genuinely unclear, **keep and
  flag**, do not guess.
- **Stale / disproven** — a lesson a later experiment overturned → delete it.
- **Overfit** — advice that only held for one dataset or building → narrow its scope, or drop it.
- **Bloat** — an essay that buried its one actionable point → cut it down to the lever.

## Rules

- Prune operates on **`skills/local/`** (this machine's emergent skills). The core `skills/*.md`,
  `packs/*`, and `lib/*` are maintainer-owned — propose changes to them as a **PR**, never edit in
  place.
- A skill survives only if its validating evidence still holds against the current eval harness.
  If you cannot point to the evidence, the skill is a hunch and should go back to `LOG.md`.
- Leave the library **leaner and non-contradictory** than you found it. Note in the relevant
  `LOG.md` what you merged or removed, and why.
