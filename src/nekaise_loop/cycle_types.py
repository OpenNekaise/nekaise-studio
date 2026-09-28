"""Teacher-authored block packages and joint assessment/reflection decisions."""
from pydantic import Field, model_validator

from .curriculum_types import ResearchPlan
from .teaching import Record, Curriculum, Revision, Evaluation, Grade, Reflection


class UnitResearch(Record):
    unit_id: str
    plan: ResearchPlan


class CycleResearch(Record):
    units: list[UnitResearch]


class BlockPlan(Record):
    unit_id: str
    curriculum: Curriculum
    revisions: list[Revision] = Field(description="Exact primary Teacher teaching targets, authored now from prior observations. No current-block student attempts exist.")

    @model_validator(mode="after")
    def exact_seeds(self):
        ids = [s.id for s in self.curriculum.lessons]
        if len(ids) != len(set(ids)) or sorted(ids) != sorted(r.id for r in self.revisions):
            raise ValueError("Every Teacher seed needs exactly one authored target")
        if self.curriculum.train_epochs and not self.curriculum.raw_target_tokens:
            raise ValueError("A positive buffered block needs an explicit raw corpus allocation")
        if any(r.errors for r in self.revisions):
            raise ValueError("Unattempted forward seeds have no observed student errors; cite historical findings in the teaching rationale")
        if self.curriculum.comparison_round_id or self.curriculum.scoring_round_ids:
            raise ValueError("Buffered preparation cannot run GPU diagnostics; use online assessment at the cycle boundary")
        if self.curriculum.experiment is not None:
            raise ValueError("Buffered cycles record their intent in the cycle plan; legacy per-round experiments require the legacy path")
        return self


class CyclePlan(Record):
    blocks: list[BlockPlan] = Field(min_length=1, max_length=8)
    assessments: list[Evaluation] = Field(min_length=1, description="Online student assessment frozen before training; observed at the last block in this cycle")
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def diagnostic_cycle(self):
        if any(b.curriculum.train_epochs == 0 for b in self.blocks) and len(self.blocks) != 1:
            raise ValueError("A diagnostic cycle contains one weight-preserving block")
        if len({e.id for e in self.assessments}) != len(self.assessments):
            raise ValueError("Cycle assessment IDs must be unique")
        return self


class CycleReview(Record):
    grades: list[Grade]
    reflection: Reflection
