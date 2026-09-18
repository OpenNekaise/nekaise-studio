"""Teacher-owned teaching decisions; validation describes executable data shapes."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import TokenMix
from .material_types import ExpansionJob, MaterialEdit
from .experiment_types import ExperimentPlan, ExperimentReview


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceRef(Record):
    document_id: str = Field(min_length=1)
    start: int = Field(ge=0)
    length: int = Field(ge=0, description="0 selects the rest of the document")


class ReplayRef(Record):
    round_id: str
    lesson_id: str


class Lesson(Record):
    id: str = Field(min_length=1)
    kind: Literal["cpt", "sft"] = Field(description="Recipe material within CoAPT Mid-training: cpt=teaching prose, sft=QA supervision")
    sources: list[SourceRef]
    concept: str
    prompt: str
    student_prompt: str = Field(min_length=1, description="Exact student content, including desired context; native chat wraps it as a user message when selected")
    student_format: Literal["campaign", "raw_text", "chat_template"] = Field(default="campaign", description="Use the configured interface, or explicitly choose literal raw text/native no-thinking chat for this prompt")
    reason: str


class CurriculumTokenMix(TokenMix):
    """Shares sum to 1; all zero is also valid when train_epochs=0"""

    @model_validator(mode="after")
    def total(self):
        if self.teacher == self.corpus == self.replay == 0:
            return self
        return super().total()


class WorkPlan(Record):
    estimated_targets_per_pass: int = Field(ge=0, description="Teacher estimate of prepared causal targets before repetition; an estimate, never a measured result or minimum")
    material_strategy: str = Field(min_length=1, description="Intended coverage and sources of distinct material, readings and review")
    dose_rationale: str = Field(min_length=1, description="Why this amount and these passes fit observed learning and fixed execution cost; explain a focused diagnostic when chosen")


class Curriculum(Record):
    lessons: list[Lesson]
    readings: list[SourceRef]
    replay: list[ReplayRef]
    token_mix: CurriculumTokenMix
    train_epochs: int = Field(ge=0, description="0 keeps weights unchanged; positive values train this many dataset passes")
    evaluation_instructions: str
    notes: str
    experiment: ExperimentPlan | None = Field(default=None, description="Record a teaching hypothesis and strategy before execution; null when no explicit experiment is proposed. Does not require a fixed diagnostic or score gate.")
    expansion_jobs: list[ExpansionJob] = Field(default_factory=list, description="Delegated material-author jobs following your plan and corrected seeds. Required for positive training under required_v1; diagnostics may leave empty. Execution budgets are provided separately.")
    work_plan: WorkPlan | None = Field(default=None, description="Explicit teacher-owned work estimate and rationale; null is retained for older plans or when an estimate is unavailable")
    comparison_round_id: str = Field(default="", description="Optional completed historical round whose checkpoint should also answer the exact lesson prompts; empty disables comparison. Evidence only, never a checkpoint replacement.")
    comparison_checkpoint: Literal["output", "input"] = Field(default="output", description="Select the historical round's output checkpoint, or its recorded input: a verified completed workspace checkpoint or pinned local Hub snapshot. Input requires immutable parent provenance and file integrity verification.")
    scoring_round_ids: list[str] = Field(default_factory=list, description="Optional completed historical training rounds to score without weight updates: exact frozen samples on each round's verified before/after checkpoints, prompt/continuation loss and target/EOS probabilities. Empty disables this diagnostic; does not select teaching content or change the current checkpoint.")

    @model_validator(mode="after")
    def training_mix(self):
        if self.comparison_checkpoint == "input" and not self.comparison_round_id:
            raise ValueError("Input comparison requires comparison_round_id")
        if self.train_epochs > 0 and self.token_mix.teacher == self.token_mix.corpus == self.token_mix.replay == 0:
            raise ValueError("Enabled training requires token shares summing to 1; all-zero shares require train_epochs=0")
        return self


class Revision(Record):
    id: str
    text: str
    training_text: str = Field(description="Exact teaching sequence for full_text/prompt_prefix; include task/context when needed. Unused for chat_response: set empty and supply training_response.")
    training_tokenization: Literal["full_text", "prompt_prefix", "chat_response"] = Field(default="full_text", description="full_text tokenizes training_text; prompt_prefix preserves literal student_prompt tokens and separately encodes its suffix. chat_response serializes the native no-thinking chat prefix for student_prompt, training_response, and native assistant terminator; training_text is unused in that mode. All use full-sequence loss.")
    training_response: str = Field(default="", description="Exact assistant continuation for chat_response only; do not include the user prompt, role markers or thinking prefill. Empty responses are an explicit teaching choice.")
    errors: list[str]
    evidence: list[str]
    use_for_training: bool
    reason: str


class Revisions(Record):
    rows: list[Revision]


class MaterialSelection(Record):
    manifest_hash: str
    accepted_ids: list[str]
    accepted_jobs: list[str] = Field(description="Exact jobs[].plan_id names whose immutable candidates are all selected, never jobs[].job_id artifact hashes. Empty is valid; no obligation to accept any job.")
    edits: list[MaterialEdit]
    seed_exclusions: list[str]
    token_mix: CurriculumTokenMix
    train_epochs: int = Field(ge=0)
    review_scope: str = Field(min_length=1, description="Describe the items/batches inspected and any deliberate sampling policy; never claim observation of unattempted student prompts")
    reason: str = Field(min_length=1, description="Selection rationale; all unselected candidates are omitted, with full originals preserved")

    @model_validator(mode="after")
    def recipe(self):
        if self.train_epochs and not sum(self.token_mix.model_dump().values()):
            raise ValueError("Positive training needs nonzero material shares")
        return self


class AssessmentCriterion(Record):
    name: str = Field(min_length=1)
    rubric: str = Field(min_length=1)


class Evaluation(Record):
    id: str = Field(min_length=1)
    sources: list[SourceRef]
    question: str
    student_prompt: str = Field(min_length=1)
    student_format: Literal["campaign", "raw_text", "chat_template"] = "campaign"
    reference: str
    rubric: list[str]
    concept: str
    evidence: str
    compare_before: bool = Field(default=False, description="Also answer this exact online question using this round's verified input checkpoint, under the same interface and generation settings. Optional evidence, not an acceptance gate.")
    dimensions: list[AssessmentCriterion] = Field(default_factory=list, description="Optional teacher-defined scoring dimensions, frozen before answers. Separate substance, procedure or presentation when useful; no prescribed names.")

    @model_validator(mode="after")
    def unique_dimensions(self):
        if len({d.name for d in self.dimensions}) != len(self.dimensions):
            raise ValueError("Assessment dimension names must be unique")
        return self


class Evaluations(Record):
    rows: list[Evaluation]


class DimensionGrade(Record):
    name: str
    score: float = Field(ge=0, le=1)
    feedback: str


class Grade(Record):
    id: str
    score: float = Field(ge=0, le=1)
    verdict: Literal["correct", "partial", "incorrect"]
    feedback: str
    gap_type: str
    needs_practice: bool
    priority: float = Field(ge=0, le=1)
    dimensions: list[DimensionGrade] = Field(default_factory=list, description="One score and explanation for each dimension frozen with this question; empty if none were requested. Overall score remains your judgment.")


class Grades(Record):
    rows: list[Grade]


class Reflection(Record):
    student_notes: str
    next_round_instructions: str
    action: Literal["continue", "pause", "complete"]
    reason: str
    experiment_review: ExperimentReview | None = Field(default=None, description="Interpret this round's recorded experiment without rewriting its plan; null if none or no conclusion is ready")
