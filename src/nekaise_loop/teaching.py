"""Teacher-owned teaching decisions; validation describes executable data shapes."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import TokenMix


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
    kind: Literal["cpt", "sft"]
    sources: list[SourceRef]
    concept: str
    prompt: str
    student_prompt: str = Field(min_length=1, description="Exact text sent to student, including any desired context")
    reason: str


class Curriculum(Record):
    lessons: list[Lesson]
    readings: list[SourceRef]
    replay: list[ReplayRef]
    token_mix: TokenMix
    train_epochs: int = Field(ge=0, description="0 keeps weights unchanged; positive values train this many dataset passes")
    evaluation_instructions: str
    notes: str


class Revision(Record):
    id: str
    text: str
    training_text: str = Field(description="Exact teaching sequence for training; include task/context when needed")
    errors: list[str]
    evidence: list[str]
    use_for_training: bool
    reason: str


class Revisions(Record):
    rows: list[Revision]


class Evaluation(Record):
    id: str = Field(min_length=1)
    sources: list[SourceRef]
    question: str
    student_prompt: str = Field(min_length=1)
    reference: str
    rubric: list[str]
    concept: str
    evidence: str


class Evaluations(Record):
    rows: list[Evaluation]


class Grade(Record):
    id: str
    score: float = Field(ge=0, le=1)
    verdict: Literal["correct", "partial", "incorrect"]
    feedback: str
    gap_type: str
    needs_practice: bool
    priority: float = Field(ge=0, le=1)


class Grades(Record):
    rows: list[Grade]


class Reflection(Record):
    student_notes: str
    next_round_instructions: str
    action: Literal["continue", "pause", "complete"]
    reason: str
