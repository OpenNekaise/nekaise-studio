"""Operator-owned progression, separate from teacher-owned teaching content."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

LearningTrack = Literal["unspecified", "corpus", "gpc", "remediation"]


class CurriculumLoop(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    policy: Literal["progressive_v1"] = "progressive_v1"
    namespace: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,100}$")
    projection_artifact: str = Field(pattern=r"^[0-9a-f]{64}$")
    forward_corpus_share: float = Field(default=0.5, gt=0, lt=1, description="Initial hint; Teacher chooses corpus share among forward targets each round")
    remediation_cap: float = Field(default=0.20, ge=0, lt=0.5)
    corpus_window_chars: int = Field(default=262144, ge=1024, le=64000000)
    span_chars: int = Field(default=2400, ge=128, le=12000)
    raw_target_tokens: int = Field(default=0, ge=0, le=16000000, description="Runtime Teacher window allocation; 0 retains the legacy ratio-derived amount")
    web_training: bool = False
    web_target_tokens: int = Field(default=0, ge=0, le=16000000)


class TeachingCycle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    policy: Literal["buffered_v1", "continuous_v1"] = "buffered_v1"
    block_target_tokens: int = Field(default=131072, ge=1024, le=16000000)
    blocks_per_cycle: int = Field(default=4, ge=1, le=8)
    initial_blocks_per_cycle: int = Field(default=2, ge=1, le=8)
    prefetch_blocks: int = Field(default=2, ge=1, le=2)
    max_author_calls_per_cycle: int = Field(default=64, ge=1, le=512)
    max_author_output_tokens_per_cycle: int = Field(default=524288, ge=256, le=4194304)
    train_timeout_seconds: int = Field(default=3600, ge=60, le=7200, description="Continuous GPU window deadline only; Teacher/Author deadlines and usage budgets remain unchanged")


class WebTrainingPermission(BaseModel):
    """Teacher's explicit source eligibility decision, backed by retrieved evidence."""
    model_config = ConfigDict(extra="forbid")
    license: Literal["CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "CC-BY-3.0", "CC-BY-SA-3.0", "public-domain", "MIT", "Apache-2.0", "PSF-2.0"]
    license_url: str = Field(min_length=10, max_length=2048)
    evidence_quote: str = Field(min_length=12, max_length=1000)
    scope_reason: str = Field(min_length=20, max_length=2000)
    collection_prefix: str = Field(default="", max_length=2048, description="Optional same-origin path prefix licensed by this evidence. Traverse ordinary page links only inside this prefix; empty selects just the requested page.")
    max_pages: int = Field(default=1, ge=1, le=256)

class ResearchSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(min_length=10, max_length=2048)
    title: str = Field(min_length=1, max_length=500)
    purpose: str = Field(min_length=1, max_length=2000)
    training: WebTrainingPermission | None = Field(default=None, description="Null means reference-only. Supply a verified reuse license and its actual evidence before selecting source prose for training. Never select excluded general datasets or benchmark collections.")


class ResearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    queries: list[str] = Field(min_length=1, max_length=6)
    sources: list[ResearchSource] = Field(min_length=1, max_length=4)
    teaching_direction: str = Field(min_length=1, max_length=4000)
