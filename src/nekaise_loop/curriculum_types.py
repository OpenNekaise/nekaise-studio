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
    corpus_window_chars: int = Field(default=262144, ge=1024, le=2000000)
    span_chars: int = Field(default=2400, ge=128, le=12000)
    raw_target_tokens: int = Field(default=0, ge=0, le=2000000, description="Runtime Teacher block allocation; 0 retains the legacy ratio-derived amount")


class TeachingCycle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    policy: Literal["buffered_v1"] = "buffered_v1"
    block_target_tokens: int = Field(default=131072, ge=1024, le=1048576)
    blocks_per_cycle: int = Field(default=4, ge=1, le=8)
    initial_blocks_per_cycle: int = Field(default=2, ge=1, le=8)
    prefetch_blocks: int = Field(default=2, ge=1, le=2)
    max_author_calls_per_cycle: int = Field(default=64, ge=1, le=512)
    max_author_output_tokens_per_cycle: int = Field(default=524288, ge=256, le=4194304)

class ResearchSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(min_length=10, max_length=2048)
    title: str = Field(min_length=1, max_length=500)
    purpose: str = Field(min_length=1, max_length=2000)


class ResearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    queries: list[str] = Field(min_length=1, max_length=6)
    sources: list[ResearchSource] = Field(min_length=1, max_length=4)
    teaching_direction: str = Field(min_length=1, max_length=4000)
