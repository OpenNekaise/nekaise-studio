"""Operator-owned progression, separate from teacher-owned teaching content."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    web_training_policy: Literal["license_evidence_v1", "teacher_selected_v1"] = Field(default="license_evidence_v1", description="Historical license admission or operator-authorized Teacher source selection without license checks")
    web_crawl_policy: Literal["bounded_v1", "inventory_v1"] = "bounded_v1"
    web_target_tokens: int = Field(default=0, ge=0, le=16000000)

    @model_validator(mode='after')
    def inventory_policy(self):
        if self.web_crawl_policy=='inventory_v1' and (not self.web_training or self.web_training_policy!='teacher_selected_v1'):
            raise ValueError('Inventory crawling requires enabled Teacher-selected webpage training')
        return self


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


class WebTrainingSelection(BaseModel):
    """Bounded source selection; optional license metadata never grants permission."""
    model_config = ConfigDict(extra="forbid")
    collection_prefix: str = Field(default="", max_length=2048)
    max_pages: int = Field(default=1, ge=1, le=256)
    seed_urls: list[str] = Field(default_factory=list, max_length=4, description="Additional table-of-contents/index pages inside collection_prefix; discovery only, not mandatory training targets")
    sitemap_urls: list[str] = Field(default_factory=list, max_length=4, description="Explicit same-origin XML sitemaps; only page URLs inside collection_prefix are eligible")
    max_seconds: int = Field(default=120, ge=10, le=300, description="Bounded acquisition work per collection invocation; unfinished frontier persists")
    license: str = Field(default="", max_length=500)
    license_url: str = Field(default="", max_length=2048)
    evidence_quote: str = Field(default="", max_length=1000)
    scope_reason: str = Field(default="", max_length=2000)


class WebTrainingPermission(WebTrainingSelection):
    """Teacher's explicit source eligibility decision, backed by retrieved evidence."""
    model_config = ConfigDict(extra="forbid")
    license: Literal["CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "CC-BY-3.0", "CC-BY-SA-3.0", "public-domain", "MIT", "Apache-2.0", "PSF-2.0"]
    license_url: str = Field(min_length=10, max_length=2048)
    evidence_quote: str = Field(min_length=12, max_length=1000)
    scope_reason: str = Field(min_length=20, max_length=2000)
    collection_prefix: str = Field(default="", max_length=2048, description="Optional same-origin path prefix licensed by this evidence. Traverse ordinary page links only inside this prefix; empty selects just the requested page.")
    max_pages: int = Field(default=1, ge=1, le=256)

class ResearchLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(min_length=10, max_length=2048)
    title: str = Field(min_length=1, max_length=500)
    training: WebTrainingSelection | None = Field(default=None, description="Under teacher_selected_v1, every nominated source is available as training prose; null uses a single page, or supply collection_prefix/max_pages. License metadata is optional and not checked. Historical license_evidence_v1 requires the full permission evidence. Excluded datasets and benchmarks remain excluded.")


class ResearchSource(ResearchLocation):
    purpose: str = Field(min_length=1, max_length=2000)
    alternatives: list[ResearchLocation] = Field(default_factory=list, max_length=3, description="Teacher-preauthorized alternatives for exactly this teaching purpose; each has its own acquisition scope")


class ResearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    queries: list[str] = Field(default_factory=list, max_length=6)
    sources: list[ResearchSource] = Field(default_factory=list, max_length=4)
    reuse_source_ids: list[str] = Field(default_factory=list, max_length=4, description="Exact IDs from the supplied persistent source catalog; reuse only sources relevant to this unit")
    teaching_direction: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def has_sources(self):
        if not self.sources and not self.reuse_source_ids:
            raise ValueError("Research requires new sources or explicit catalog references")
        return self
