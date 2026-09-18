"""Teacher-authored experimental intent and judgments, never execution gates."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StrategyDraft(Record):
    name: str = Field(min_length=1)
    approach: str = Field(min_length=1, description="Reusable teaching approach; describe your choices, not host-enforced rules")
    parent_version: str = Field(default="", pattern=r"^([0-9a-f]{64})?$", description="Previously recorded strategy version being revised, or empty for an original strategy")


class ExperimentPlan(Record):
    title: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1, description="Your hypothesis before this round's student observations and training")
    intervention: str = Field(min_length=1, description="What you intend to change or continue investigating")
    budget_basis: str = Field(min_length=1, description="Intended resource comparison, or explain why budgets are not matched; not an execution limit")
    observation_plan: str = Field(min_length=1, description="What evidence you intend to inspect; does not freeze evaluation questions or require assessment")
    reconsider_if: str = Field(min_length=1, description="What observations would make you reconsider this hypothesis")
    related_round_ids: list[str] = Field(default_factory=list, description="Earlier teaching rounds providing context; references do not establish a controlled comparison")
    strategy_version: str = Field(default="", pattern=r"^([0-9a-f]{64})?$", description="Reuse a recorded version, or empty when supplying a new strategy")
    strategy: StrategyDraft | None = Field(default=None, description="Define a version when strategy_version is empty; identical content reuses its content hash")

    @model_validator(mode="after")
    def one_strategy(self):
        if bool(self.strategy_version) == (self.strategy is not None):
            raise ValueError("Supply either strategy_version or a strategy definition")
        if len(set(self.related_round_ids)) != len(self.related_round_ids):
            raise ValueError("Related round references must be unique")
        return self


class ExperimentReview(Record):
    status: Literal["ongoing", "concluded", "abandoned"]
    findings: str = Field(min_length=1, description="Teacher interpretation of observed evidence, including null or negative findings")
    limitations: str = Field(min_length=1, description="Uncertainty, changed conditions, missing observations or lack of a controlled comparison")
    next_action: str = Field(min_length=1)
    evaluation_ids: list[str] = Field(default_factory=list, description="This round's graded question IDs supporting your judgment; empty is valid")

