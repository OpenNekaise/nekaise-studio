"""Teacher delegation and auxiliary text; no fabricated student observations."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError


class MaterialRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExpansionJob(MaterialRecord):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    author_id: str
    seed_ids: list[str]
    reading_indices: list[int] = Field(default_factory=list)
    instructions: str = Field(min_length=1)
    expected_items: int = Field(ge=1)
    max_output_tokens: int = Field(default=4096, ge=128, le=131072)


class Candidate(MaterialRecord):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    kind: Literal["cpt", "sft"]
    concept: str = Field(min_length=1)
    student_prompt: str = ""
    training_text: str = ""
    training_response: str = ""
    training_tokenization: Literal["full_text", "chat_response", "prompt_prefix"] = "chat_response"
    source_keys: list[str] = Field(default_factory=list)
    seed_ids: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def executable_text(self):
        if self.training_tokenization == "chat_response":
            if not self.student_prompt.strip():
                raise PydanticCustomError("chat_prompt_required", "Native chat material needs a user prompt")
        elif not self.training_text.strip():
            raise PydanticCustomError("training_text_required", "Text material needs an explicit training sequence")
        if self.training_tokenization == "prompt_prefix" and (not self.student_prompt or not self.training_text.startswith(self.student_prompt)):
            raise PydanticCustomError("prompt_prefix_mismatch", "Prompt-prefix material must preserve the exact prompt")
        return self


class CandidateBatch(MaterialRecord):
    rows: list[Candidate]

    @model_validator(mode="after")
    def ids(self):
        if len({r.id for r in self.rows}) != len(self.rows):
            raise ValueError("Author returned duplicate candidate IDs")
        return self


class MaterialEdit(MaterialRecord):
    candidate_id: str
    replacement: Candidate
