"""Versioned student identity snapshots, separate from checkpoint iteration numbers."""
from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field


class StudentIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1, max_length=80)
    version: str = Field(strict=True, pattern=r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$", max_length=32)
    developer: str = Field(min_length=1, max_length=160)
    origin: str = Field(min_length=1, max_length=160)
    foundation_model: str = Field(min_length=1, max_length=240)
    charter: str = Field(min_length=1, max_length=18000, description="Exact character charter snapshot; never read from a mutable file during execution")

    def public_metadata(self):
        return {**self.model_dump(exclude={"charter"}), "display_name": f"{self.name} {self.version}",
                "charter_sha256": hashlib.sha256(self.charter.encode()).hexdigest()}


def validate_identity_update(before, after):
    """A named identity version cannot silently acquire a different contract."""
    if before and after and (before.name, before.version) == (after.name, after.version) and before != after:
        raise ValueError("Changing an identity contract requires a new major.minor version")
