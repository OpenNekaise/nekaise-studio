"""Validated campaign recipes and machine-local execution settings."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parents[2]
STAGES = ("select", "plan", "draft", "revise", "gate", "freeze", "train", "evaluate", "answer", "grade", "adapt")
STAGE_LABELS = dict(zip(STAGES, ("Teacher curriculum", "Prepare teacher tasks", "Student attempts", "Teacher revisions", "Teacher training choices", "Freeze dataset", "Train student", "Teacher assessment design", "Student evaluation", "Teacher judgment", "Teacher reflection & next steps")))


class TokenMix(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    teacher: float = Field(default=0.6, ge=0, le=1)
    corpus: float = Field(default=0.2, ge=0, le=1)
    replay: float = Field(default=0.2, ge=0, le=1)

    @model_validator(mode="after")
    def total(self):
        if abs(self.teacher + self.corpus + self.replay - 1) > 1e-6:
            raise ValueError("Token shares must sum to 1")
        return self


class CampaignConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    student_model: str = Field(default="openbmb/MiniCPM5-1B-Base", min_length=1, max_length=500)
    teacher_provider: Literal["claude", "codex"] = "codex"
    teacher_model: str = Field(default="gpt-5.6-terra", min_length=1, max_length=150)
    corpus_path: str = "../nekaise-corpus"
    focus: str = Field(default="building energy heat transfer", min_length=1, max_length=240)
    source_prefix: str = Field(default="crawl-energyplus-docs", max_length=150, description="Initial search suggestion, not a corpus restriction")
    rounds: int = Field(default=-1, ge=-1, le=100)
    lessons_per_round: int = Field(default=4, ge=0, le=32, description="Suggested count; the teacher decides actual tasks")
    eval_questions: int = Field(default=4, ge=0, le=24, description="Suggested count; the teacher decides actual assessment")
    train_steps: int = Field(default=0, ge=0, le=10000, description="0 derives updates from token budget; positive values are explicit overrides")
    train_epochs: int = Field(default=1, ge=0, le=10, description="Suggested passes; teacher chooses, including 0 for no weight updates")
    tokens_per_update: int = Field(default=2048, ge=64, le=65536)
    warmup_tokens: int = Field(default=8192, ge=0, le=10000000)
    inherit_optimizer: bool = True
    learning_rate: float = Field(default=1e-5, ge=1e-7, le=0.01)
    token_mix: TokenMix = Field(default_factory=TokenMix)
    max_seq_len: int = Field(default=512, ge=64, le=4096)
    max_new_tokens: int = Field(default=192, ge=16, le=1024)
    passage_chars: int = Field(default=2400, ge=400, le=6000)
    replay_fraction: float = Field(default=0.2, ge=0, le=0.5)
    max_stage_seconds: int = Field(default=900, ge=30, le=7200)
    max_teacher_calls: int = Field(default=-1, ge=-1, le=2000, description="-1 has no local call cap; Resume renews a positive allowance")
    auto_recover: bool = True
    teacher_retry_seconds: int = Field(default=1800, ge=30, le=86400)
    orchestrator_provider: Literal["codex", "claude"] = "codex"
    orchestrator_model: str = Field(default="gpt-6-astra", min_length=1, max_length=150)
    orchestrator_timeout: int = Field(default=1800, ge=30, le=7200)
    max_repair_attempts: int = Field(default=3, ge=1, le=10)
    seed: int = Field(default=3407, ge=0, le=2**31-1)

    @field_validator("rounds", "max_teacher_calls")
    @classmethod
    def unlimited_or_positive(cls, value):
        if value == 0:
            raise ValueError("Use -1 for unlimited, or a positive number")
        return value

    @field_validator("student_model", "teacher_model", "orchestrator_model", "corpus_path", "focus", "source_prefix")
    @classmethod
    def no_control_characters(cls, value: str) -> str:
        if any(ord(c) < 32 for c in value):
            raise ValueError("Control characters are not allowed")
        return value.strip()


class Settings:
    def __init__(self, workspace: str | Path | None = None):
        self.root = ROOT
        self.workspace = Path(workspace or os.environ.get("NEKAISE_LOOP_WORKSPACE", ROOT / "workspace")).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        default_ml = Path.home() / "miniconda3/envs/nekaise-studio/bin/python"
        self.model_python = os.environ.get("NEKAISE_MODEL_PYTHON", str(default_ml))
        self.claude = os.environ.get("NEKAISE_CLAUDE", "claude")
        self.codex = os.environ.get("NEKAISE_CODEX", "codex")
        unit = "nekaise-loop-supervisor.service"
        installed = self.workspace == ROOT/"workspace" and (Path.home()/".config/systemd/user"/unit).is_file()
        self.supervisor_service = os.environ.get("NEKAISE_LOOP_SUPERVISOR_SERVICE", unit if installed else "")
        self.dashboard = ROOT / "dashboard/dist"


def resolve_student(reference: str) -> str:
    path = Path(reference).expanduser()
    if path.is_dir():
        return str(path.resolve())
    hub = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", Path(os.environ.get("HF_HOME", Path.home()/".cache/huggingface"))/"hub"))
    model = hub / ("models--" + reference.replace("/", "--"))
    ref = model / "refs/main"
    if ref.is_file():
        snapshot = model / "snapshots" / ref.read_text().strip()
        if snapshot.is_dir():
            return str(snapshot.resolve())
    raise RuntimeError(f"Student {reference!r} is not available locally. Provide a cached model or a checkpoint directory.")
