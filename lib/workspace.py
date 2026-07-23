"""First-class local workspace overlay for Nekaise Studio.

The repository is the shared, reviewable bootloader.  An external workspace owns every
user-specific recipe, configuration, skill, run record, dataset, and checkpoint.  The
legacy in-repository layout remains the default so existing installations and campaigns
continue to work without migration.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

ENV = "NEKAISE_WORKSPACE"
REPO_ENV = "NEKAISE_STUDIO_REPO"
MANIFEST = ".nekaise-workspace.json"
SCHEMA_VERSION = 1
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _legacy_id(repo: Path) -> str:
    digest = hashlib.sha256(str(repo).encode()).hexdigest()[:16]
    return f"legacy-{digest}"


def validate_name(name: str, kind: str = "name") -> str:
    """Keep agent-controlled names inside one workspace directory."""
    if not _NAME.fullmatch(name):
        raise ValueError(
            f"invalid {kind} {name!r}; use letters, numbers, '.', '_' or '-'")
    return name


@dataclass(frozen=True)
class Workspace:
    repo: Path
    root: Path
    workspace_id: str
    external: bool
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def legacy(cls, repo: str | Path) -> "Workspace":
        repo_path = Path(repo).resolve()
        return cls(
            repo=repo_path, root=repo_path, workspace_id=_legacy_id(repo_path),
            external=False,
        )

    @classmethod
    def resolve(
        cls, repo: str | Path, root: str | Path | None = None,
    ) -> "Workspace":
        repo_path = Path(repo).resolve()
        selected = root if root is not None else os.environ.get(ENV)
        if selected is None or not str(selected).strip():
            return cls.legacy(repo_path)
        root_path = Path(selected).expanduser().resolve()
        manifest_path = root_path / MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"{root_path} is not an initialized Nekaise workspace; run "
                f"`python -m studio.cli workspace init {root_path}`")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported workspace schema at {manifest_path}: "
                f"{manifest.get('schema_version')!r}")
        workspace_id = manifest.get("workspace_id")
        if not isinstance(workspace_id, str) or not workspace_id:
            raise ValueError(f"workspace manifest has no workspace_id: {manifest_path}")
        return cls(
            repo=repo_path, root=root_path, workspace_id=workspace_id,
            external=True, schema_version=manifest["schema_version"],
        )

    @classmethod
    def initialize(
        cls, repo: str | Path, root: str | Path, *, copy_configs: bool = True,
    ) -> "Workspace":
        """Create missing workspace structure without overwriting user files."""
        repo_path = Path(repo).resolve()
        root_path = Path(root).expanduser().resolve()
        if root_path == repo_path:
            raise ValueError(
                "the repository already provides the legacy workspace; choose a "
                "separate directory for an external workspace")
        try:
            inside_repo = root_path.relative_to(repo_path)
        except ValueError:
            inside_repo = None
        if inside_repo is not None and inside_repo.parts[0] not in {
            "workspace", "user-workspace",
        }:
            raise ValueError(
                "an external workspace inside the repository must live under "
                "`workspace/` or `user-workspace/`, which are git-ignored")
        root_path.mkdir(parents=True, exist_ok=True)
        manifest_path = root_path / MANIFEST
        if manifest_path.exists():
            workspace = cls.resolve(repo_path, root_path)
        else:
            record = {
                "schema_version": SCHEMA_VERSION,
                "workspace_id": f"ws-{secrets.token_hex(12)}",
                "created": time.time(),
                "studio": "nekaise-studio",
            }
            _atomic_json(manifest_path, record)
            workspace = cls.resolve(repo_path, root_path)

        for directory in (
            workspace.store_dir,
            workspace.experiments_dir,
            workspace.configs_dir,
            workspace.data_dir,
            workspace.local_skills_dir,
            workspace.extensions_dir,
            workspace.scratch_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        if copy_configs:
            for source in sorted((repo_path / "configs").glob("*.yaml")):
                destination = workspace.configs_dir / source.name
                if not destination.exists():
                    shutil.copy2(source, destination)

        agents = root_path / "AGENTS.md"
        if not agents.exists():
            agents.write_text(_agents_template(repo_path))
        ignore = root_path / ".gitignore"
        if not ignore.exists():
            ignore.write_text(_gitignore_template())
        return workspace

    @property
    def experiments_dir(self) -> Path:
        return self.root / "experiments" if self.external else self.repo / "experiments"

    @property
    def store_dir(self) -> Path:
        return self.root / ".studio" if self.external else self.experiments_dir / ".studio"

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs" if self.external else self.repo / "configs"

    @property
    def local_skills_dir(self) -> Path:
        return self.root / "skills" / "local" if self.external else self.repo / "skills" / "local"

    @property
    def data_dir(self) -> Path:
        return self.root / "nekaise_data" if self.external else self.repo / "nekaise_data"

    @property
    def extensions_dir(self) -> Path:
        return self.root / "extensions" if self.external else self.repo / "workspace" / "extensions"

    @property
    def scratch_dir(self) -> Path:
        return self.root / "scratch" if self.external else self.repo / "workspace"

    def experiment_dir(self, name: str) -> Path:
        return self.experiments_dir / validate_name(name, "experiment")

    def resolve_config(self, value: str | Path) -> Path:
        """Resolve user overlays first, then immutable core templates."""
        path = Path(value).expanduser()
        if path.is_absolute():
            return path.resolve()
        candidates = []
        if self.external:
            candidates.extend((self.root / path, self.configs_dir / path.name))
        candidates.append(self.repo / path)
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return candidates[0].resolve()

    def resolve_input(self, value: str | Path) -> Path:
        """Resolve data paths relative to workspace first and core repo second."""
        path = Path(value).expanduser()
        if path.is_absolute():
            return path.resolve()
        candidates = [self.root / path] if self.external else []
        candidates.append(self.repo / path)
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()

    def resolve_experiment_file(self, experiment: str, filename: str) -> Path:
        validate_name(experiment, "experiment")
        validate_name(filename, "filename")
        candidates = []
        if self.external:
            candidates.append(self.experiment_dir(experiment) / filename)
        candidates.append(self.repo / "experiments" / experiment / filename)
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return candidates[0].resolve()

    def provenance(self) -> dict:
        return {
            "workspace_id": self.workspace_id,
            "workspace_schema_version": self.schema_version,
            "workspace_mode": "external" if self.external else "legacy",
        }

    def apply_environment(self) -> "Workspace":
        """Expose shared paths to subprocesses and existing gym adapters."""
        os.environ[REPO_ENV] = str(self.repo)
        if self.external:
            os.environ.setdefault("NEKAISE_DATA", str(self.data_dir))
        return self

    def describe(self) -> dict:
        return {
            **self.provenance(),
            "root": str(self.root),
            "repo": str(self.repo),
            "store": str(self.store_dir),
            "experiments": str(self.experiments_dir),
            "configs": str(self.configs_dir),
            "data": str(self.data_dir),
            "local_skills": str(self.local_skills_dir),
            "extensions": str(self.extensions_dir),
            "scratch": str(self.scratch_dir),
        }


def _agents_template(repo: Path) -> str:
    return f"""# AGENTS.md — Nekaise user workspace

This directory is the mutable, user-owned overlay for the Nekaise Studio bootloader at:

    {repo}

Rules for coding agents:

- Read the bootloader's `AGENTS.md`, `SPEC.md`, `STATUS.md`, and core skills first.
- Treat the bootloader as read-only during experiment work.
- Put experiment recipes and runtime data in `experiments/<name>/`.
- Put proprietary building inputs in `nekaise_data/`.
- Put config overlays in `configs/`; never change a copied `frozen:` section.
- Put emergent skills in `skills/local/` and private integrations in `extensions/`.
- Query runs through `python -m studio.cli --workspace <this-dir> ...`; do not scan run
  directories or parse logs as the primary interface.
- A core contribution is a separate, explicit promotion/PR action. Never copy local
  paths, secrets, proprietary data, run artifacts, or checkpoints into that PR.
"""


def _gitignore_template() -> str:
    return """# Runtime scientific state stays local even if this workspace uses Git.
.studio/
experiments/**/data/
experiments/**/runs/
scratch/
*.safetensors
*.gguf
*.pt
*.bin
.env
"""
