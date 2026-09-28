"""Import a curriculum authoring document, not its proposed runtime or benchmarks."""
import hashlib
import json
from pathlib import Path

from .artifacts import Artifacts
from .curriculum_types import CurriculumLoop
from .storage import new_id

UNIT_FIELDS = ("title", "scope", "key_concepts", "learning_objectives", "search_queries",
               "search_keywords", "teaching_strategy", "practice_specs", "common_errors", "prerequisites")


def project_curriculum(path: Path):
    raw = path.read_bytes()
    source = json.loads(raw)
    units, domains = {}, []
    for domain_id, domain in source["taxonomy"].items():
        domains.append({"id": domain_id, "title": domain["title"], "scope": domain["scope"]})
        for name, unit in domain["subdomains"].items():
            unit_id = domain_id + "." + name
            if unit.get("unit_id", unit_id) != unit_id:
                raise ValueError("Curriculum unit ID differs from its taxonomy path")
            units[unit_id] = {"id": unit_id, "domain_id": domain_id,
                              **{k: unit[k] for k in UNIT_FIELDS if k in unit}}
    order = source["learning_order"]
    if not units or len(order) != len(units) or set(order) != units.keys():
        raise ValueError("Learning order must include every unit exactly once")
    return {"format": "general_teaching_projection_v1", "source_sha256": hashlib.sha256(raw).hexdigest(),
            "title": source["title"], "domains": domains, "units": [units[k] for k in order],
            "semantics": "Ordered exposure cycles; no mastery, benchmark or prerequisite acceptance gates"}


def import_curriculum(workspace, path, *, namespace=None, **options):
    projection = project_curriculum(Path(path))
    key = Artifacts(workspace).put(projection)
    return CurriculumLoop(namespace=namespace or new_id("curriculum"), projection_artifact=key, **options)
