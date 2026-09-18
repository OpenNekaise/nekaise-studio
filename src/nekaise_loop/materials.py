"""Teacher-owned delegation and exact candidate selection before freezing."""
from __future__ import annotations

from collections import Counter
import copy

from .artifacts import digest
from .material_jobs import run_jobs
from .material_types import Candidate
from .teaching import MaterialSelection


def validate_expansion_plan(config, curriculum):
    """Enforce the operator's workflow requirement, without judging teaching content."""
    jobs = curriculum.get("expansion_jobs", [])
    if config.expansion_policy == "required_v1" and curriculum["train_epochs"] > 0 and not jobs:
        raise ValueError("Positive training requires Material Author expansion_jobs under required_v1; only train_epochs=0 diagnostics may skip expansion")
    limits = config.material_authors
    if len(jobs) > limits.max_calls_per_round or sum(j["max_output_tokens"] for j in jobs) > limits.max_output_tokens_per_round:
        raise ValueError("Teacher expansion plan exceeds the declared author allowance")
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Expansion job IDs must be unique")
    authors = {a.id for a in limits.authors}
    seeds = {r["id"] for r in curriculum["lessons"]}
    for job in jobs:
        if job["author_id"] not in authors or not set(job["seed_ids"]) <= seeds:
            raise ValueError("Expansion requested an unknown author or seed")
        if any(i < 0 or i >= len(curriculum["readings"]) for i in job["reading_indices"]):
            raise ValueError("Expansion requested an unknown reading index")


def expansion_receipt(ctx, curriculum, frozen):
    if ctx.config.expansion_policy != "required_v1" or curriculum["train_epochs"] == 0:
        return None
    validate_expansion_plan(ctx.config, curriculum)
    manifest = ctx.output("expand")
    candidates = manifest["candidates"]
    selected_ids = {r["id"] for r in ctx.output("material_select")["materials"] if r["use_for_training"]}
    targets = sum(len(s["input_ids"])-1 for s in frozen["samples"] if s["row_id"] in selected_ids and s["stream"] == "teacher")
    if not manifest["jobs"] or not candidates or not selected_ids or not targets:
        raise ValueError("Positive training requires completed expansion and teacher-selected expanded training targets; revise the package or explicitly choose train_epochs=0")
    return {"policy": "required_v1", "round_id": ctx.round["id"],
            "manifest_hash": manifest["manifest_hash"], "job_ids": [j["job_id"] for j in manifest["jobs"]],
            "produced_candidates": len(candidates), "selected_candidates": len(selected_ids),
            "prepared_expanded_targets_per_pass": targets}


def expand(ctx):
    selected = ctx.output("select")
    validate_expansion_plan(ctx.config, selected["curriculum"])
    jobs = selected["curriculum"].get("expansion_jobs", [])
    if not jobs:
        return {"jobs": [], "candidates": [], "manifest_hash": digest([])}
    seeds = {r["id"]: r for r in ctx.output("revise")["lessons"]}
    seed_artifact = ctx.store.one("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='revise' AND status='complete' ORDER BY id DESC LIMIT 1", (ctx.round["id"],))["artifact"]
    specs = []
    for job in jobs:
        sources, examples = {}, []
        for seed_id in job["seed_ids"]:
            seed = seeds[seed_id]
            keys = []
            for source in seed["sources"]:
                key = digest(source)
                sources[key] = source
                keys.append(key)
            examples.append({k: seed.get(k) for k in ("id", "kind", "concept", "student_prompt", "teacher", "training_text", "training_response", "training_tokenization")} | {"source_keys": keys})
        for i in job["reading_indices"]:
            source = selected["readings"][i]
            sources[digest(source)] = source
        specs.append({"job": job, "teacher_plan": selected["curriculum"]["notes"],
                      "seed_artifact": seed_artifact, "seeds": examples, "sources": sources})
    results = run_jobs(ctx, specs)
    rows = []
    for result in results:
        for candidate in result["rows"]:
            # ':' cannot occur in either component, so different job/candidate
            # pairs cannot collapse to one ID when their names contain '-'.
            rows.append({"id": f"material-{result['plan_id']}:{candidate['id']}", "candidate": candidate,
                         "job_id": result["job_id"], "plan_id": result["plan_id"],
                         "author_id": result["author_id"], "model": result["model"],
                         "response_artifact": result["response_artifact"], "seed_artifact": result["seed_artifact"],
                         "sources": {k: result["sources"][k] for k in candidate["source_keys"]}})
    counts = Counter(digest({k: r["candidate"][k] for k in ("student_prompt", "training_text", "training_response")}) for r in rows)
    for row in rows:
        row["checks"] = {"exact_content_occurrences": counts[digest({k: row["candidate"][k] for k in ("student_prompt", "training_text", "training_response")})],
                         "student_observation": "not_requested", "semantic_correctness": "teacher_decides"}
    manifest = {"jobs": [{k: v for k, v in result.items() if k not in {"rows", "sources"}} | {"candidate_count": len(result["rows"])} for result in results], "candidates": rows}
    return {**manifest, "manifest_hash": digest(manifest)}


def selection_brief(ctx, manifest):
    # The prompt carries a bounded preview, never a context-dependent teaching
    # cutoff. The paged archive exposes every item, including omitted previews.
    summaries = [{"id": r["id"], "author_id": r["author_id"], "plan_id": r["plan_id"],
                  "concept": r["candidate"]["concept"], "kind": r["candidate"]["kind"],
                  "characters": sum(len(r["candidate"][k]) for k in ("student_prompt", "training_text", "training_response")),
                  "preview": (r["candidate"]["training_response"] or r["candidate"]["training_text"])[:240],
                  "checks": r["checks"]} for r in manifest["candidates"]]
    return {"manifest_hash": manifest["manifest_hash"], "round_id": ctx.round["id"],
            "jobs": manifest["jobs"], "candidate_count": len(summaries), "candidate_preview": summaries[:40],
            "full_candidates": {"op": "material_candidates", "round_id": ctx.round["id"], "offset": 0, "limit": 20},
            "curriculum": ctx.output("select")["curriculum"],
            "seed_choices": [{"id": r["id"], "use_for_training": r["use_for_training"], "reason": r["reason"]} for r in ctx.output("revise")["lessons"]]}


def select_materials(ctx):
    manifest = ctx.output("expand")
    curriculum = copy.deepcopy(ctx.output("select")["curriculum"])
    lessons = ctx.output("revise")["lessons"]
    if not curriculum.get("expansion_jobs"):
        return {"curriculum": curriculum, "lessons": lessons, "selection": None, "materials": []}
    choice = MaterialSelection.model_validate(ctx.teacher.select_materials(selection_brief(ctx, manifest))).model_dump()
    if choice["manifest_hash"] != manifest["manifest_hash"]:
        raise ValueError("Teacher material choice does not match the frozen candidate manifest")
    by_id = {r["id"]: r for r in manifest["candidates"]}
    if by_id.keys() & {r["id"] for r in lessons}:
        raise ValueError("Synthetic material IDs collide with primary lesson IDs")
    jobs = {r["plan_id"] for r in manifest["jobs"]}
    unknown_jobs = set(choice["accepted_jobs"]) - jobs
    if unknown_jobs:
        raise ValueError(
            f"Teacher accepted_jobs must use manifest plan_id names, not job_id artifact hashes; "
            f"unknown: {sorted(unknown_jobs)}; allowed plan_id values: {sorted(jobs)}"
        )
    selected = set(choice["accepted_ids"])
    edits = {e["candidate_id"]: e["replacement"] for e in choice["edits"]}
    if (len(selected) != len(choice["accepted_ids"]) or len(edits) != len(choice["edits"])
            or not selected <= by_id.keys() or not edits.keys() <= by_id.keys()
            or selected.intersection(edits)):
        raise ValueError("Teacher selected missing, duplicate or conflicting material IDs")
    if not set(choice["seed_exclusions"]) <= {r["id"] for r in lessons}:
        raise ValueError("Teacher excluded an unknown seed")
    selected.update(r["id"] for r in by_id.values() if r["plan_id"] in choice["accepted_jobs"])
    selected.update(edits)
    material_rows = []
    for key, original in by_id.items():
        item = Candidate.model_validate(edits.get(key, original["candidate"])).model_dump()
        # Edited examples may cite any source and seed provided to this job.
        task = ctx.artifacts.get(original["job_id"])["spec"]
        if not set(item["source_keys"]) <= task["sources"].keys() or not set(item["seed_ids"]) <= set(task["job"]["seed_ids"]):
            raise ValueError("Edited material cites unprovided provenance; re-author with explicit sources")
        sources = [task["sources"][k] for k in item["source_keys"]]
        document = dict(sources[0]) if sources else {"id": "", "title": "Authored teaching material", "url": "", "license": "generated", "topic": item["concept"], "source_sha256": "", "text": ""}
        document["selection_reason"] = choice["reason"]
        material_rows.append({"id": key, "kind": item["kind"], "concept": item["concept"],
            "prompt": item["student_prompt"] or item["concept"], "student_prompt": item["student_prompt"],
            "student_format": "chat_template" if item["training_tokenization"] == "chat_response" else "raw_text",
            "student": None, "student_observation": "not_requested", "teacher": item["training_response"] if item["training_tokenization"] == "chat_response" else item["training_text"],
            "training_text": item["training_text"], "training_response": item["training_response"], "training_tokenization": item["training_tokenization"],
            "sources": sources, "document": document, "errors": [], "evidence": [],
            "use_for_training": key in selected, "reason": choice["reason"],
            "material_origin": {"type": "auxiliary_synthetic", "author_id": original["author_id"], "model": original["model"],
                "job_id": original["job_id"], "candidate_id": original["candidate"]["id"], "response_artifact": original["response_artifact"],
                "seed_artifact": original["seed_artifact"], "seed_ids": item["seed_ids"], "teacher_edited": key in edits,
                "manifest_hash": manifest["manifest_hash"], "review_scope": choice["review_scope"]}})
    for row in lessons:
        if row["id"] in choice["seed_exclusions"]:
            row.update(use_for_training=False, reason=choice["reason"])
    curriculum.update(token_mix=choice["token_mix"], train_epochs=choice["train_epochs"])
    ctx.project("material", material_rows)
    ctx.event("material_selection", "Teacher selected the final material package", {"candidates": len(material_rows), "selected": len(selected), "review_scope": choice["review_scope"]})
    return {"curriculum": curriculum, "lessons": lessons + material_rows, "selection": choice, "materials": material_rows}


def source_accounting(frozen):
    origins = {r["id"]: r.get("material_origin", {}).get("author_id", "primary_teacher" if r["stream"] in {"cpt", "sft"} else r["stream"]) for r in frozen.get("rows", [])}
    targets = Counter()
    for sample in frozen.get("samples", []):
        targets[origins[sample["row_id"]]] += len(sample["input_ids"])-1
    return {"basis": "prepared_target_occurrences_per_pass_not_actual_updates", "targets_by_origin": dict(targets)}
