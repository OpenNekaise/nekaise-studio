"""Light Teacher decisions; ordinary stages still own Author, dataset and train evidence."""
import copy
import json

import httpx

from . import stages
from .artifacts import digest
from .author_config import catalog
from .assessment import apply_grades, grade_requests
from .curriculum_progress import assignment, required_authors, validate_plan
from .cycle_store import cycle_for, install_plan
from .cycle_types import CycleResearch, CyclePlan, CycleReview
from .materials import validate_expansion_plan
from .processes import Cancelled
from .storage import now
from .teaching import Evaluation


def units_for(ctx, cycle):
    projection = ctx.artifacts.get(ctx.config.curriculum_loop.projection_artifact)
    start = cycle["contract"]["initial_state"]["gpc_completed"]
    return [projection["units"][(start+i) % len(projection["units"])]
            for i in range(cycle["contract"]["requested_blocks"])]


def research(ctx):
    cycle = cycle_for(ctx.store, ctx.round["id"])
    block = ctx.store.one("SELECT * FROM teaching_blocks WHERE round_id=?", (ctx.round["id"],))
    if block["assignment_artifact"]:
        work = ctx.artifacts.get(block["assignment_artifact"])
    elif block['predecessor_round_id']:
        work = assignment(ctx, prepared_state=cycle['contract']['initial_state'], sequence=block['sequence'],
                          predecessor_round_id=block['predecessor_round_id'])
    else:
        work = assignment(ctx)
    ctx.store.execute("UPDATE teaching_blocks SET assignment_artifact=? WHERE round_id=?", (ctx.artifacts.put(work), ctx.round["id"]))
    units = units_for(ctx, cycle)
    saved = ctx.artifacts.get(cycle["research_artifact"]) if cycle["research_artifact"] else {"units": [], "plans": None}
    available = {u["unit_id"]: u for u in saved["units"]}
    existing = ctx.store.one("SELECT research_artifact FROM curriculum_assignments WHERE namespace=? AND sequence=?",
                            (work["namespace"], work["sequence"]))
    if not existing or not existing["research_artifact"]:
        # Older buffered blocks saved references only on the block. Recover
        # those exact fetched bytes when a continuation reuses its assignment.
        existing = ctx.store.one(
            "SELECT research_artifact FROM teaching_blocks WHERE namespace=? AND sequence=? "
            "AND assignment_artifact=? AND research_artifact IS NOT NULL ORDER BY rowid DESC LIMIT 1",
            (work["namespace"], work["sequence"], digest(work)))
    if existing and existing["research_artifact"]:
        previous = ctx.artifacts.get(existing["research_artifact"])
        if previous["unit_id"] != work["unit"]["id"] or not previous["sources"]:
            raise ValueError("Saved assignment research has a different unit or no retrieved sources")
        available.setdefault(previous["unit_id"], previous)
    def preserve_assignment_research():
        if work["unit"]["id"] in available:
            ctx.store.execute(
                "UPDATE curriculum_assignments SET research_artifact=? "
                "WHERE namespace=? AND sequence=? AND artifact=? AND research_artifact IS NULL",
                (ctx.artifacts.put(available[work["unit"]["id"]]), work["namespace"], work["sequence"], digest(work)))
    preserve_assignment_research()
    if ctx.config.curriculum_loop.web_training:
        # A new source-use contract needs an explicit licensing decision. Preserve
        # old reference artifacts, but do not relabel them as training permission.
        available = {k: v for k, v in available.items() if 'training_collections' in v}
    requested = list({u["id"]: u for u in units if u["id"] not in available}.values())
    plans = saved["plans"]
    if plans is None:
        plans = CycleResearch.model_validate(ctx.teacher.cycle_research({"units": requested})).model_dump()["units"] if requested else []
        if len(plans) != len(requested) or {u["unit_id"] for u in plans} != {u["id"] for u in requested}:
            raise ValueError("Cycle research must cover every requested unit exactly once")
        key = ctx.artifacts.put({"plans": plans, "units": list(available.values())})
        ctx.store.execute("UPDATE teaching_cycles SET research_artifact=? WHERE id=?", (key, cycle["id"]))
    cached = {s["requested_url"] if "requested_url" in s else s["url"]: s
              for u in available.values() for s in u["sources"]}
    from .curriculum_research import fetch_source
    for entry in plans:
        if entry["unit_id"] in available:
            continue
        sources, failures = [], []
        for request in entry["plan"]["sources"]:
            try:
                if request["url"] not in cached:
                    cached[request["url"]] = fetch_source(request, cancelled=ctx.cancelled)
                sources.append({**cached[request["url"]], "purpose": request["purpose"]})
            except Cancelled:
                raise
            except (httpx.HTTPError, ValueError, OSError) as exc:
                failures.append({"url": request["url"], "error": str(exc)[:500]})
        evidence = {"unit_id": entry["unit_id"], "plan": entry["plan"], "sources": sources,
                    "fetch_failures": failures, "basis": "Real web references for original teaching"}
        if not sources:
            raise ValueError("No research source retrieved for cycle unit; inspect " + ctx.artifacts.put(evidence))
        available[entry["unit_id"]] = evidence
        preserve_assignment_research()
        key = ctx.artifacts.put({"plans": plans, "units": list(available.values())})
        ctx.store.execute("UPDATE teaching_cycles SET research_artifact=? WHERE id=?", (key, cycle["id"]))
    from .web_training import add_collections
    for unit in units:
        available[unit["id"]] = add_collections(ctx, available[unit["id"]])
    result = {"plans": plans, "units": [available[u["id"]] for u in units]}
    ctx.store.execute("UPDATE teaching_cycles SET research_artifact=? WHERE id=?",
                      (ctx.artifacts.put(result), cycle["id"]))
    return result


def plan(ctx):
    cycle = cycle_for(ctx.store, ctx.round["id"])
    if cycle["plan_artifact"]:
        return ctx.artifacts.get(cycle["plan_artifact"])
    units = units_for(ctx, cycle)
    block = ctx.store.one("SELECT assignment_artifact FROM teaching_blocks WHERE round_id=?", (ctx.round["id"],))
    work = ctx.artifacts.get(block["assignment_artifact"])
    authors = required_authors(ctx)
    references = ctx.output('cycle_research')['units']
    for unit in references:
        unit['training_supply'] = []
        for key in unit.get('training_collections', []):
            collection = ctx.artifacts.get(key)
            unit['training_supply'].append({'artifact':key, 'pages':len(collection['pages']),
                'source_chars':collection['collected_chars'], 'license':collection['permission']['license'],
                'basis':'Collected supply before prior-exposure deduplication; not a tokenizer count'})
    brief = {"cycle_id": cycle["id"], "units": units, "block_target_tokens": ctx.config.teaching_cycle.block_target_tokens,
        "maximum_blocks": len(units), "research": references,
        "current_corpus_preview": [{k: v for k, v in s.items() if k != "after"} for s in work["spans"][:2]],
        "full_assignment_artifact": block["assignment_artifact"], "required_authors": authors,
        "authors": catalog(ctx.config.material_authors, ctx.engine.settings.root/".env"),
        "allowances": {k: cycle["contract"][k] for k in ("per_block_calls", "per_block_output_tokens", "cycle_calls", "cycle_output_tokens")},
        "remediation_cap": ctx.config.curriculum_loop.remediation_cap,
        "web_training": ctx.config.curriculum_loop.web_training,
        "previous_production": production_summary(ctx, previous=True)}
    result = CyclePlan.model_validate(ctx.teacher.cycle_plan(brief)).model_dump()
    if len(result["blocks"]) > len(units) or [b["unit_id"] for b in result["blocks"]] != [u["id"] for u in units[:len(result["blocks"])]]:
        raise ValueError("Cycle plan must cover an ordered prefix of the assigned GPC units")
    jobs = []
    for i, block_plan in enumerate(result["blocks"]):
        curriculum = block_plan["curriculum"]
        validate_expansion_plan(ctx.config, curriculum)
        validate_plan(ctx, curriculum, {"unit": units[i], "spans": work["spans"] if i == 0 else []}, authors)
        jobs.extend(curriculum["expansion_jobs"])
    count = len(result["blocks"])
    if (len(jobs) > min(cycle["contract"]["cycle_calls"], count*cycle["contract"]["per_block_calls"])
            or sum(j["max_output_tokens"] for j in jobs) > min(cycle["contract"]["cycle_output_tokens"], count*cycle["contract"]["per_block_output_tokens"])):
        raise ValueError("Cycle plan exceeds its aggregate Author reservation envelope")
    install_plan(ctx, result, {**cycle, "contract": json.dumps(cycle["contract"])})
    return result


def select(ctx):
    cycle = cycle_for(ctx.store, ctx.round["id"])
    planned = ctx.artifacts.get(cycle["plan_artifact"])
    block_plan = planned["blocks"][cycle["position"]]
    block = ctx.store.one("SELECT * FROM teaching_blocks WHERE round_id=?", (ctx.round["id"],))
    if block["assignment_artifact"]:
        work = ctx.artifacts.get(block["assignment_artifact"])
    else:
        previous = ctx.store.one("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='freeze' AND status='complete' ORDER BY id DESC LIMIT 1", (block["predecessor_round_id"],))
        if not previous:
            raise ValueError("Prefetch needs its predecessor's complete frozen coverage prefix")
        prepared = ctx.artifacts.get(previous["artifact"])["curriculum_progress"]
        work = assignment(ctx, prepared_state=prepared["after"], sequence=block["sequence"],
                          predecessor_round_id=block["predecessor_round_id"])
    research_by_unit = {u["unit_id"]: u for u in ctx.artifacts.get(cycle["research_artifact"])["units"]}
    references = research_by_unit[block_plan["unit_id"]]
    progression = {"assignment_artifact": ctx.artifacts.put(work), "research_artifact": ctx.artifacts.put(references),
                   "required_authors": required_authors(ctx)}
    ctx.store.execute("UPDATE teaching_blocks SET assignment_artifact=?,research_artifact=? WHERE round_id=?",
                      (progression["assignment_artifact"], progression["research_artifact"], ctx.round["id"]))
    ctx.progression = progression
    curriculum = copy.deepcopy(block_plan["curriculum"])
    if work["unit"]["id"] != block_plan["unit_id"]:
        raise ValueError("Prepared GPC unit differs from the frozen Teacher plan")
    validate_plan(ctx, curriculum, work, progression["required_authors"])
    lessons = []
    for task in curriculum["lessons"]:
        sources = stages._sources(ctx, task["sources"])
        if task["learning_track"] == "gpc":
            sources = list({s["id"]: s for s in [*sources, *references["sources"]]}.values())
        document = stages._primary(sources, task["concept"])
        document["selection_reason"] = task["reason"]
        lessons.append({**task, "sources": sources, "document": document, "student": None,
                        "student_observation": "not_requested", "teacher": "", "errors": [], "evidence": [], "gate": None})
    from .teacher_tools import replay_lesson
    result = {"curriculum": curriculum, "lessons": lessons, "readings": stages._sources(ctx, curriculum["readings"]),
        "replay": [replay_lesson(ctx.engine.settings.workspace, r["round_id"], r["lesson_id"]) for r in curriculum["replay"]],
        "comparison": None, "scoring": [], "experiment": None, "progression": progression,
        "cycle_id": cycle["id"], "cycle_plan_artifact": cycle["plan_artifact"],
        "cycle_position": cycle["position"], "tokenization_checkpoint": cycle["anchor_checkpoint"]}
    ctx.project("lesson", lessons)
    return result


def draft(ctx):
    return {"lessons": ctx.output("select")["lessons"], "comparison": None, "historical_scoring": [],
            "execution": "Forward targets preauthored by Teacher; no current student attempt requested"}


def revise(ctx):
    cycle = cycle_for(ctx.store, ctx.round["id"])
    targets = ctx.artifacts.get(cycle["plan_artifact"])["blocks"][cycle["position"]]["revisions"]
    lessons = stages._merge(ctx.output("draft")["lessons"], targets, {"text": "teacher"})
    for row in lessons:
        if row.get("material_scope") == "general_chat" and row["use_for_training"]:
            if row["training_tokenization"] != "chat_response" or not row["training_response"].strip():
                raise ValueError("General Teacher chat targets must contain a native chat response")
    ctx.project("lesson", lessons)
    return {"lessons": lessons, "execution": "Exact Teacher-authored cycle targets; no separate revision call"}


def last_block(ctx):
    cycle = cycle_for(ctx.store, ctx.round["id"])
    return cycle["position"] == len(ctx.artifacts.get(cycle["plan_artifact"])["blocks"])-1


def evaluate(ctx):
    if not last_block(ctx):
        return {"items": [], "reference_hash": digest([]), "comparison_pair": None, "review_pending": True}
    cycle = cycle_for(ctx.store, ctx.round["id"])
    plan = ctx.artifacts.get(cycle["plan_artifact"])
    ctx.additional_sources = [s for u in ctx.artifacts.get(cycle["research_artifact"])["units"] for s in u["sources"]]
    items = []
    for value in plan["assessments"]:
        row = Evaluation.model_validate(value).model_dump()
        documents = stages._sources(ctx, row["sources"])
        source = stages._primary(documents, row["concept"])
        items.append({**row, "sources": documents, "document_id": source["id"], "student": "", "grade": None,
                      "source_title": source["title"], "source_url": source["url"], "novel_source": None,
                      "cycle_plan_artifact": cycle["plan_artifact"]})
    ctx.project("evaluation", items)
    from .assessment import evaluation_pair
    return {"items": items, "reference_hash": digest(items),
            "comparison_pair": evaluation_pair(ctx) if any(r["compare_before"] for r in items) else None,
            "cycle_plan_artifact": cycle["plan_artifact"], "assessment_scope": "cycle; optional comparison covers the final block"}


def production_summary(ctx, *, previous=False):
    if previous:
        from .learning_work import latest_work
        work = latest_work(ctx.store, ctx.artifacts, ctx.campaign["id"])
        return [{k: work[k] for k in ("id", "stage_seconds", "teacher_stage_seconds", "author_yield", "prepared_targets_per_pass", "retained_training", "teacher_efficiency") if k in work}] if work else []
    else:
        cycle = cycle_for(ctx.store, ctx.round["id"])
        rows = ctx.store.query("SELECT round_id AS id FROM teaching_blocks WHERE cycle_id=? ORDER BY position", (cycle["id"],))
    result = []
    for row in rows:
        frozen = ctx.store.one("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='freeze' AND status='complete' ORDER BY id DESC LIMIT 1", (row["id"],))
        trained = ctx.store.one("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='train' AND status='complete' ORDER BY id DESC LIMIT 1", (row["id"],))
        if not frozen:
            continue
        from .preparation_summary import read
        dataset = read(ctx.store, ctx.artifacts, frozen["artifact"])
        training = ctx.artifacts.get(trained["artifact"]) if trained else {}
        preparation = dataset.get("progressive_preparation", {})
        from .learning_work import safe_round_work
        work = safe_round_work(ctx.store, ctx.artifacts, row["id"])
        result.append({"round_id": row["id"], "prepared_targets": dataset["ledger"]["total_tokens"],
            "trained_targets": training.get("manifest", {}).get("tokens"),
            "raw_targets": preparation.get("raw_targets_prepared"), "raw_target_shortfall": preparation.get("corpus_window_exhausted"),
            "actual_forward_corpus_share": preparation.get("actual_forward_corpus_share"),
            "targets": {k:v for k,v in (preparation.get("targets") or {}).items() if k != "by_row"}, "frozen_artifact": frozen["artifact"],
            "production": {k:work[k] for k in ("stage_seconds", "teacher_stage_seconds", "author_yield", "teacher_efficiency", "retained_training") if k in work},
            "train_artifact": trained["artifact"] if trained else None})
    return result


def grade(ctx):
    if not last_block(ctx):
        return {"items": [], "score": None, "review_pending": True}
    items = ctx.output("answer")["items"]
    requests, mapping = grade_requests(items, ctx.round["id"])
    cycle = cycle_for(ctx.store, ctx.round["id"])
    value = CycleReview.model_validate(ctx.teacher.cycle_review({"cycle_id": cycle["id"],
        "plan_artifact": cycle["plan_artifact"], "items": requests, "blocks": production_summary(ctx)})).model_dump()
    if value["reflection"].get("experiment_review") is not None:
        raise ValueError("A cycle review cannot claim a legacy per-round experiment")
    items = apply_grades(items, value["grades"], mapping)
    ctx.project("evaluation", items)
    return {"items": items, "score": sum(r["grade"]["score"] for r in items)/len(items) if items else None,
            "reflection": value["reflection"], "cycle_id": cycle["id"]}


def adapt(ctx):
    result = ctx.output("grade")
    if result.get("review_pending"):
        return {"action": "continue", "decision": "continue", "score": None, "review_pending": True,
                "decision_origin": "frozen_cycle_plan", "reason": "Continue the Teacher-preauthorized package; assessment remains pending"}
    gaps = [{"id": r["id"], "concept": r["concept"], "type": r["grade"]["gap_type"],
             "priority": r["grade"]["priority"], "document_id": r["document_id"], "round": ctx.round["number"]}
            for r in result["items"] if r["grade"]["needs_practice"]]
    gaps.sort(key=lambda g: g["priority"], reverse=True)
    ctx.project("gap", gaps)
    reflection = result["reflection"]
    ctx.event("teacher", "Teacher assessed the cycle and recorded its next strategy", reflection)
    return {"gaps": gaps, "score": result["score"], **reflection, "decision": reflection["action"],
            "cycle_id": result["cycle_id"], "review_pending": False}
