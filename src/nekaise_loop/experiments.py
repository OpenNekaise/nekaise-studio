"""Immutable teaching plans, strategy versions and read-only evidence views.

The select artifact owns intent; adapt owns the Teacher's later interpretation.
The records index is published in the same transaction as select completion.
No benchmark data, model execution or automatic teaching decisions belong here.
"""
import json

from .artifacts import digest
from .experiment_types import ExperimentPlan, StrategyDraft


def strategy(store, artifacts, version):
    recorded = store.one("""SELECT id FROM records WHERE kind='experiment'
        AND json_extract(data,'$.strategy_version')=? LIMIT 1""", (version,))
    if not recorded:
        raise ValueError("Unknown recorded teaching strategy version")
    value = artifacts.get(version)
    if value.get("kind") != "teaching_strategy" or value.get("schema_version") != 1:
        raise ValueError("Artifact is not a teaching strategy version")
    StrategyDraft.model_validate(value["definition"])
    return value


def prepare(ctx, proposed):
    if proposed is None:
        return None
    plan = ExperimentPlan.model_validate(proposed).model_dump()
    if plan["strategy_version"]:
        definition = strategy(ctx.store, ctx.artifacts, plan["strategy_version"])
        version = plan["strategy_version"]
    else:
        if plan["strategy"]["parent_version"]:
            strategy(ctx.store, ctx.artifacts, plan["strategy"]["parent_version"])
        definition = {"kind": "teaching_strategy", "schema_version": 1,
                      "definition": plan["strategy"]}
        version = ctx.artifacts.put(definition)
    stage = ctx.store.one("SELECT input_hash,started_at FROM stage_runs WHERE id=?", (ctx.stage_id,))
    for round_id in plan["related_round_ids"]:
        previous = ctx.store.one("""SELECT s.id FROM stage_runs s WHERE s.round_id=?
            AND s.stage='select' AND s.status='complete' AND s.finished_at<=?""",
            (round_id, stage["started_at"]))
        if round_id == ctx.round["id"] or not previous:
            raise ValueError("Experiment references require an earlier recorded teaching round")
    parent = ctx.store.one("""SELECT s.artifact FROM stage_runs s JOIN rounds r ON r.id=s.round_id
        WHERE r.checkpoint=? AND s.stage='train' AND s.status='complete'
        AND r.checkpoint!=r.model_before ORDER BY s.id DESC LIMIT 1""", (ctx.round["model_before"],))
    return {"id": ctx.round["id"], "schema_version": 1,
            "strategy_version": version, "strategy": definition["definition"],
            "plan": {k: v for k, v in plan.items() if k not in {"strategy", "strategy_version"}},
            "starting_point": {"checkpoint": ctx.round["model_before"],
                "parent_train_artifact": parent["artifact"] if parent else None,
                "campaign_config_hash": digest(ctx.config.model_dump()),
                "select_input_hash": stage["input_hash"]},
            "intent_basis": "recorded_at_select_before_student_observations_and_training",
            "evaluation_basis": "observation_plan_only_questions_are_frozen_later_before_answers"}


def index_selection(store, round_id, stage_id, artifact, card, db):
    if card is None:
        return
    entry = {"id": round_id,
        "select_stage_id": stage_id, "select_artifact": artifact,
        "strategy_version": card["strategy_version"],
        "strategy_name": card["strategy"]["name"],
        "title": card["plan"]["title"], "hypothesis": card["plan"]["hypothesis"]}
    prior = db.execute("SELECT data FROM records WHERE round_id=? AND kind='experiment'", (round_id,)).fetchone()
    if prior and json.loads(prior["data"]) != entry:
        raise ValueError("Cannot replace a committed experiment plan")
    store.put_records(round_id, "experiment", [entry], db=db)


def rebuild_index(store, artifacts):
    """Rebuild missing projections from completed select artifacts only."""
    count = 0
    for stage in store.query("SELECT id,round_id,artifact FROM stage_runs WHERE stage='select' AND status='complete' ORDER BY id"):
        card = artifacts.get(stage["artifact"]).get("experiment")
        if card is not None:
            with store.connect(immediate=True) as db:
                index_selection(store, stage["round_id"], stage["id"], stage["artifact"], card, db)
            count += 1
    return {"indexed_experiments": count}


def validate_review(card, review, graded_items):
    if review is None:
        return
    if card is None:
        raise ValueError("Experiment review requires this round's pre-training experiment plan")
    ids = review["evaluation_ids"]
    if len(set(ids)) != len(ids) or not set(ids) <= {r["id"] for r in graded_items}:
        raise ValueError("Experiment evidence must reference unique graded question IDs from this round")


def plan_artifact(ctx):
    selected = ctx.output("select")
    return digest(selected) if selected.get("experiment") is not None else None


_INDEX = """FROM records x JOIN rounds r ON r.id=x.round_id
    JOIN campaigns c ON c.id=r.campaign_id
    JOIN stage_runs s ON s.id=json_extract(x.data,'$.select_stage_id')
        AND s.round_id=r.id AND s.stage='select' AND s.status='complete'
        AND s.artifact=json_extract(x.data,'$.select_artifact')
    WHERE x.kind='experiment'"""


def catalog(store, *, campaign_id=None, strategy_version=None, before=None, limit=20):
    if not 1 <= limit <= 100 or (before is not None and before < 1):
        raise ValueError("Use limit 1..100 and a positive before cursor")
    filters, args = "", []
    for column, value in (("r.campaign_id", campaign_id),
                          ("json_extract(x.data,'$.strategy_version')", strategy_version)):
        if value is not None:
            filters += f" AND {column}=?"
            args.append(value)
    total = store.one("SELECT COUNT(*) AS n " + _INDEX + filters, args)["n"]
    if before is not None:
        filters += " AND s.id<?"
        args.append(before)
    rows = store.query("""SELECT x.data,s.id AS cursor,s.finished_at AS planned_at,
        r.id AS round_id,r.campaign_id,r.number AS round_number,r.status AS execution_status,
        c.name AS campaign_name """ + _INDEX + filters + " ORDER BY s.id DESC LIMIT ?", (*args, limit+1))
    items = [{**json.loads(row.pop("data")), **row} for row in rows[:limit]]
    return {"items": items, "total": total,
            "next_before": items[-1]["cursor"] if len(rows) > limit else None}


def _stages(store, round_id):
    rows = store.query("""SELECT id,stage,attempt,status,artifact,finished_at FROM stage_runs
        WHERE round_id=? AND status='complete' ORDER BY id""", (round_id,))
    return {r["stage"]: r for r in rows}


def detail(store, artifacts, round_id, *, include_work=True):
    row = store.one("SELECT id,campaign_id,number,status,stage,error FROM rounds WHERE id=?", (round_id,))
    if row is None:
        raise KeyError(round_id)
    stages = _stages(store, round_id)
    if "select" not in stages:
        return None
    selected = artifacts.get(stages["select"]["artifact"])
    card = selected.get("experiment")
    if card is None:
        return None
    if digest({"kind": "teaching_strategy", "schema_version": 1,
               "definition": card["strategy"]}) != card["strategy_version"]:
        raise ValueError("Experiment strategy differs from its recorded version")
    review = artifacts.get(stages["adapt"]["artifact"]).get("experiment_review") if "adapt" in stages else None
    grade = artifacts.get(stages["grade"]["artifact"]) if "grade" in stages else None
    evaluation = artifacts.get(stages["evaluate"]["artifact"]) if "evaluate" in stages else None
    paired = []
    for item in grade["items"] if grade else []:
        pair = item.get("comparison")
        if not pair:
            continue
        comparable = pair.get("weights_changed") and pair.get("conditions", {}).get("comparable")
        paired.append({"id": item["id"], "question": item.get("question", ""),
            "before_score": pair.get("grade", {}).get("score"),
            "after_score": item.get("grade", {}).get("score"),
            "score_delta": pair.get("score_delta") if comparable else None,
            "comparable": bool(comparable), "weights_changed": pair.get("weights_changed"),
            "differences": pair.get("conditions", {}).get("differences", [])})
    result = {**card, "round_id": round_id, "campaign_id": row["campaign_id"],
        "round_number": row["number"], "planned_at": stages["select"]["finished_at"],
        "execution": {k: row[k] for k in ("status", "stage", "error")},
        "review": review, "reviewed_at": stages.get("adapt", {}).get("finished_at"),
        "related_rounds": [store.one("SELECT id AS round_id,campaign_id,number AS round_number FROM rounds WHERE id=?", (rid,)) for rid in card["plan"]["related_round_ids"]],
        "artifacts": {name: {"hash": stage["artifact"], "stage_id": stage["id"]}
                      for name, stage in stages.items()},
        "assessment": {"status": "complete" if grade else "not_recorded",
            "question_count": len(grade["items"]) if grade else None, "paired": paired,
            "requested_pairs": sum(bool(r.get("compare_before")) for r in evaluation["items"]) if evaluation else None},
        "limitations": "Teacher hypotheses and conclusions are judgments. This records one sequential teaching round, not a matched-start A/B experiment. Questions are frozen before answers, after training; different rounds may use different questions. No cross-round learning gain is computed."}
    if include_work:
        from .learning_work import safe_round_work
        result["learning_work"] = safe_round_work(store, artifacts, round_id)
    return result


def safe_detail(store, artifacts, round_id):
    try:
        return detail(store, artifacts, round_id, include_work=False)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"round_id": round_id, "error": f"Experiment evidence unavailable: {exc}"[:600]}


def latest_context(store, artifacts, campaign_id):
    """Follow ancestry, stopping each ancestor at its child's creation boundary."""
    visited, boundary = set(), None
    while campaign_id and campaign_id not in visited:
        visited.add(campaign_id)
        args = [campaign_id]
        cutoff = ""
        if boundary:
            cutoff = " AND s.finished_at<=?"
            args.append(boundary)
        row = store.one("SELECT r.id AS round_id " + _INDEX + " AND r.campaign_id=?" + cutoff + " ORDER BY s.id DESC LIMIT 1", args)
        if row:
            value = detail(store, artifacts, row["round_id"], include_work=False)
            if value:
                if boundary and value["reviewed_at"] and value["reviewed_at"] > boundary:
                    value["review"] = None
                # This is a quick reference. The complete archive remains available.
                return {**{k: value[k] for k in ("round_id", "campaign_id", "strategy_version", "strategy", "plan", "review", "planned_at")},
                        "execution_as_of_now": value["execution"]}
        campaign = store.one("SELECT parent_campaign_id,created_at FROM campaigns WHERE id=?", (campaign_id,))
        if not campaign:
            break
        boundary = min(boundary, campaign["created_at"]) if boundary else campaign["created_at"]
        campaign_id = campaign["parent_campaign_id"]
    return None
