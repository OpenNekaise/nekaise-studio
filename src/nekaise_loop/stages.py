"""Text CoAPT stages. Each returns serializable artifacts; no HTTP dependencies."""
from __future__ import annotations

from pathlib import Path

from .artifacts import digest, unchanged_checkpoint
from .corpus import read_source
from .teaching import Curriculum, Reflection
from .teacher_tools import latest_strategy, replay_lesson


def _merge(rows, results, rename=None):
    result_map = {r["id"]: r for r in results}
    if len(results) != len(rows) or len(result_map) != len(rows) or set(result_map) != {r["id"] for r in rows}:
        raise ValueError("Provider returned missing or unexpected IDs")
    return [{**r, **{(rename or {}).get(k, k): v for k, v in result_map[r["id"]].items()}} for r in rows]


def select(ctx):
    brief = {"campaign_id": ctx.campaign["id"], "round_id": ctx.round["id"], "round_number": ctx.round["number"], "student_checkpoint": ctx.round["model_before"], "latest_strategy": latest_strategy(ctx.engine.settings.workspace, ctx.campaign["id"])}
    curriculum = Curriculum.model_validate(ctx.teacher.curriculum(brief)).model_dump()
    if len({r["id"] for r in curriculum["lessons"]}) != len(curriculum["lessons"]):
        raise ValueError("Teacher curriculum contains duplicate lesson IDs")
    lessons = []
    for task in curriculum["lessons"]:
        sources = _sources(ctx, task["sources"])
        document = _primary(sources, task["concept"])
        document["selection_reason"] = task["reason"]
        lessons.append({**task, "sources": sources, "document": document, "student": "", "teacher": "", "errors": [], "evidence": [], "gate": None})
    readings = _sources(ctx, curriculum["readings"])
    replay = [replay_lesson(ctx.engine.settings.workspace, row["round_id"], row["lesson_id"]) for row in curriculum["replay"]]
    ctx.project("lesson", lessons)
    ctx.event("teacher", "Teacher selected curriculum", {"lessons": len(lessons), "readings": len(readings), "replay": len(replay), "token_mix": curriculum["token_mix"], "notes": curriculum["notes"]})
    return {"curriculum": curriculum, "lessons": lessons, "readings": readings, "replay": replay}


def _sources(ctx, references):
    return [read_source((ctx.engine.settings.root/ctx.config.corpus_path).resolve(), r["document_id"], r["start"], r["length"]) for r in references]


def _primary(sources, concept):
    return dict(sources[0]) if sources else {"id": "", "title": "Teacher-authored material", "url": "", "license": "teacher-authored", "topic": concept, "source_sha256": "", "text": "", "selection_reason": "Designed by the teacher", "replay": False}


def _teaching_text(row):
    if "training_text" in row:
        return row["training_text"]
    # Historical artifacts retain their original serialization when replayed.
    return row["teacher"] if row["kind"] == "cpt" else f"Question: {row['prompt']}\nAnswer: {row['teacher']}"


def plan(ctx):
    lessons = ctx.output("select")["lessons"]
    ctx.project("lesson", lessons)
    return {"lessons": lessons}


def draft(ctx):
    lessons = ctx.output("plan")["lessons"]
    def progress(answer):
        for row in lessons:
            if row["id"] == answer["id"]:
                row["student"] = answer["text"]
                row["generation"] = {k: v for k, v in answer.items() if k not in {"id", "text"}}
        ctx.project("lesson", lessons)
        ctx.event("student", f"Student drafted {answer['id']}", {"lesson_id": answer["id"], "tokens": answer.get("tokens")})
    answers = ctx.student.generate(ctx.round["model_before"], [{"id": r["id"], "prompt": r["student_prompt"]} for r in lessons], progress) if lessons else []
    lessons = _merge(lessons, answers, {"text": "student"})
    ctx.project("lesson", lessons)
    return {"lessons": lessons}


def revise(ctx):
    lessons = ctx.output("draft")["lessons"]
    lessons = _merge(lessons, ctx.teacher.revise(lessons), {"text": "teacher"}) if lessons else []
    ctx.project("lesson", lessons)
    return {"lessons": lessons}


def gate(ctx):
    lessons = ctx.output("revise")["lessons"]
    for row in lessons:
        if row["use_for_training"] and not row["training_text"].strip():
            raise ValueError("Teacher selected an empty training sequence")
        # Historical stage name retained for artifact inspection. The teacher is the
        # teaching authority; this boundary adds no second semantic review.
        row["gate"] = {"id": row["id"], "passed": row["use_for_training"], "reason": row["reason"], "mode": "trusted_teacher"}
    ctx.project("lesson", lessons)
    return {"lessons": lessons}


def freeze(ctx):
    lessons = ctx.output("gate")["lessons"]
    accepted = [r for r in lessons if r["gate"]["passed"]]
    dataset = []
    for row in accepted:
        text = _teaching_text(row)
        provenance = {"lesson_id": row["id"], "document_id": row["document"]["id"], "source_sha256": row["document"]["source_sha256"]}
        dataset.append({"id": row["id"], "stream": row["kind"], "text": text, **provenance})
        dataset[-1]["sources"] = [{k:d[k] for k in ("id", "source_sha256", "span_start", "span_length")} for d in row["sources"]]
    selected = ctx.output("select")
    for index, document in enumerate(selected["readings"]):
        dataset.append({"id": f"reading-{index}", "stream": "corpus", "text": document["text"], "lesson_id": "", "document_id": document["id"], "source_sha256": document["source_sha256"], "span_start": document["span_start"], "span_length": document["span_length"]})
    for index, row in enumerate(selected["replay"]):
        if not _teaching_text(row).strip():
            raise ValueError("Selected historical lesson has no teacher text")
        text = _teaching_text(row)
        dataset.append({"id": f"history-{index}", "stream": "replay", "text": text, "lesson_id": row["id"], "document_id": row["document"]["id"], "source_sha256": row["document"]["source_sha256"], "origin_round_id": row["origin_round_id"], "origin_artifact": row["origin_artifact"]})
    prepared = ctx.trainer.prepare(ctx.round["model_before"], dataset)
    return {**prepared, "dataset_hash": digest(prepared), "accepted": len(accepted), "rejected": len(lessons)-len(accepted)}


def train(ctx):
    dataset = ctx.output("freeze")
    if ctx.output("select")["curriculum"]["train_epochs"] == 0 or not dataset["ledger"]["total_tokens"]:
        ctx.event("training_skipped", "Teacher chose a diagnostic round; weights are unchanged")
        return unchanged_checkpoint(ctx.round["model_before"], dataset["dataset_hash"])
    result = ctx.trainer.train(ctx.round["model_before"], dataset, dataset["dataset_hash"], ctx.metric)
    if result["manifest"]["dataset_hash"] != dataset["dataset_hash"] or result["manifest"]["parent"] != ctx.round["model_before"]:
        raise ValueError("Checkpoint does not match its dataset or parent")
    return result


def evaluate(ctx):
    sources = ctx.output("select")
    items = ctx.teacher.evaluate(sources["curriculum"], ctx.output("gate")["lessons"])
    if len({r["id"] for r in items}) != len(items):
        raise ValueError("Teacher evaluation contains duplicate IDs")
    taught = {d["id"] for r in sources["lessons"] for d in r["sources"]}
    for row in items:
        documents = _sources(ctx, row["sources"])
        primary = _primary(documents, row["concept"])
        row["sources"] = documents
        row["document_id"] = primary["id"]
        row["student"] = ""
        row["grade"] = None
        row["source_title"] = primary["title"]
        row["source_url"] = primary["url"]
        row["novel_source"] = bool(primary["id"] and primary["id"] not in taught)
    ctx.project("evaluation", items)
    return {"items": items, "reference_hash": digest(items)}


def answer(ctx):
    items = ctx.output("evaluate")["items"]
    def progress(result):
        for row in items:
            if row["id"] == result["id"]:
                row["student"] = result["text"]
        ctx.project("evaluation", items)
        ctx.event("student", f"Student answered online evaluation {result['id']}", {"item_id": result["id"]})
    # Execute exactly the teacher's prompt; never append hidden scoring fields.
    results = ctx.student.generate(ctx.output("train")["checkpoint"], [{"id": r["id"], "prompt": r["student_prompt"]} for r in items], progress) if items else []
    items = _merge(items, results, {"text": "student"})
    ctx.project("evaluation", items)
    return {"items": items}


def grade(ctx):
    items = ctx.output("answer")["items"]
    raw_grades = ctx.teacher.grade(items) if items else []
    grades = {r["id"]: r for r in raw_grades}
    if len(raw_grades) != len(items) or set(grades) != {r["id"] for r in items}:
        raise ValueError("Missing evaluation grades")
    for row in items:
        row["grade"] = grades[row["id"]]
    ctx.project("evaluation", items)
    return {"items": items, "score": sum(r["grade"]["score"] for r in items)/len(items) if items else None}


def adapt(ctx):
    result = ctx.output("grade")
    gaps = [{"id": r["id"], "concept": r["concept"], "type": r["grade"]["gap_type"], "priority": r["grade"]["priority"], "document_id": r["document_id"], "round": ctx.round["number"]} for r in result["items"] if r["grade"]["needs_practice"]]
    gaps.sort(key=lambda g: g["priority"], reverse=True)
    ctx.project("gap", gaps)
    reflection = Reflection.model_validate(ctx.teacher.reflect({"curriculum": ctx.output("select")["curriculum"], "lessons": ctx.output("gate")["lessons"], "assessment": result, "training": ctx.output("train"), "metrics": ctx.store.metrics(ctx.round["id"])})).model_dump()
    ctx.event("teacher", "Teacher recorded learning strategy", reflection)
    return {"gaps": gaps, "score": result["score"], **reflection, "decision": reflection["action"]}
