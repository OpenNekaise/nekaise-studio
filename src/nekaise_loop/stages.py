"""CoAPT Mid-training stages. Each returns serializable artifacts; no HTTP dependencies."""
from __future__ import annotations

from pathlib import Path

from .artifacts import digest, unchanged_checkpoint, verify_checkpoint, verified_snapshot
from .assessment import evaluation_pair, comparable_generations, grade_requests, apply_grades
from .corpus import read_source
from .prompt_evidence import prompt_training_prefixes
from .learning_work import latest_work, safe_round_work
from .work_accounting import preparation_work
from .scoring import historical_training_pair
from .teaching import Curriculum, Evaluation, Reflection
from .teacher_tools import latest_strategy, replay_lesson
from .author_config import catalog
from .materials import expand, select_materials as material_select, source_accounting, validate_expansion_plan, expansion_receipt


def _merge(rows, results, rename=None):
    result_map = {r["id"]: r for r in results}
    if len(results) != len(rows) or len(result_map) != len(rows) or set(result_map) != {r["id"] for r in rows}:
        raise ValueError("Provider returned missing or unexpected IDs")
    return [{**r, **{(rename or {}).get(k, k): v for k, v in result_map[r["id"]].items()}} for r in rows]


def select(ctx):
    brief = {"campaign_id": ctx.campaign["id"], "round_id": ctx.round["id"], "round_number": ctx.round["number"], "student_checkpoint": ctx.round["model_before"], "latest_strategy": latest_strategy(ctx.engine.settings.workspace, ctx.campaign["id"])}
    brief["previous_learning_work"] = latest_work(ctx.store, ctx.artifacts, ctx.campaign["id"])
    brief["execution_capabilities"] = {"generation": "same_checkpoint_batched_greedy_fp32",
        "max_prompts_per_batch": ctx.config.generation_batch_size,
        "batch_token_positions": ctx.config.generation_batch_tokens,
        "teaching_task_count": "Teacher chooses independently of execution batch size"}
    brief["material_authors"] = {"expansion_policy": ctx.config.expansion_policy, "authors": catalog(ctx.config.material_authors, ctx.engine.settings.root/".env"),
        "max_calls_per_round": ctx.config.material_authors.max_calls_per_round,
        "max_output_tokens_per_round": ctx.config.material_authors.max_output_tokens_per_round,
        "concurrency": ctx.config.material_authors.concurrency}
    if ctx.campaign.get("context_artifact"):
        brief["campaign_context_artifact"] = ctx.campaign["context_artifact"]
    curriculum = Curriculum.model_validate(ctx.teacher.curriculum(brief)).model_dump()
    validate_expansion_plan(ctx.config, curriculum)
    scoring = [historical_training_pair(ctx, rid) for rid in curriculum["scoring_round_ids"]]
    if len({r["id"] for r in curriculum["lessons"]}) != len(curriculum["lessons"]):
        raise ValueError("Teacher curriculum contains duplicate lesson IDs")
    comparison = None
    if curriculum["comparison_round_id"]:
        if not curriculum["lessons"]:
            raise ValueError("Checkpoint comparison needs lesson prompts")
        comparison = _comparison(ctx, curriculum["comparison_round_id"], curriculum["comparison_checkpoint"])
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
    return {"curriculum": curriculum, "lessons": lessons, "readings": readings, "replay": replay, "comparison": comparison, "scoring": scoring}


def _comparison(ctx, round_id, checkpoint_kind="output"):
    # A later assessment/reflection failure does not invalidate completed training.
    row = ctx.store.one("SELECT checkpoint,model_before FROM rounds WHERE id=?", (round_id,))
    stage = ctx.store.one("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='train' AND status='complete' ORDER BY attempt DESC LIMIT 1", (round_id,))
    if not row or not stage or not row["checkpoint"]:
        raise ValueError("Comparison requires a recorded checkpoint and completed training artifact")
    result = ctx.artifacts.get(stage["artifact"])
    interface = result.get("student_format", result["manifest"].get("config", {}).get("student_format", "raw_text"))
    if checkpoint_kind == "input":
        # The immutable output artifact corroborates the DB's original input path.
        # Do not infer an input from mutable config or accept a teacher-supplied path.
        if (result["checkpoint"] != row["checkpoint"] or result.get("trained") is False
                or result["manifest"].get("parent") != row["model_before"]):
            raise ValueError("Input comparison does not match immutable training parent provenance")
        checkpoint = row["model_before"]
        if Path(checkpoint).resolve() == Path(ctx.round["model_before"]).resolve():
            raise ValueError("Comparison checkpoint is identical to the current checkpoint")
        if Path(checkpoint).resolve().is_relative_to(ctx.engine.settings.workspace.resolve() / "runs"):
            parent = ctx.store.one("""SELECT s.artifact FROM stage_runs s JOIN rounds r ON r.id=s.round_id
                WHERE r.checkpoint=? AND s.stage='train' AND s.status='complete'
                AND r.model_before!=r.checkpoint
                ORDER BY s.id DESC LIMIT 1""", (checkpoint,))
            if not parent:
                raise ValueError("Input comparison requires a completed parent training artifact")
            before = ctx.artifacts.get(parent["artifact"])
            if before["checkpoint"] != checkpoint:
                raise ValueError("Input comparison parent checkpoint provenance differs")
            verify_checkpoint(before, require_optimizer=False)
            identity = {"kind": "workspace_checkpoint", "artifact": parent["artifact"],
                        "files": before["manifest"]["files"]}
        else:
            identity = verified_snapshot(checkpoint)
        return {"round_id": round_id, "artifact": stage["artifact"], "checkpoint": checkpoint,
                "checkpoint_kind": "input", "identity": identity, "student_format": interface}
    if checkpoint_kind != "output":
        raise ValueError("Unknown comparison checkpoint kind")
    checkpoint = Path(result["checkpoint"]).resolve()
    if result["checkpoint"] != row["checkpoint"] or not checkpoint.is_relative_to(ctx.engine.settings.workspace.resolve() / "runs"):
        raise ValueError("Comparison checkpoint must match its round and belong to this workspace's runs")
    if checkpoint == Path(ctx.round["model_before"]).resolve():
        raise ValueError("Comparison checkpoint is identical to the current checkpoint")
    verify_checkpoint(result, require_optimizer=False)
    return {"round_id": round_id, "artifact": stage["artifact"], "checkpoint": result["checkpoint"], "student_format": interface}


def _sources(ctx, references):
    return [read_source((ctx.engine.settings.root/ctx.config.corpus_path).resolve(), r["document_id"], r["start"], r["length"]) for r in references]


def _primary(sources, concept):
    return dict(sources[0]) if sources else {"id": "", "title": "Teacher-authored material", "url": "", "license": "teacher-authored", "topic": concept, "source_sha256": "", "text": "", "selection_reason": "Designed by the teacher", "replay": False}


def _teaching_text(row):
    if row.get("training_tokenization") == "chat_response":
        return row["training_response"]
    if "training_text" in row:
        return row["training_text"]
    # Historical artifacts retain their original serialization when replayed.
    return row["teacher"] if row["kind"] == "cpt" else f"Question: {row['prompt']}\nAnswer: {row['teacher']}"


def _tokenization(row):
    # Legacy rows keep their original full-text serialization, including replay.
    mode = row.get("training_tokenization", "full_text")
    if mode == "full_text":
        return {}
    if mode == "chat_response":
        result = {"training_tokenization": mode, "training_prompt": row["student_prompt"],
                  "training_response": row["training_response"]}
        serialization = row.get("prompt_serialization", {})
        expected = row.get("training_template_sha256") or (serialization.get("template_sha256") if serialization.get("format") == "chat_template" else None)
        if expected:
            result["training_template_sha256"] = expected
        if row.get("origin_freeze_artifact"):
            result["origin_freeze_artifact"] = row["origin_freeze_artifact"]
        return result
    if mode != "prompt_prefix":
        raise ValueError(f"Unknown training tokenization: {mode}")
    prefix = row.get("student_prompt")
    if not isinstance(prefix, str) or not prefix or not _teaching_text(row).startswith(prefix):
        raise ValueError("prompt_prefix requires training_text to start with the exact student_prompt")
    return {"training_tokenization": mode, "training_prompt": prefix}


def _student_prompts(rows):
    prompts = []
    for row in rows:
        prompt = {"id": row["id"], "prompt": row["student_prompt"]}
        if row.get("student_format", "campaign") != "campaign":
            prompt["format"] = row["student_format"]
        prompts.append(prompt)
    return prompts


def plan(ctx):
    lessons = ctx.output("select")["lessons"]
    ctx.project("lesson", lessons)
    return {"lessons": lessons}


def draft(ctx):
    lessons = ctx.output("plan")["lessons"]
    pairs = ctx.output("select").get("scoring", [])
    for pair in pairs:
        if historical_training_pair(ctx, pair["round_id"]) != pair:
            raise ValueError("Historical scoring reference changed after selection")
    scoring = ctx.student.score_history(pairs) if pairs else []
    if scoring:
        ctx.event("diagnostic_observation", "Scored frozen training samples on verified before/after checkpoints",
                  {"round_ids": [p["round_id"] for p in pairs]})
    def progress(answer):
        for row in lessons:
            if row["id"] == answer["id"]:
                row["student"] = answer["text"]
                row["generation"] = {k: v for k, v in answer.items() if k not in {"id", "text"}}
        ctx.project("lesson", lessons)
        ctx.event("student", f"Student drafted {answer['id']}", {"lesson_id": answer["id"], "tokens": answer.get("tokens")})
    prompts = _student_prompts(lessons)
    comparison = ctx.output("select").get("comparison")
    if comparison:
        # Revalidate the frozen reference before loading it, including on retry.
        verified = _comparison(ctx, comparison["round_id"], comparison.get("checkpoint_kind", "output"))
        if verified != comparison:
            raise ValueError("Comparison reference changed after selection")
        options = ({"reference_format": comparison["student_format"]}
                   if comparison["student_format"] != ctx.config.student_format else {})
        results = ctx.student.compare(ctx.round["model_before"], comparison["checkpoint"], prompts, progress, **options)
        answers = results["current"]
        references = _merge(prompts, results["reference"])
        by_id = {r["id"]: r for r in references}
        for lesson in lessons:
            override = lesson.get("student_format", "campaign")
            lesson["comparison"] = {**comparison, "basis": "checkpoint_and_interface",
                "current_student_format": override if override != "campaign" else ctx.config.student_format,
                "reference_student_format": override if override != "campaign" else comparison["student_format"],
                "generation": by_id[lesson["id"]]}
        ctx.event("diagnostic_observation", "Compared lesson prompts with a preserved checkpoint", comparison)
    else:
        answers = ctx.student.generate(ctx.round["model_before"], prompts, progress) if lessons else []
    lessons = _merge(lessons, answers, {"text": "student"})
    ctx.project("lesson", lessons)
    return {"lessons": lessons, "comparison": comparison, "historical_scoring": scoring}


def revise(ctx):
    lessons = ctx.output("draft")["lessons"]
    lessons = _merge(lessons, ctx.teacher.revise(lessons), {"text": "teacher"}) if lessons else []
    ctx.project("lesson", lessons)
    return {"lessons": lessons}


def gate(ctx):
    lessons = ctx.output("material_select")["lessons"]
    for row in lessons:
        if row["use_for_training"] and row.get("training_tokenization") != "chat_response" and not row["training_text"].strip():
            raise ValueError("Teacher selected an empty training sequence")
        # Historical stage name retained for artifact inspection. The teacher is the
        # teaching authority; this boundary adds no second semantic review.
        row["gate"] = {"id": row["id"], "passed": row["use_for_training"], "reason": row["reason"], "mode": "trusted_teacher"}
    ctx.project("lesson", [r for r in lessons if not r.get("material_origin")])
    ctx.project("material", [r for r in lessons if r.get("material_origin")])
    return {"lessons": lessons}


def freeze(ctx):
    lessons = ctx.output("gate")["lessons"]
    accepted = [r for r in lessons if r["gate"]["passed"]]
    dataset = []
    for row in accepted:
        text = _teaching_text(row)
        provenance = {"lesson_id": row["id"], "document_id": row["document"]["id"], "source_sha256": row["document"]["source_sha256"]}
        dataset.append({"id": row["id"], "stream": row["kind"], "text": text, **provenance, **_tokenization(row)})
        if row.get("material_origin"):
            dataset[-1]["material_origin"] = row["material_origin"]
        dataset[-1]["sources"] = [{k:d[k] for k in ("id", "source_sha256", "span_start", "span_length")} for d in row["sources"]]
    selected = {**ctx.output("select"), "curriculum": ctx.output("material_select")["curriculum"]}
    for index, document in enumerate(selected["readings"]):
        dataset.append({"id": f"reading-{index}", "stream": "corpus", "text": document["text"], "lesson_id": "", "document_id": document["id"], "source_sha256": document["source_sha256"], "span_start": document["span_start"], "span_length": document["span_length"]})
    for index, row in enumerate(selected["replay"]):
        if row.get("training_tokenization") != "chat_response" and not _teaching_text(row).strip():
            raise ValueError("Selected historical lesson has no teacher text")
        text = _teaching_text(row)
        dataset.append({"id": f"history-{index}", "stream": "replay", "text": text, "lesson_id": row["id"], "document_id": row["document"]["id"], "source_sha256": row["document"]["source_sha256"], "origin_round_id": row["origin_round_id"], "origin_artifact": row["origin_artifact"], **_tokenization(row)})
        if row.get("material_origin"):
            dataset[-1]["material_origin"] = row["material_origin"]
    prepared = ctx.trainer.prepare(ctx.round["model_before"], dataset)
    return {**prepared, "dataset_hash": digest(prepared), "accepted": len(accepted), "rejected": len(lessons)-len(accepted),
            "preparation_work": preparation_work(prepared, {**ctx.config.model_dump(), "train_epochs": selected["curriculum"]["train_epochs"]}),
            "material_sources": source_accounting(prepared),
            "material_expansion": expansion_receipt(ctx, selected["curriculum"], prepared),
            "prompt_training_prefixes": prompt_training_prefixes(lessons, prepared)}


def train(ctx):
    dataset = ctx.output("freeze")
    if ctx.output("material_select")["curriculum"]["train_epochs"] == 0 or not dataset["ledger"]["total_tokens"]:
        ctx.event("training_skipped", "Teacher chose a diagnostic round; weights are unchanged")
        return {**unchanged_checkpoint(ctx.round["model_before"], dataset["dataset_hash"]),
                "student_format": ctx.config.student_format}
    receipt = expansion_receipt(ctx, ctx.output("material_select")["curriculum"], dataset)
    if receipt != dataset.get("material_expansion"):
        raise ValueError("Frozen material expansion receipt does not match this round")
    result = ctx.trainer.train(ctx.round["model_before"], dataset, dataset["dataset_hash"], ctx.metric)
    if result["manifest"]["dataset_hash"] != dataset["dataset_hash"] or result["manifest"]["parent"] != ctx.round["model_before"]:
        raise ValueError("Checkpoint does not match its dataset or parent")
    return {**result, "student_format": ctx.config.student_format}


def evaluate(ctx):
    sources = ctx.output("material_select")
    items = [Evaluation.model_validate(r).model_dump() for r in
             ctx.teacher.evaluate(sources["curriculum"], ctx.output("gate")["lessons"])]
    if len({r["id"] for r in items}) != len(items):
        raise ValueError("Teacher evaluation contains duplicate IDs")
    taught = {d["id"] for r in sources["lessons"] if r["use_for_training"] for d in r["sources"]}
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
    pair = evaluation_pair(ctx) if any(r["compare_before"] for r in items) else None
    return {"items": items, "reference_hash": digest(items), "comparison_pair": pair}


def answer(ctx):
    frozen = ctx.output("evaluate")
    items = frozen["items"]
    pair = frozen.get("comparison_pair")
    def progress(result):
        for row in items:
            if row["id"] == result["id"]:
                row["student"] = result["text"]
        ctx.project("evaluation", items)
        ctx.event("student", f"Student answered online evaluation {result['id']}", {"item_id": result["id"]})
    # Execute exactly the teacher's prompt; never append hidden scoring fields.
    reference = []
    selected = [r for r in items if r.get("compare_before")]
    if pair:
        if evaluation_pair(ctx) != pair:
            raise ValueError("Online comparison identities changed after evaluation was frozen")
        if pair["weights_changed"]:
            answers = ctx.student.compare(pair["after"], pair["before"], _student_prompts(items), progress,
                                          reference_rows=_student_prompts(selected))
            results, reference = answers["current"], answers["reference"]
        else:
            results = ctx.student.generate(pair["after"], _student_prompts(items), progress)
            reference = [r for r in results if r["id"] in {s["id"] for s in selected}]
        _merge(selected, reference)  # Validate reference IDs before recording evidence.
    else:
        results = ctx.student.generate(ctx.output("train")["checkpoint"], _student_prompts(items), progress) if items else []
    items = _merge(items, results, {"text": "student"})
    references = {r["id"]: r for r in reference}
    for row in items:
        if row["id"] in references:
            before = references[row["id"]]
            row["comparison"] = {**pair, "generation": before,
                "observation": "separate_input_checkpoint_generation" if pair["weights_changed"] else "same_generation_reused_weights_unchanged",
                "conditions": comparable_generations(row, before)}
    ctx.project("evaluation", items)
    return {"items": items}


def grade(ctx):
    items = ctx.output("answer")["items"]
    requests, mapping = grade_requests(items, ctx.round["id"])
    raw_grades = ctx.teacher.grade(requests) if requests else []
    items = apply_grades(items, raw_grades, mapping)
    ctx.project("evaluation", items)
    return {"items": items, "score": sum(r["grade"]["score"] for r in items)/len(items) if items else None}


def adapt(ctx):
    result = ctx.output("grade")
    gaps = [{"id": r["id"], "concept": r["concept"], "type": r["grade"]["gap_type"], "priority": r["grade"]["priority"], "document_id": r["document_id"], "round": ctx.round["number"]} for r in result["items"] if r["grade"]["needs_practice"]]
    gaps.sort(key=lambda g: g["priority"], reverse=True)
    ctx.project("gap", gaps)
    dataset = ctx.output("freeze")
    work = safe_round_work(ctx.store, ctx.artifacts, ctx.round["id"])
    reflection = Reflection.model_validate(ctx.teacher.reflect({"curriculum": ctx.output("material_select")["curriculum"], "lessons": ctx.output("gate")["lessons"], "assessment": result, "training": ctx.output("train"), "metrics": ctx.store.metrics(ctx.round["id"]),
        "frozen_dataset_hash": dataset["dataset_hash"], "prompt_training_prefixes": dataset.get("prompt_training_prefixes"),
        "historical_scoring": ctx.output("draft").get("historical_scoring", []), "learning_work": work})).model_dump()
    ctx.event("teacher", "Teacher recorded learning strategy", reflection)
    return {"gaps": gaps, "score": result["score"], **reflection, "decision": reflection["action"], "learning_work": work}
