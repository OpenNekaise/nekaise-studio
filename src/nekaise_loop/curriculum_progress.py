"""Two forward cursors, advanced only with the verified train-stage transaction."""
import copy
import json
from pathlib import Path

from .artifacts import digest
from .author_config import catalog
from .corpus import eligible, policy_at
from .curriculum_inventory import build_inventory, next_document, read_inventory_source
from .storage import encode, now


def initial_state():
    return {"corpus_cycle": 0, "inventory": None, "position": "", "document_artifact": None,
            "offset": 0, "documents_completed": 0, "chars_trained": 0,
            "gpc_completed": 0, "completed_rounds": 0, "checkpoint": None}


def cursor_state(state):
    """Coverage identity without a checkpoint that may not have been saved yet."""
    return {k: v for k, v in state.items() if k != "checkpoint"}


def assignment(ctx, *, prepared_state=None, sequence=None, predecessor_round_id=None):
    policy = ctx.config.curriculum_loop
    root = (ctx.engine.settings.root / ctx.config.corpus_path).resolve()
    contract = {"projection_artifact": policy.projection_artifact, "corpus_path": str(root)}
    with ctx.store.connect(immediate=True) as db:
        db.execute("INSERT OR IGNORE INTO curriculum_progress VALUES (?,?,0,?,?)",
                   (policy.namespace, encode(contract), encode(initial_state()), now()))
        state_row = dict(db.execute("SELECT * FROM curriculum_progress WHERE namespace=?", (policy.namespace,)).fetchone())
        if json.loads(state_row["contract"]) != contract:
            raise ValueError("Curriculum namespace has a different pinned curriculum or corpus; use an explicit new namespace")
        parent_checkpoint = json.loads(state_row["state"]).get("checkpoint")
        actual_parent = getattr(ctx, "round", {}).get("model_before", ctx.config.student_model)
        if prepared_state is not None:
            if sequence is None or sequence < state_row["sequence"]:
                raise ValueError("Speculative assignment cannot precede committed coverage")
            if sequence == state_row["sequence"] and cursor_state(prepared_state) != cursor_state(json.loads(state_row["state"])):
                raise ValueError("Prepared frontier differs from committed coverage")
            state_row = {**state_row, "sequence": sequence,
                         "state": encode({**cursor_state(prepared_state), "checkpoint": None})}
        elif parent_checkpoint and Path(parent_checkpoint).resolve() != Path(actual_parent).resolve():
            raise ValueError("Curriculum progress belongs to a different checkpoint; an older-weight branch requires a new namespace")
        previous = db.execute("SELECT artifact FROM curriculum_assignments WHERE namespace=? AND sequence=?",
                              (policy.namespace, state_row["sequence"])).fetchone()
        if previous:
            work = ctx.artifacts.get(previous["artifact"])
            if cursor_state(work["before"]) != cursor_state(json.loads(state_row["state"])):
                raise ValueError("Saved assignment belongs to a different prepared frontier")
            return work
    projection = ctx.artifacts.get(policy.projection_artifact)
    if projection.get("format") != "general_teaching_projection_v1" or not projection.get("units"):
        raise ValueError("Invalid pinned general curriculum projection")
    state = json.loads(state_row["state"])
    cursor = copy.deepcopy(state)
    if not cursor["inventory"]:
        cursor["inventory"] = build_inventory(root, ctx.engine.settings.workspace, ctx.cancelled)
    inventory = cursor["inventory"]
    spans, remaining = [], policy.corpus_window_chars
    while remaining > 0:
        if ctx.cancelled():
            from .processes import Cancelled
            raise Cancelled("Curriculum assignment cancelled")
        if cursor["document_artifact"]:
            document = ctx.artifacts.get(cursor["document_artifact"])
            if not eligible({**document, "status": "ok"}, policy_at(root)):
                raise ValueError("Pinned corpus document is no longer eligible: " + document["id"])
        else:
            entry = next_document(ctx.engine.settings.workspace, cursor["inventory"], cursor["position"])
            if entry is None:
                raise ValueError("Inventory cursor has no remaining document; investigate coverage state")
            document = read_inventory_source(root, cursor["inventory"], entry)
            cursor.update(position=entry["position"], document_artifact=ctx.artifacts.put(document), offset=0)
        start = cursor["offset"]
        # Whole spans; do not split the last span to fill a ratio or byte budget.
        end = min(len(document["text"]), start + policy.span_chars)
        text = document["text"][start:end]
        cursor["offset"] = end
        cursor["chars_trained"] += len(text)
        last = end == len(document["text"])
        cycle_end = False
        if last:
            cursor.update(document_artifact=None, offset=0, documents_completed=cursor["documents_completed"]+1)
            if next_document(ctx.engine.settings.workspace, cursor["inventory"], cursor["position"]) is None:
                cursor.update(inventory=None, position="", corpus_cycle=cursor["corpus_cycle"]+1)
                cycle_end = True
        spans.append({"id": "forward-" + digest([document["id"], document["source_sha256"], start, end])[:24],
                      "stream": "corpus", "learning_track": "corpus", "material_scope": "domain",
                      "document_id": document["id"], "source_sha256": document["source_sha256"],
                      "title": document["title"], "url": document["url"], "license": document["license"],
                      "span_start": start, "span_length": end-start, "text": text,
                      "document_chars": len(document["text"]),
                      "after": copy.deepcopy(cursor)})
        remaining -= len(text)
        if cycle_end:
            break
    unit_index = state["gpc_completed"] % len(projection["units"])
    result = {"namespace": policy.namespace, "sequence": state_row["sequence"],
              "before": state, "unit": projection["units"][unit_index], "unit_index": unit_index,
              "unit_count": len(projection["units"]), "gpc_cycle": state["gpc_completed"] // len(projection["units"]),
              "spans": spans, "inventory": inventory,
              "projection_artifact": policy.projection_artifact}
    if prepared_state is not None:
        result.update(deferred_checkpoint_binding=True, predecessor_round_id=predecessor_round_id)
    key = ctx.artifacts.put(result)
    if prepared_state is not None:
        # Keep future windows in pending cycle blocks. Only verified training
        # claims their immutable namespace/sequence, so invalidated preparation
        # cannot occupy a future coverage position forever.
        return result
    with ctx.store.connect(immediate=True) as db:
        current = db.execute("SELECT sequence FROM curriculum_progress WHERE namespace=?", (policy.namespace,)).fetchone()
        if (prepared_state is None and current[0] != state_row["sequence"]) or current[0] > state_row["sequence"]:
            raise ValueError("Curriculum cursor changed during assignment")
        db.execute("INSERT OR IGNORE INTO curriculum_assignments VALUES (?,?,?,NULL,?)",
                   (policy.namespace, state_row["sequence"], key, now()))
        saved = db.execute("SELECT artifact FROM curriculum_assignments WHERE namespace=? AND sequence=?",
                           (policy.namespace, state_row["sequence"])).fetchone()[0]
    return ctx.artifacts.get(saved)


def required_authors(ctx):
    authors = catalog(ctx.config.material_authors, ctx.engine.settings.root / ".env")
    # Missing credentials are named configuration failures, not silent removal.
    unavailable = [a["id"] for a in authors if not a["credentials_configured"]]
    if unavailable:
        raise ValueError("Registered Authors need credentials or explicit removal: " + ", ".join(unavailable))
    if not authors:
        raise ValueError("Progressive teaching requires a registered Author pool")
    if len(authors) > ctx.config.material_authors.max_calls_per_round:
        raise ValueError("All-Author participation exceeds the call allowance; explicitly adjust the pool or budget")
    return [a["id"] for a in authors]


def validate_plan(ctx, curriculum, work, authors):
    if curriculum["train_epochs"] == 0:
        return
    unit_id = work["unit"]["id"]
    seeds = {r["id"]: r for r in curriculum["lessons"]}
    for row in [*seeds.values(), *curriculum["expansion_jobs"]]:
        track = row.get("learning_track")
        if track not in {"corpus", "gpc", "remediation"}:
            raise ValueError("Progressive material must declare corpus, gpc or remediation learning_track")
        if track == "gpc" and (row.get("curriculum_unit_id") != unit_id or row.get("material_scope") not in {"general_chat", "general_prose"}):
            raise ValueError("GPC material must name the assigned unit and a general material scope")
    forward = {j["author_id"] for j in curriculum["expansion_jobs"] if j["learning_track"] in {"corpus", "gpc"}}
    if forward != set(authors):
        raise ValueError("Every registered Author must contribute forward material: " + ", ".join(sorted(set(authors)-forward)))
    if not any(r["learning_track"] == "gpc" for r in seeds.values()):
        raise ValueError("Primary Teacher must teach the assigned GPC unit")
    if not any(j["learning_track"] == "gpc" for j in curriculum["expansion_jobs"]):
        raise ValueError("At least one Author must teach the assigned GPC unit")
    assigned = {s["document_id"] for s in work["spans"]}
    for row in seeds.values():
        if row["learning_track"] == "corpus" and not any(r["document_id"] in assigned for r in row["sources"]):
            raise ValueError("Forward corpus lessons must cite a document from the assigned window")
    # GPC research is attached to every GPC job by the host, not optional citations.


def progress_receipt(ctx, dataset):
    selection = ctx.output("select").get("progression")
    if not selection or ctx.output("material_select")["curriculum"]["train_epochs"] == 0:
        return None
    work = ctx.artifacts.get(selection["assignment_artifact"])
    prepared = dataset.get("progressive_preparation")
    if not prepared or not prepared["corpus_row_ids"]:
        raise ValueError("Forward corpus targets are missing")
    count = len(prepared["corpus_row_ids"])
    selected = work["spans"][:count]
    if [s["id"] for s in selected] != prepared["corpus_row_ids"]:
        raise ValueError("Corpus preparation must consume a contiguous whole-span prefix")
    rows = {r["id"]: r for r in dataset["rows"]}
    for span in selected:
        if any(rows[span["id"]].get(k) != span[k] for k in ("text", "source_sha256", "document_id", "span_start", "span_length")):
            raise ValueError("Prepared corpus span differs from its assignment")
    from .progressive_preparation import inspect_targets
    measured = inspect_targets(dataset, ctx.config.curriculum_loop.remediation_cap)
    if measured != prepared["targets"]:
        raise ValueError("Progressive target receipt differs from frozen samples")
    if set(selection["required_authors"]) - measured["forward_authors"].keys():
        raise ValueError("A required Author has no prepared forward targets")
    if (measured["gpc_units"] != [work["unit"]["id"]] or not measured["teacher_gpc"]
            or not measured["author_gpc"] or measured["by_row"] != prepared["whole_row_targets"]):
        raise ValueError("Assigned GPC unit needs actual Teacher and Author targets")
    after = copy.deepcopy(selected[-1]["after"])
    after.update(gpc_completed=work["before"]["gpc_completed"]+1,
                 completed_rounds=work["before"]["completed_rounds"]+1)
    if ctx.config.curriculum_loop.web_training:
        from .web_training import coverage
        web_key, web_chars = coverage(ctx, work, dataset["rows"])
        if web_key:
            after.update(web_coverage_artifact=web_key,
                         web_chars_trained=work["before"].get("web_chars_trained", 0)+web_chars)
    return {"namespace": work["namespace"], "sequence": work["sequence"],
            "assignment_artifact": selection["assignment_artifact"], "before": work["before"], "after": after,
            "unit_id": work["unit"]["id"], "targets": measured, "corpus_spans": count,
            **({"deferred_checkpoint_binding": True, "predecessor_round_id": work.get("predecessor_round_id")}
               if work.get("deferred_checkpoint_binding") else {})}


def bind_progress(ctx, receipt):
    """Bind an immutable prepared frontier to the actual, verified parent before GPU work."""
    if not receipt or not receipt.get("deferred_checkpoint_binding"):
        return receipt
    receipt = copy.deepcopy(receipt)
    current = ctx.store.one("SELECT sequence,state FROM curriculum_progress WHERE namespace=?", (receipt["namespace"],))
    state = json.loads(current["state"]) if current else None
    if (not state or current["sequence"] != receipt["sequence"]
            or cursor_state(state) != cursor_state(receipt["before"])):
        raise ValueError("Prepared block does not start at the committed corpus/GPC frontier")
    if state.get("checkpoint") and state["checkpoint"] != ctx.round["model_before"]:
        raise ValueError("Prepared block cannot bind to a different model lineage")
    predecessor = receipt.get("predecessor_round_id")
    if predecessor:
        prior = ctx.store.one("SELECT round_id FROM curriculum_receipts WHERE namespace=? AND sequence=?",
                              (receipt["namespace"], receipt["sequence"]-1))
        if not prior or prior["round_id"] != predecessor:
            raise ValueError("Prepared block's predecessor has not committed verified training")
    receipt["before"] = state
    return receipt


def commit_progress(db, round_id, result, train_artifact):
    receipt = result.get("curriculum_progress")
    if not receipt:
        return
    if result.get("trained") is False or result.get("material_portfolio", {}).get("status") != "verified":
        raise ValueError("Unverified training cannot advance curriculum progress")
    existing = db.execute("SELECT artifact FROM curriculum_receipts WHERE round_id=?", (round_id,)).fetchone()
    if existing:
        if existing[0] != train_artifact:
            raise ValueError("Round already has a different curriculum receipt")
        return
    current = db.execute("SELECT sequence,state FROM curriculum_progress WHERE namespace=?", (receipt["namespace"],)).fetchone()
    if not current or current[0] != receipt["sequence"] or json.loads(current[1]) != receipt["before"]:
        raise ValueError("Stale curriculum assignment cannot advance progress")
    if (receipt["after"].get("checkpoint") != result["checkpoint"]
            or (receipt["before"].get("checkpoint") and receipt["before"]["checkpoint"] != result["manifest"]["parent"])):
        raise ValueError("Curriculum checkpoint lineage differs from the verified training save")
    if receipt.get("deferred_checkpoint_binding"):
        saved = db.execute("SELECT artifact FROM curriculum_assignments WHERE namespace=? AND sequence=?",
                           (receipt["namespace"], receipt["sequence"])).fetchone()
        if saved and saved[0] != receipt["assignment_artifact"]:
            raise ValueError("Another assignment already owns this coverage position")
        db.execute("INSERT OR IGNORE INTO curriculum_assignments VALUES (?,?,?,NULL,?)",
                   (receipt["namespace"], receipt["sequence"], receipt["assignment_artifact"], now()))
    db.execute("INSERT INTO curriculum_receipts VALUES (?,?,?,?,?)",
               (round_id, receipt["namespace"], receipt["sequence"], train_artifact, now()))
    db.execute("UPDATE curriculum_progress SET sequence=sequence+1,state=?,updated_at=? WHERE namespace=?",
               (encode(receipt["after"]), now(), receipt["namespace"]))


def status(store, artifacts, config):
    policy = config.get("curriculum_loop")
    if not policy:
        return None
    projection = artifacts.get(policy["projection_artifact"])
    row = store.one("SELECT * FROM curriculum_progress WHERE namespace=?", (policy["namespace"],))
    state = json.loads(row["state"]) if row else initial_state()
    count = len(projection["units"])
    latest = store.one("SELECT artifact FROM curriculum_receipts WHERE namespace=? ORDER BY sequence DESC LIMIT 1", (policy["namespace"],))
    receipt = artifacts.get(latest["artifact"])["curriculum_progress"] if latest else None
    return {"policy": policy, "state": state, "domains": len(projection["domains"]), "units": count,
            "next_unit": projection["units"][state["gpc_completed"] % count],
            "gpc_cycle": state["gpc_completed"] // count, "last_receipt": receipt,
            "basis": "Verified completed training exposure; not mastery or benchmark performance"}
