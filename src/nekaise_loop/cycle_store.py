"""Durable cycle contracts, pending blocks, and write-once parent bindings."""
import json
from pathlib import Path

from .artifacts import digest, verify_checkpoint
from .config import CampaignConfig, resolve_student
from .curriculum_progress import initial_state, bind_progress
from .curriculum_inventory import file_hash
from .ownership import source_fingerprint
from .storage import encode, new_id, now


def tokenizer_identity(checkpoint):
    root = Path(checkpoint)
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "chat_template.jinja", "generation_config.json")
    result = {name: file_hash(root/name) for name in names if (root/name).is_file()}
    if "tokenizer.json" not in result or "tokenizer_config.json" not in result:
        raise ValueError("Buffered preparation requires a pinned local tokenizer")
    return result


def cycle_for(store, round_id):
    row = store.one("SELECT c.*,b.position,b.sequence,b.predecessor_round_id,b.parent_binding_artifact "
                    "FROM teaching_cycles c JOIN teaching_blocks b ON b.cycle_id=c.id WHERE b.round_id=?", (round_id,))
    if row:
        row["contract"] = json.loads(row["contract"])
    return row


def open_cycle(engine, campaign):
    store, artifacts = engine.store, engine.artifacts
    config = CampaignConfig.model_validate(campaign["config"])
    cycle = store.one("SELECT * FROM teaching_cycles WHERE campaign_id=? AND status!='complete' ORDER BY number LIMIT 1", (campaign["id"],))
    if cycle:
        contract = json.loads(cycle["contract"])
        if cycle["source_hash"] != source_fingerprint() or contract["config_hash"] != digest(config.model_dump()):
            raise ValueError("Pending cycle implementation/configuration changed; preserve it and create a continuation")
        return cycle
    last = store.one("SELECT COALESCE(MAX(number),0) AS n FROM rounds WHERE campaign_id=?", (campaign["id"],))["n"]
    if config.rounds != -1 and last >= config.rounds:
        return None
    latest = store.one("SELECT s.artifact FROM stage_runs s JOIN rounds r ON r.id=s.round_id "
                       "WHERE r.campaign_id=? AND s.stage='train' AND s.status='complete' ORDER BY r.number DESC,s.id DESC LIMIT 1", (campaign["id"],))
    if latest:
        trained = artifacts.get(latest["artifact"])
        verify_checkpoint(trained)
        anchor = trained["checkpoint"]
    else:
        anchor = resolve_student(config.student_model)
        if campaign.get("context_artifact"):
            restoration = artifacts.get(campaign["context_artifact"]).get("restoration")
            if restoration and anchor == restoration["reference"]["checkpoint"]:
                from .restoration import verify_restoration
                verify_restoration(store, artifacts, restoration)
        manifest = Path(anchor)/"checkpoint.json"
        if manifest.exists():
            verify_checkpoint({"checkpoint": anchor, "manifest": json.loads(manifest.read_text())})
    n = store.one("SELECT COALESCE(MAX(number),0)+1 AS n FROM teaching_cycles WHERE campaign_id=?", (campaign["id"],))["n"]
    limits = config.teaching_cycle
    count = limits.initial_blocks_per_cycle if n == 1 else limits.blocks_per_cycle
    if config.rounds != -1:
        count = min(count, config.rounds-last)
    progress = store.one("SELECT sequence,state FROM curriculum_progress WHERE namespace=?", (config.curriculum_loop.namespace,))
    state = json.loads(progress["state"]) if progress else initial_state()
    if state.get("checkpoint") and state["checkpoint"] != anchor:
        raise ValueError("Cycle anchor differs from the committed curriculum lineage")
    contract = {"config_hash": digest(config.model_dump()), "tokenizer": tokenizer_identity(anchor),
        "initial_sequence": progress["sequence"] if progress else 0, "initial_state": state,
        "requested_blocks": count, "first_round_number": last+1,
        "budget_since": campaign.get("teacher_budget_since") or campaign["created_at"],
        "per_block_calls": config.material_authors.max_calls_per_round,
        "per_block_output_tokens": config.material_authors.max_output_tokens_per_round,
        "cycle_calls": min(limits.max_author_calls_per_cycle, count*config.material_authors.max_calls_per_round),
        "cycle_output_tokens": min(limits.max_author_output_tokens_per_cycle, count*config.material_authors.max_output_tokens_per_round)}
    cycle_id, round_id = new_id("cycle"), new_id("round")
    with store.connect(immediate=True) as db:
        db.execute("INSERT INTO teaching_cycles VALUES (?,?,?,'planning',?,?,?,NULL,NULL,NULL,?,?)",
                   (cycle_id, campaign["id"], n, anchor, source_fingerprint(), encode(contract), now(), now()))
        db.execute("INSERT INTO rounds(id,campaign_id,number,status,model_before,created_at,updated_at) VALUES(?,?,?,'ready',?,?,?)",
                   (round_id, campaign["id"], last+1, anchor, now(), now()))
        db.execute("INSERT INTO teaching_blocks(round_id,cycle_id,position,namespace,sequence) VALUES(?,?,0,?,?)",
                   (round_id, cycle_id, config.curriculum_loop.namespace, contract["initial_sequence"]))
        store.event(campaign["id"], round_id, "teaching_cycle", "Preparing a bounded Teacher cycle",
                    {"cycle_id": cycle_id, "number": n, "contract": contract}, db=db)
    return store.one("SELECT * FROM teaching_cycles WHERE id=?", (cycle_id,))


def install_plan(ctx, plan, cycle):
    """Create future block identities only after a validated, saved Teacher plan."""
    contract = json.loads(cycle["contract"])
    key = ctx.artifacts.put(plan)
    with ctx.store.connect(immediate=True) as db:
        existing = db.execute("SELECT plan_artifact FROM teaching_cycles WHERE id=?", (cycle["id"],)).fetchone()[0]
        if existing and existing != key:
            raise ValueError("Cycle plan is immutable once installed")
        previous = ctx.round["id"]
        for position in range(1, len(plan["blocks"])):
            saved = db.execute("SELECT round_id FROM teaching_blocks WHERE cycle_id=? AND position=?", (cycle["id"], position)).fetchone()
            if saved:
                previous = saved[0]
                continue
            round_id = new_id("round")
            # Empty means explicitly unbound. Never label the tokenizer anchor as
            # the training parent of a future block.
            db.execute("INSERT INTO rounds(id,campaign_id,number,status,model_before,created_at,updated_at) VALUES(?,?,?,'ready','',?,?)",
                       (round_id, ctx.campaign["id"], contract["first_round_number"]+position, now(), now()))
            db.execute("INSERT INTO teaching_blocks(round_id,cycle_id,position,namespace,sequence,predecessor_round_id) VALUES(?,?,?,?,?,?)",
                       (round_id, cycle["id"], position, ctx.config.curriculum_loop.namespace,
                        contract["initial_sequence"]+position, previous))
            previous = round_id
        db.execute("UPDATE teaching_cycles SET plan_artifact=?,status='training',updated_at=? WHERE id=?", (key, now(), cycle["id"]))
    return key


def bind_block(ctx, cycle, frozen):
    """Actual model lineage is bound separately from speculative preparation."""
    contract = cycle["contract"]
    if cycle["source_hash"] != source_fingerprint():
        raise ValueError("Prepared block source fingerprint differs at train binding")
    predecessor = cycle.get("predecessor_round_id")
    if predecessor:
        prior = ctx.store.one("SELECT s.artifact,r.checkpoint FROM stage_runs s JOIN rounds r ON r.id=s.round_id "
                              "WHERE r.id=? AND s.stage='train' AND s.status='complete' ORDER BY s.id DESC LIMIT 1", (predecessor,))
        if not prior:
            raise ValueError("Cannot bind a block before its predecessor saves")
        saved = ctx.artifacts.get(prior["artifact"])
        verify_checkpoint(saved)
        parent = saved["checkpoint"]
    else:
        parent = cycle["anchor_checkpoint"]
    if tokenizer_identity(parent) != contract["tokenizer"]:
        raise ValueError("Actual parent tokenizer differs from the frozen preparation basis")
    ctx.round["model_before"] = parent
    bound = bind_progress(ctx, frozen.get("curriculum_progress"))
    if bound and not bound.get("deferred_checkpoint_binding"):
        current = ctx.store.one("SELECT sequence,state FROM curriculum_progress WHERE namespace=?", (bound["namespace"],))
        if not current or current["sequence"] != bound["sequence"] or json.loads(current["state"]) != bound["before"]:
            raise ValueError("Legacy pending assignment no longer matches committed coverage")
        if bound["before"].get("checkpoint") and bound["before"]["checkpoint"] != parent:
            raise ValueError("Pending assignment cannot bind to a different model lineage")
    value = {"cycle_id": cycle["id"], "round_id": ctx.round["id"], "parent_checkpoint": parent,
             "source_hash": cycle["source_hash"], "tokenizer": contract["tokenizer"],
             "frozen_dataset_hash": frozen["dataset_hash"], "progress": bound}
    key = ctx.artifacts.put(value)
    with ctx.store.connect(immediate=True) as db:
        saved = db.execute("SELECT parent_binding_artifact FROM teaching_blocks WHERE round_id=?", (ctx.round["id"],)).fetchone()[0]
        if saved and saved != key:
            raise ValueError("Training parent binding cannot be rewritten")
        db.execute("UPDATE teaching_blocks SET parent_binding_artifact=? WHERE round_id=?", (key, ctx.round["id"]))
        db.execute("UPDATE rounds SET model_before=? WHERE id=?", (parent, ctx.round["id"]))
    return key


def snapshot(store, artifacts, campaign_id):
    rows = store.query("SELECT * FROM teaching_cycles WHERE campaign_id=? ORDER BY number DESC LIMIT 1", (campaign_id,))
    if not rows:
        return None
    cycle = rows[0]
    contract = json.loads(cycle["contract"])
    blocks = store.query("SELECT r.id,r.number,r.status,r.stage,b.position,b.sequence,b.parent_binding_artifact, "
                         "(SELECT artifact FROM stage_runs WHERE round_id=r.id AND stage='freeze' AND status='complete' ORDER BY id DESC LIMIT 1) AS freeze_artifact "
                         "FROM teaching_blocks b JOIN rounds r ON r.id=b.round_id WHERE b.cycle_id=? ORDER BY b.position", (cycle["id"],))
    for block in blocks:
        prepared = artifacts.get(block["freeze_artifact"]) if block["freeze_artifact"] else {}
        block["prepared_targets"] = prepared.get("ledger", {}).get("total_tokens")
        block["raw_targets"] = prepared.get("progressive_preparation", {}).get("raw_targets_prepared")
        block["preparation_only"] = not bool(block["parent_binding_artifact"])
    return {"id": cycle["id"], "number": cycle["number"], "status": cycle["status"],
            "plan_artifact": cycle["plan_artifact"], "review_artifact": cycle["review_artifact"],
            "blocks": blocks, "budget": {k: contract[k] for k in ("per_block_calls", "per_block_output_tokens", "cycle_calls", "cycle_output_tokens")},
            "basis": "Prepared blocks are not completed training; reviews belong to the cycle"}
