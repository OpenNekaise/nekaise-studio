#!/usr/bin/env python3
"""eval_heldout_nll.py — frozen held-out next-token loss, a DIAGNOSTIC evaluator.

Perplexity is never a keep/revert metric (SPEC §1, AGENTS.md). This tool exists for the
CPT scale-ladder campaign, where the question is literally "does held-out next-token
loss keep falling with more corpus tokens", and for the forgetting guard. Every metric
it records carries ``diagnostic=True``.

    # mint the frozen views once (training env: needs transformers + datasets)
    python tools/eval_heldout_nll.py mint

    # audit a training dataset for held-out leakage + pool-document exposure
    python tools/eval_heldout_nll.py audit --dataset-id <id> [--experiment cpt]

    # score a checkpoint (eval env: vLLM offline engine)
    $NEKAISE_EVAL_PYTHON tools/eval_heldout_nll.py score --run-id <run_id> --split dev
    $NEKAISE_EVAL_PYTHON tools/eval_heldout_nll.py score --checkpoint openbmb/MiniCPM5-1B-Base --split dev,generic
    $NEKAISE_EVAL_PYTHON tools/eval_heldout_nll.py score --run-id <run_id> --split frozen --milestone

Views (all under ``<workspace scratch>/probe_eval/heldout-nll-v1/``, minted once,
never overwritten):

- ``dev``     : the corpus_probes transfer-dev document ids (frozen in
                gym/tasks/corpus_probes/provenance.json, excluded from every CPT and
                teacher stream). One contiguous window of <= 2048 tokens per document
                at a hash-derived offset (no first-page bias). Cleaned exactly like the
                raw ladder (``corpusprep.clean_body`` over the manifest ``text_path``).
- ``frozen``  : same construction on the transfer-frozen ids. MILESTONE-ONLY.
- ``generic`` : Salesforce/wikitext ``wikitext-103-raw-v1`` test split at a pinned
                revision, concatenated and cut into independent 2048-token windows.
                The forgetting guard; the 20% anchor stream is corpus text and cannot
                play this role.

Scoring: vLLM ``prompt_logprobs`` over the stored token ids, the first token of every
window excluded (no context), float64 sums, missing/non-finite logprobs are fatal.
NLL = sum(-logprob) / scored_tokens; ppl = exp(NLL) exactly, never a mean of per-doc
perplexities. Also reports per-topic token-weighted NLL, the equal-topic macro, and a
document-level bootstrap 95% interval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ["PATH"] = f"{Path(sys.executable).resolve().parent}:{os.environ.get('PATH', '')}"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lib"))

import corpusprep  # noqa: E402
from runstore import RunStore  # noqa: E402
from workspace import Workspace  # noqa: E402

ACTIVE_WORKSPACE = Workspace.resolve(REPO).apply_environment()
VIEW_NAME = "heldout-nll-v1"
VIEW_DIR = ACTIVE_WORKSPACE.scratch_dir / "probe_eval" / VIEW_NAME
RECORDS_DIR = ACTIVE_WORKSPACE.scratch_dir / "probe_eval" / "heldout_nll_records"
PROBE_PROVENANCE = REPO / "gym" / "tasks" / "corpus_probes" / "provenance.json"
TOKENIZER = "openbmb/MiniCPM5-1B-Base"
WINDOW_TOKENS = 2048
MIN_DOC_TOKENS = 64
GENERIC_REPO, GENERIC_CONFIG, GENERIC_SPLIT = "Salesforce/wikitext", "wikitext-103-raw-v1", "test"
GENERIC_LICENSE = "CC BY-SA 3.0"
SPLITS = ("dev", "frozen", "generic")


def evaluator_version() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state(root: Path) -> dict:
    import subprocess
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--short"],
                               capture_output=True, text=True, timeout=30).stdout
        return {"head": head, "dirty_paths": len(dirty.splitlines())}
    except Exception:  # noqa: BLE001
        return {"head": "", "dirty_paths": None}


def load_tokenizer():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    path = snapshot_download(TOKENIZER, local_files_only=True)
    return AutoTokenizer.from_pretrained(path, local_files_only=True), Path(path).name


def tokenizer_fingerprint(tok) -> str:
    """sha256 over the token->id mapping (vocab + added tokens): stored token ids are
    only meaningful against a tokenizer with this exact mapping. The special-token
    ROLES (pad/bos/eos) are deliberately excluded: Unsloth rewrites pad_token when it
    saves a checkpoint, which does not change any id the views store."""
    vocab = sorted(tok.get_vocab().items())
    added = sorted((t, i) for t, i in (tok.get_added_vocab() or {}).items())
    payload = json.dumps({"vocab": vocab, "added": added}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def special_token_map(tok) -> dict:
    return {k: str(v) for k, v in (tok.special_tokens_map or {}).items()}


def heldout_hash_sets(corpus_dir: Path) -> tuple[set[str], set[str], set[str]]:
    """Held-out doc ids plus their raw-artifact and cleaned-text hashes."""
    probe = json.loads(PROBE_PROVENANCE.read_text())
    ids = set(probe["doc_splits"]["dev"]) | set(probe["doc_splits"]["frozen"])
    raw, clean = set(), set()
    for row in corpusprep.load_manifest(corpus_dir):
        if row.get("id") in ids:
            if row.get("sha256"):
                raw.add(row["sha256"])
            if row.get("corpus_sha256"):
                clean.add(row["corpus_sha256"])
    return ids, raw, clean


# ------------------------------------------------------------------ mint

def window_offset(doc_id: str, n_tokens: int, length: int) -> int:
    span = n_tokens - length
    if span <= 0:
        return 0
    digest = hashlib.sha256(f"{VIEW_NAME}:{doc_id}".encode()).hexdigest()
    return int(digest, 16) % (span + 1)


def mint(args) -> None:
    if VIEW_DIR.exists():
        raise SystemExit(f"{VIEW_DIR} already exists: views are minted once, never rebuilt")
    corpus_dir = ACTIVE_WORKSPACE.resolve_input(args.corpus)
    tok, tok_revision = load_tokenizer()
    probe = json.loads(PROBE_PROVENANCE.read_text())
    manifest = {row["id"]: row for row in corpusprep.load_manifest(corpus_dir)}
    corpus_git = git_state(corpus_dir)
    temp = VIEW_DIR.with_name(f".{VIEW_DIR.name}.{os.getpid()}.tmp")
    temp.mkdir(parents=True, exist_ok=False)
    provenance = {
        "schema_version": 1, "view": VIEW_NAME, "created": time.time(),
        "evaluator_version": evaluator_version(), "window_tokens": WINDOW_TOKENS,
        "min_doc_tokens": MIN_DOC_TOKENS, "tokenizer": TOKENIZER,
        "tokenizer_revision": tok_revision, "tokenizer_fingerprint": tokenizer_fingerprint(tok),
        "tokenizer_special_tokens": special_token_map(tok),
        "cleaner_sha256": sha256_file(REPO / "lib" / "corpusprep.py"),
        "probe_provenance_sha256": sha256_file(PROBE_PROVENANCE),
        "corpus": str(corpus_dir),
        "corpus_git": corpus_git, "probe_fingerprint": probe["fingerprint"],
        "cleaner": "lib/corpusprep.clean_body", "files": {},
    }
    for split in ("dev", "frozen"):
        ids = sorted(probe["doc_splits"][split])
        rows, missing, thin = [], [], []
        manifest_hash = hashlib.sha256()
        for doc_id in ids:
            row = manifest.get(doc_id)
            if row is None or row.get("status") != "ok" or not row.get("text_path"):
                missing.append(doc_id)
                continue
            path = corpus_dir / row["text_path"]
            if not path.is_file():
                missing.append(doc_id)
                continue
            text = corpusprep.clean_body(path.read_text(errors="replace"))
            token_ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(token_ids) < MIN_DOC_TOKENS:
                thin.append(doc_id)
                continue
            length = min(WINDOW_TOKENS, len(token_ids))
            start = window_offset(doc_id, len(token_ids), length)
            window = token_ids[start:start + length]
            manifest_hash.update(json.dumps(
                {k: row.get(k) for k in ("id", "sha256", "corpus_sha256", "text_chars")},
                sort_keys=True).encode())
            rows.append({
                "id": doc_id, "topic": row.get("topic") or "unknown",
                "source": row.get("source") or "unknown",
                "doc_tokens": len(token_ids), "offset": start, "n_tokens": len(window),
                "token_ids": window, "text": tok.decode(window),
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            })
        out = temp / f"{split}.jsonl"
        out.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                               for r in rows))
        provenance["files"][split] = {
            "sha256": sha256_file(out), "documents": len(rows),
            "tokens": sum(r["n_tokens"] for r in rows), "ids_requested": len(ids),
            "missing_ids": missing, "thin_ids": thin,
            "manifest_rows_sha256": manifest_hash.hexdigest(),
        }
        print(f"[mint] {split}: {len(rows)} docs, {provenance['files'][split]['tokens']:,} "
              f"tokens, {len(missing)} missing, {len(thin)} thin")

    from datasets import load_dataset
    ds = load_dataset(GENERIC_REPO, GENERIC_CONFIG, split=GENERIC_SPLIT,
                      revision=args.generic_revision)
    text = "\n".join(t for t in ds["text"])
    token_ids = tok(text, add_special_tokens=False)["input_ids"]
    rows = []
    for i in range(0, len(token_ids) - MIN_DOC_TOKENS + 1, WINDOW_TOKENS):
        window = token_ids[i:i + WINDOW_TOKENS]
        if len(window) < MIN_DOC_TOKENS:
            break
        rows.append({"id": f"wikitext103-test::w{len(rows):04d}", "topic": "generic",
                     "source": "wikitext-103-raw-v1", "offset": i,
                     "n_tokens": len(window), "token_ids": window,
                     "text": tok.decode(window)})
    out = temp / "generic.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                           for r in rows))
    provenance["files"]["generic"] = {
        "sha256": sha256_file(out), "documents": len(rows),
        "tokens": sum(r["n_tokens"] for r in rows), "repo": GENERIC_REPO,
        "config": GENERIC_CONFIG, "split": GENERIC_SPLIT,
        "revision": args.generic_revision, "license": GENERIC_LICENSE,
        "source_rows": len(ds), "source_chars": len(text),
    }
    print(f"[mint] generic: {len(rows)} windows, {provenance['files']['generic']['tokens']:,} tokens")
    (temp / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))
    os.replace(temp, VIEW_DIR)
    print(f"[mint] frozen at {VIEW_DIR}")


# ------------------------------------------------------------------ audit

def audit(args) -> None:
    """Scan one immutable training dataset once: held-out leakage + pool exposure."""
    import datakit
    exp_dir = ACTIVE_WORKSPACE.experiment_dir(args.experiment)
    obj_dir = datakit.data_root(exp_dir) / "objects" / args.dataset_id
    data_file = datakit.data_file(obj_dir)
    provenance = datakit.provenance(obj_dir)
    corpus_dir = ACTIVE_WORKSPACE.resolve_input(args.corpus)
    heldout, heldout_raw, heldout_clean = heldout_hash_sets(corpus_dir)
    manifest = {row["id"]: row for row in corpusprep.load_manifest(corpus_dir)}
    pool_ids: set[str] = set()
    if args.pool:
        for line in Path(ACTIVE_WORKSPACE.resolve_input(args.pool)).read_text().splitlines():
            if line.strip():
                pool_ids.add(json.loads(line)["doc_id"] if line.startswith("{") else line.strip())
    seen: set[str] = set()
    rows = tokens = 0
    pool_tokens = 0
    # Authenticate what is being approved: hash the actual JSONL bytes while scanning
    # and reconcile row/token totals with the dataset's provenance.
    content_hash = hashlib.sha256()
    with data_file.open("rb") as f:
        for raw in f:
            content_hash.update(raw)
            line = raw.decode()
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row.get("meta") or {}
            doc_id = meta.get("doc_id") or ""
            n = int(meta.get("content_tokens") or 0)
            rows += 1
            tokens += n
            if doc_id:
                seen.add(doc_id)
                if doc_id in pool_ids:
                    pool_tokens += n
    leaked = sorted(seen & heldout)
    hash_leaked = sorted(
        doc_id for doc_id in seen if doc_id in manifest and (
            manifest[doc_id].get("sha256") in heldout_raw
            or manifest[doc_id].get("corpus_sha256") in heldout_clean))
    unknown_docs = sorted(doc_id for doc_id in seen if doc_id not in manifest)
    # Documents absent from the live manifest are acceptable only when the dataset's
    # own stream plan lists them (a composite plan keeps pruned base documents as
    # tombstones): explicit historical resolution, never a shrug.
    spec = provenance.get("spec") or {}
    plan_name = spec.get("stream_plan") or (provenance.get("stats") or {}).get("stream_plan")
    plan_rows: dict[str, dict] = {}
    plan_authenticated = False
    if plan_name:
        plan_dir = datakit.data_root(exp_dir) / "plans" / str(plan_name)
        plan_docs = plan_dir / "documents.jsonl"
        if plan_docs.is_file():
            payload = plan_docs.read_bytes()
            # The plan is evidence only if it is the plan the dataset was built from.
            plan_authenticated = (
                hashlib.sha256(payload).hexdigest() == spec.get("ordered_documents_sha256"))
            for line in payload.decode().splitlines():
                if line.strip():
                    row = json.loads(line)
                    plan_rows[row["id"]] = row
    resolved_by_plan = sorted(doc_id for doc_id in unknown_docs
                              if plan_authenticated and doc_id in plan_rows)
    unresolved_docs = sorted(doc_id for doc_id in unknown_docs if doc_id not in resolved_by_plan)
    # Historical identities of plan-resolved documents (their recorded raw hash) are
    # checked against the frozen held-out raw hashes exactly like live documents.
    hash_leaked += sorted(
        doc_id for doc_id in resolved_by_plan
        if plan_rows[doc_id].get("sha256") and plan_rows[doc_id]["sha256"] in heldout_raw)
    actual_sha256 = content_hash.hexdigest()
    recorded_sha256 = provenance.get("content_sha256")
    stats = provenance.get("stats") or {}
    reconciled = (actual_sha256 == recorded_sha256
                  and int(provenance.get("n") or -1) == rows
                  and int(stats.get("content_tokens") or -1) == tokens)
    exposed_pool = sorted(seen & pool_ids)
    result = {
        "dataset_id": args.dataset_id, "experiment": args.experiment, "rows": rows,
        "dataset_content_sha256": recorded_sha256,
        "dataset_content_sha256_actual": actual_sha256,
        "content_sha256_verified": reconciled,
        "stream_plan": plan_name, "stream_plan_authenticated": plan_authenticated,
        "docs_resolved_by_plan": len(resolved_by_plan), "docs_unresolved": unresolved_docs,
        "heldout_hash_leaked_ids": hash_leaked, "docs_missing_from_manifest": unknown_docs,
        "content_tokens": tokens, "documents": len(seen),
        "heldout_leaked_ids": leaked, "pool_size": len(pool_ids),
        "pool_docs_exposed": len(exposed_pool), "pool_docs_exposed_ids": exposed_pool,
        "pool_tokens_exposed": pool_tokens, "audited": time.time(),
        "evaluator_version": evaluator_version(),
    }
    result["passed"] = bool(reconciled and not unresolved_docs and not leaked and not hash_leaked)
    VIEW_DIR.mkdir(parents=True, exist_ok=True)
    audit_path = VIEW_DIR / "dataset_audit.json"
    audits = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
    audits[args.dataset_id] = result
    audit_path.write_text(json.dumps(audits, indent=2, sort_keys=True))
    print(f"AUDIT dataset={args.dataset_id} docs={len(seen)} tokens={tokens:,} "
          f"heldout_leaked={len(leaked)} hash_leaked={len(hash_leaked)} "
          f"unknown_docs={len(unknown_docs)} pool_exposed={len(exposed_pool)}/{len(pool_ids)} "
          f"pool_tokens={pool_tokens:,}")
    if not reconciled:
        raise SystemExit(
            f"dataset {args.dataset_id} bytes/provenance mismatch: content sha256 "
            f"{actual_sha256[:12]} vs recorded {str(recorded_sha256)[:12]}, rows {rows} vs "
            f"{provenance.get('n')}, tokens {tokens:,} vs {stats.get('content_tokens')}")
    if unresolved_docs:
        raise SystemExit(
            f"dataset {args.dataset_id} has {len(unresolved_docs)} documents absent from "
            f"the corpus manifest and from its stream plan (e.g. {unresolved_docs[:3]})")
    if leaked or hash_leaked:
        raise SystemExit(f"HELD-OUT LEAKAGE in {args.dataset_id}: ids {leaked[:5]} "
                         f"hashes {hash_leaked[:5]}")


# ------------------------------------------------------------------ score

def load_view(split: str) -> tuple[list[dict], dict]:
    prov_path = VIEW_DIR / "provenance.json"
    if not prov_path.is_file():
        raise SystemExit(f"no frozen view at {VIEW_DIR}: run `mint` first")
    provenance = json.loads(prov_path.read_text())
    path = VIEW_DIR / f"{split}.jsonl"
    expected = provenance["files"][split]["sha256"]
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(f"{path} drifted: {actual} != {expected}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows, provenance


def score_rows(student, rows: list[dict], batch: int) -> list[dict]:
    """Score windows through the sanctioned vLLM engine wrapper (tools/student.py)."""
    out_rows = []
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        try:
            scored = student.score_token_ids([r["token_ids"] for r in chunk])
        except RuntimeError as exc:
            raise SystemExit(f"scoring failed in batch at {start}: {exc}")
        for row, (total, count) in zip(chunk, scored):
            out_rows.append({"id": row["id"], "topic": row["topic"],
                             "source": row.get("source"), "scored_tokens": count,
                             "nll_sum": total, "nll": total / count})
        done = min(start + batch, len(rows))
        print(f"  [{done}/{len(rows)}]", flush=True)
    return out_rows


def quantile(sorted_values: list[float], q: float) -> float:
    """Linear interpolation between order statistics (numpy default method)."""
    if not sorted_values:
        raise ValueError("empty sample")
    pos = q * (len(sorted_values) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def aggregate(records: list[dict], *, bootstrap: int, seed: int) -> dict:
    if bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    total = sum(r["nll_sum"] for r in records)
    count = sum(r["scored_tokens"] for r in records)
    nll = total / count
    by_topic = {}
    for topic in sorted({r["topic"] for r in records}):
        rows = [r for r in records if r["topic"] == topic]
        by_topic[topic] = {
            "nll": sum(r["nll_sum"] for r in rows) / sum(r["scored_tokens"] for r in rows),
            "documents": len(rows), "tokens": sum(r["scored_tokens"] for r in rows)}
    macro = sum(t["nll"] for t in by_topic.values()) / len(by_topic)
    rnd = random.Random(seed)
    samples = []
    n = len(records)
    for _ in range(bootstrap):
        picked = [records[rnd.randrange(n)] for _ in range(n)]
        samples.append(sum(r["nll_sum"] for r in picked) / sum(r["scored_tokens"] for r in picked))
    samples.sort()
    ci = (quantile(samples, 0.025), quantile(samples, 0.975))
    return {"nll": nll, "ppl": math.exp(nll), "topic_macro_nll": macro,
            "by_topic": by_topic, "documents": n, "scored_tokens": count,
            "bootstrap_ci95": ci, "bootstrap_resamples": bootstrap,
            "bootstrap_method": "document resample, linear-interpolated quantiles"}


def score(args) -> None:
    splits = [s.strip() for s in args.split.split(",") if s.strip()]
    if "all" in splits:
        splits = list(SPLITS)
    for split in splits:
        if split not in SPLITS:
            raise SystemExit(f"unknown split {split!r}; choose from {SPLITS}")
    if "frozen" in splits and not args.milestone:
        raise SystemExit("--split frozen is MILESTONE-ONLY: pass --milestone (and mean it)")
    store = RunStore(REPO)
    run_id = args.run_id
    if run_id:
        run = store.get_run(run_id)
        if run.get("checkpoint_path"):
            checkpoint = str(store.resolve_checkpoint(run_id))
        elif (run.get("metadata") or {}).get("reference_target"):
            checkpoint = str(run["metadata"]["reference_target"]).removeprefix("ckpt:")
        else:
            raise SystemExit(f"run {run_id} has no checkpoint")
    else:
        checkpoint = args.checkpoint
    dataset_id = store.get_run(run_id).get("dataset_id") if run_id else None
    audit_path = VIEW_DIR / "dataset_audit.json"
    audits = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
    if dataset_id and not args.limit:
        audit_row = audits.get(dataset_id)
        if audit_row is None:
            raise SystemExit(f"dataset {dataset_id} has no leakage audit: run "
                             f"`audit --dataset-id {dataset_id}` first")
        if audit_row.get("heldout_leaked_ids") or audit_row.get("heldout_hash_leaked_ids"):
            raise SystemExit(f"dataset {dataset_id} leaks held-out documents; refusing to score")
        if not audit_row.get("content_sha256_verified") or not audit_row.get("passed"):
            raise SystemExit(f"dataset {dataset_id} has no PASSED byte-authenticated audit: "
                             f"re-run `audit --dataset-id {dataset_id}`")
        if audit_row.get("docs_unresolved"):
            raise SystemExit(f"dataset {dataset_id} audit has unresolved documents; refusing to score")
        import datakit
        run_exp = ACTIVE_WORKSPACE.experiment_dir(store.get_run(run_id)["experiment"])
        recorded = datakit.provenance(
            datakit.data_root(run_exp) / "objects" / dataset_id).get("content_sha256")
        if not recorded or audit_row.get("dataset_content_sha256_actual") != recorded:
            raise SystemExit(f"dataset {dataset_id} audit hash does not match the run's "
                             "dataset provenance; refusing to score")
    if "frozen" in splits:
        RECORDS_DIR.mkdir(parents=True, exist_ok=True)
        with (RECORDS_DIR / "frozen_split_audit.log").open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                    f"target={checkpoint} run_id={run_id}\n")

    sys.path.insert(0, str(REPO / "tools"))
    from student import Student
    print(f"[nll] loading {checkpoint}")
    student = Student(checkpoint, max_model_len=args.max_model_len, seed=args.seed,
                      gpu_memory_utilization=args.gpu_memory_utilization)
    tok_name = getattr(student.tokenizer, "name_or_path", "")
    from transformers import AutoTokenizer
    target_fp = tokenizer_fingerprint(AutoTokenizer.from_pretrained(checkpoint))
    version = evaluator_version()
    results = {}
    for split in splits:
        rows, provenance = load_view(split)
        expected_fp = provenance.get("tokenizer_fingerprint") or (
            provenance.get("amendments") or {}).get("tokenizer_fingerprint")
        if expected_fp and target_fp != expected_fp:
            raise SystemExit(f"tokenizer fingerprint mismatch for {checkpoint}: stored token "
                             f"ids are not valid for this tokenizer")
        if args.limit:
            rows = rows[:args.limit]
        print(f"[nll] {checkpoint} on {split}: {len(rows)} windows, "
              f"{sum(r['n_tokens'] for r in rows):,} tokens")
        records = score_rows(student, rows, args.batch_size)
        agg = aggregate(records, bootstrap=args.bootstrap, seed=args.seed)
        split_hash = provenance["files"][split]["sha256"]
        RECORDS_DIR.mkdir(parents=True, exist_ok=True)
        ident = run_id or hashlib.sha256(checkpoint.encode()).hexdigest()[:16]
        marker = f".limit{args.limit}" if args.limit else ""
        records_file = RECORDS_DIR / f"{ident}.{split}.{split_hash[:12]}.{version}{marker}.jsonl"
        payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
        if records_file.exists() and records_file.read_text() != payload:
            raise SystemExit(f"{records_file} exists with different bytes; refusing to overwrite")
        tmp = records_file.with_suffix(".tmp")
        tmp.write_text(payload)
        os.replace(tmp, records_file)
        prefix = "generic_heldout" if split == "generic" else "corpus_heldout"
        suffix = "" if split == "generic" else f"_{split}"
        diagnostic = True
        metadata = {
            "target": checkpoint, "dataset_id": dataset_id, "view": VIEW_NAME,
            "tokenizer": tok_name, "tokenizer_revision": provenance["tokenizer_revision"],
            "tokenizer_fingerprint": target_fp,
            "window_tokens": WINDOW_TOKENS, "documents": agg["documents"],
            "scored_tokens": agg["scored_tokens"], "by_topic": agg["by_topic"],
            "bootstrap_ci95": agg["bootstrap_ci95"], "records_file": str(records_file),
            "records_sha256": sha256_file(records_file), "limit": args.limit,
            "pool_exposure": (audits.get(dataset_id) or {}).get("pool_docs_exposed")
            if dataset_id else None,
        }
        print(f"NLL[{checkpoint}] split={split} nll={agg['nll']:.5f} ppl={agg['ppl']:.3f} "
              f"macro={agg['topic_macro_nll']:.5f} tokens={agg['scored_tokens']:,} "
              f"ci95=({agg['bootstrap_ci95'][0]:.5f},{agg['bootstrap_ci95'][1]:.5f})")
        if run_id and not args.limit:
            store.record_metric(run_id, name=f"{prefix}_nll{suffix}", value=round(agg["nll"], 6),
                                n=agg["scored_tokens"], split=split, split_hash=split_hash,
                                evaluator="heldout_nll", evaluator_version=version,
                                diagnostic=diagnostic, metadata=metadata)
            store.record_metric(run_id, name=f"{prefix}_ppl{suffix}", value=round(agg["ppl"], 6),
                                n=agg["scored_tokens"], split=split, split_hash=split_hash,
                                evaluator="heldout_nll", evaluator_version=version,
                                diagnostic=diagnostic)
            if split != "generic":
                store.record_metric(run_id, name=f"{prefix}_nll_topic_macro{suffix}",
                                    value=round(agg["topic_macro_nll"], 6), n=agg["documents"],
                                    split=split, split_hash=split_hash,
                                    evaluator="heldout_nll", evaluator_version=version,
                                    diagnostic=diagnostic)
        results[split] = {k: agg[k] for k in ("nll", "ppl", "topic_macro_nll", "documents",
                                                "scored_tokens", "bootstrap_ci95")}
    print("EVAL_RESULT " + json.dumps({"run_id": run_id, "target": checkpoint,
                                       "results": results, "evaluator_version": version},
                                      sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    m = sub.add_parser("mint", help="materialize the frozen views once")
    m.add_argument("--corpus", default="../nekaise-corpus")
    m.add_argument("--generic-revision", required=True,
                   help="pinned git revision of Salesforce/wikitext on the HF hub")
    a = sub.add_parser("audit", help="held-out leakage + pool exposure of one dataset")
    a.add_argument("--dataset-id", required=True)
    a.add_argument("--experiment", default="cpt")
    a.add_argument("--pool", default=None, help="doc-id pool file (JSONL doc_id rows or ids)")
    a.add_argument("--corpus", default="../nekaise-corpus")
    s = sub.add_parser("score", help="score a checkpoint on the frozen views")
    who = s.add_mutually_exclusive_group(required=True)
    who.add_argument("--run-id")
    who.add_argument("--checkpoint")
    s.add_argument("--split", default="dev", help="comma list of dev,frozen,generic or all")
    s.add_argument("--milestone", action="store_true")
    s.add_argument("--limit", type=int, default=0, help="smoke only; metrics not recorded")
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--bootstrap", type=int, default=1000)
    s.add_argument("--seed", type=int, default=3407)
    s.add_argument("--max-model-len", type=int, default=4096)
    s.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    args = ap.parse_args()
    {"mint": mint, "audit": audit, "score": score}[args.mode](args)


if __name__ == "__main__":
    main()
