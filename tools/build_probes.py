#!/usr/bin/env python3
"""Mint the versioned, document-disjoint corpus probe referee.

This is a maintainer reform command, never an experiment-loop move.  It deterministically
selects source documents from the live sibling ``nekaise-corpus`` checkout, with a fixed
number per topic for three mutually-exclusive roles:

* transfer/dev: unseen documents used by the loop metric;
* transfer/frozen: unseen milestone documents, never used for recipe decisions;
* absorption/dev: train-eligible documents used only as a learning diagnostic.

The output is committed under ``gym/tasks/corpus_probes`` and is frozen after R0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACK_DIR = REPO / "gym" / "tasks" / "corpus_probes"
DEFAULT_CORPUS = REPO.parent / "nekaise-corpus"
sys.path.insert(0, str(REPO / "lib"))
import corpusprep  # noqa: E402

MAX_PER_DOC = 3
MIN_PROMPT_CHARS = 60
MAX_ANSWER_CHARS = 20
MIN_DOMAIN_TERMS = 0
MIN_ENGLISH_RATIO = 0.0
_UNIT = (r"%|°\s?[CF]|kWh?|MWh|GWh|Wh|W\b|Btu|MBtu|kBtu|therms?|tons?\b|Pa\b|kPa|psi[ag]?|bar\b|"
         r"cfm|m3/s|m³/s|L/s|gpm|K\b|ppm|Hz|volts?|V\b|amps?|A\b|mm\b|cm\b|m2|m²|ft2|ft²|"
         r"years?|hours?|minutes?|seconds?|°")
_SPAN = re.compile(r"\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s?(" + _UNIT + r")",
                   re.UNICODE)
_SENT = re.compile(r"(?<=[.!?])\s+")
_NUMS = re.compile(r"\d+(?:\.\d+)?")


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def probes_from_doc(doc: dict, text: str) -> list[dict]:
    out = []
    for para in text.split("\n\n"):
        para = " ".join(para.split())
        if len(para) < 80 or para.startswith("#"):
            continue
        for sent in _SENT.split(para):
            if not (80 <= len(sent) <= 400):
                continue
            matches = list(_SPAN.finditer(sent))
            if not matches:
                continue
            match = matches[-1]
            prompt, answer = sent[:match.start()].rstrip(), match.group(0).strip()
            if len(prompt) < MIN_PROMPT_CHARS or len(answer) > MAX_ANSWER_CHARS:
                continue
            if match.group(1) in prompt:
                continue
            compact = prompt.replace(" ", "")
            if not compact or sum(c.isalpha() for c in prompt) < 0.65 * len(compact):
                continue
            if sum(c.isascii() for c in compact) < 0.85 * len(compact):
                continue
            if len(_NUMS.findall(prompt)) > 4:
                continue
            probe_id = hashlib.md5(
                f"{doc['id']}|{prompt}|{answer}".encode()).hexdigest()[:12]
            out.append({
                "id": f"cp2-{probe_id}", "doc_id": doc["id"],
                "topic": doc["topic"], "prompt": prompt, "answer": answer,
                "value": match.group(1).replace(",", ""),
            })
            if len(out) >= MAX_PER_DOC:
                return out
    return out


def eligible_documents(corpus: Path, seed: int) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    seen_hashes: set[str] = set()
    for row in corpusprep.load_manifest(corpus):
        doc_id = row.get("id")
        rel = Path(row.get("text_path") or f"text/{doc_id}.md")
        path = corpus / rel
        source_hash = row.get("sha256") or ""
        if (not doc_id or row.get("status") != "ok" or not path.is_file()
                or int(row.get("text_chars") or 0) < corpusprep.MIN_DOC_CHARS):
            continue
        quality = (row.get("quality") or {}).get("w20") or {}
        words = int(quality.get("words") or 0)
        english = int(quality.get("en") or 0)
        domain = int(quality.get("domain") or 0)
        if (not words or english / words < MIN_ENGLISH_RATIO
                or domain < MIN_DOMAIN_TERMS):
            continue
        if source_hash and source_hash in seen_hashes:
            continue
        if source_hash:
            seen_hashes.add(source_hash)
        item = {**row, "_text_path": path}
        groups[row.get("topic") or "unknown"].append(item)
    for topic, docs in groups.items():
        def rank(row):
            quality = (row.get("quality") or {}).get("w20") or {}
            words = int(quality.get("words") or 0)
            density = int(quality.get("domain") or 0) / max(1, words)
            tie = hashlib.sha256(
                f"probe-v2:{seed}:{topic}:{row['id']}".encode()).hexdigest()
            return -density, tie
        docs.sort(key=rank)
    return groups


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--docs-per-topic", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    corpus = args.corpus.resolve()
    out_path = PACK_DIR / "probes.jsonl"
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists; a referee reform requires --force")

    groups = eligible_documents(corpus, args.seed)
    probes: list[dict] = []
    seen_prompts: set[str] = set()
    doc_splits = {"dev": [], "frozen": [], "absorption": []}
    roles = (("dev", "transfer", "dev"),
             ("frozen", "transfer", "frozen"),
             ("absorption", "absorption", "dev"))
    by_topic: dict[str, dict[str, int]] = {}

    for topic in sorted(groups):
        by_topic[topic] = {}
        selected = {role: 0 for role, _, _ in roles}
        selected_probes = {role: 0 for role, _, _ in roles}
        assignment = 0
        for doc in groups[topic]:
            if all(n >= args.docs_per_topic for n in selected.values()):
                break
            body = corpusprep.clean_body(doc["_text_path"].read_text(errors="replace"))
            raw_candidates = probes_from_doc(doc, body)
            candidates = []
            for probe in raw_candidates:
                key = probe["prompt"][-80:].lower()
                if key not in seen_prompts:
                    seen_prompts.add(key)
                    candidates.append(probe)
            if not candidates:
                continue
            # Interleave the three roles over the quality-ranked document list so dev,
            # frozen, and absorption have the same quality distribution.
            while selected[roles[assignment % len(roles)][0]] >= args.docs_per_topic:
                assignment += 1
            role, kind, split = roles[assignment % len(roles)]
            assignment += 1
            doc_splits[role].append(doc["id"])
            probes.extend({**probe, "kind": kind, "split": split}
                          for probe in candidates)
            selected[role] += 1
            selected_probes[role] += len(candidates)
        for role, _, _ in roles:
            if selected[role] < args.docs_per_topic:
                raise SystemExit(
                    f"topic {topic!r} produced only {selected[role]}/"
                    f"{args.docs_per_topic} {role} probe documents")
            by_topic[topic][f"{role}_docs"] = selected[role]
            by_topic[topic][f"{role}_probes"] = selected_probes[role]

    PACK_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in probes))
    manifest_paths = sorted((corpus / "manifest").glob("*.jsonl"))
    provenance = {
        "schema_version": 2,
        "version": "corpus-probes-v2",
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "docs_per_topic_role": args.docs_per_topic,
        "eligibility": {
            "min_source_chars": corpusprep.MIN_DOC_CHARS,
            "min_domain_terms_w20": MIN_DOMAIN_TERMS,
            "min_english_ratio_w20": MIN_ENGLISH_RATIO,
        },
        "corpus_commit": subprocess.run(
            ["git", "-C", str(corpus), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True).stdout.strip(),
        "manifest_sha256": _hash_files(manifest_paths),
        "n_probes": len(probes),
        "by_kind": dict(sorted(__import__("collections").Counter(
            row["kind"] for row in probes).items())),
        "by_split": dict(sorted(__import__("collections").Counter(
            row["split"] for row in probes).items())),
        "by_topic": by_topic,
        "doc_splits": doc_splits,
        "fingerprint": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }
    (PACK_DIR / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps({
        "type": "probe_build", "version": provenance["version"],
        "n": len(probes), "topics": len(by_topic),
        "fingerprint": provenance["fingerprint"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
