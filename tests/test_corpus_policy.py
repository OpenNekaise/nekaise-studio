"""Default-view admission stays independent of the publisher's collection policy."""
import json
import sqlite3

import pytest

from nekaise_loop.corpus import eligible, policy_at, read_source, search_sources, settled_corpus


def write_policy(root, restrictions, version=2):
    (root / "registry/eligibility.json").write_text(json.dumps({
        "version": version, "restrictions": restrictions,
    }))


def rule(match, collection="allow", default_corpus="deny"):
    return {"status": "restricted", "match": match,
            "effects": {"collection": collection, "default_corpus": default_corpus}}


@pytest.mark.parametrize("version", [1, 2])
def test_default_view_deny_wins_over_collection_permission(corpus, version):
    row = search_sources(corpus, limit=1)["rows"][0]
    restriction = rule({"source": "fixture", "id_prefix": row["id"]})
    if version == 1:
        restriction.pop("effects")
    write_policy(corpus, {"held": restriction}, version)
    assert row["id"] not in {r["id"] for r in search_sources(corpus)["rows"]}
    with pytest.raises(ValueError, match="ineligible"):
        read_source(corpus, row["id"])
    assert read_source(corpus, search_sources(corpus, limit=1)["rows"][0]["id"])["text"]


@pytest.mark.parametrize("reverse", [False, True])
def test_v2_all_matching_rules_and_conjunctive_license_selector(corpus, reverse):
    row = {"id": "doc-1", "source": "fixture", "license": "cc-by", "status": "ok"}
    restrictions = {
        "fetch_only": rule({"source": "fixture"}, "deny", "allow"),
        "held": rule({"id_prefix": "doc-", "license": "cc-by"}),
    }
    if reverse:
        restrictions = dict(reversed(list(restrictions.items())))
    write_policy(corpus, restrictions)
    policy = policy_at(corpus)
    assert not eligible(row, policy)
    assert eligible({**row, "license": "cc0"}, policy)
    assert eligible({**row, "id": "other"}, policy)
    assert not eligible({**row, "status": "failed", "id": "other"}, policy)


@pytest.mark.parametrize("license", ["public-domain", "cc-by", "cc-by-sa", "cc0", "open"])
def test_default_view_open_licenses(license):
    assert eligible({"status": "ok", "license": license}, {})


@pytest.mark.parametrize("license", [None, "", "cc-by-nc", "cc-by-nc-sa", "cc-by-nd",
    "cc-by-nc-nd", "arxiv-nonexclusive", "publisher-oa", "unverified", "proprietary",
    "proprietary-internal", "restricted", "pointer-only", "unknown-future-license"])
def test_other_collected_classes_are_not_default_view_inputs(corpus, license):
    manifest = corpus / "manifest/curated.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]["license"] = license
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_policy(corpus, {})
    assert not eligible(rows[0], policy_at(corpus))
    # Even stale default-view bytes cannot make a held class admissible.
    assert (corpus / "corpus" / f"{rows[0]['id']}.md").exists()
    with pytest.raises(ValueError, match="ineligible"):
        read_source(corpus, rows[0]["id"])
    assert rows[0]["id"] not in {r["id"] for r in search_sources(corpus)["rows"]}


@pytest.mark.parametrize("policy", [
    [], {"version": True, "restrictions": {}}, {"version": 3, "restrictions": {}},
    {"version": 2, "restrictions": []},
    {"version": 2, "restrictions": {"bad": None}},
    {"version": 2, "restrictions": {"": rule({"source": "fixture"})}},
    {"version": 2, "restrictions": {"bad": rule({})}},
    {"version": 2, "restrictions": {"bad": rule({"source": ""})}},
    {"version": 2, "restrictions": {"bad": rule({"id_prefix": 42})}},
    {"version": 2, "restrictions": {"bad": rule({"unknown": "fixture"})}},
    {"version": 2, "restrictions": {"bad": {"status": "restricted", "match": {"source": "fixture"}}}},
    {"version": 2, "restrictions": {"bad": rule({"source": "fixture"}, "allow", "allow")}},
    {"version": 2, "restrictions": {"bad": rule({"source": "fixture"}, "deny", "maybe")}},
    {"version": 2, "restrictions": {"bad": {**rule({"source": "fixture"}), "effects": {"default_corpus": "deny"}}}},
    {"version": 2, "restrictions": {"bad": {**rule({"source": "fixture"}), "effects": {"collection": "deny", "default_corpus": "deny", "other": "allow"}}}},
    {"version": 1, "restrictions": {"bad": rule({"source": "fixture"})}},
    {"version": 1, "restrictions": {"bad": {"status": "restricted", "match": {"license": "cc-by"}}}},
])
def test_malformed_or_unknown_policy_fails_closed(corpus, policy):
    (corpus / "registry/eligibility.json").write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="[Cc]orpus eligibility"):
        policy_at(corpus)


def test_v2_index_is_discovery_only_and_hashes_still_required(corpus):
    row = search_sources(corpus, limit=1)["rows"][0]
    (corpus / "workspace").mkdir()
    with sqlite3.connect(corpus / "workspace/corpus-index.sqlite3") as db:
        db.execute("CREATE TABLE documents(id,title,topic,source,license,status,manifest_shard)")
        db.execute("INSERT INTO documents VALUES(?,?,?,?,?,?,?)", (
            row["id"], row["title"], row["topic"], "stale", "cc-by", "ok", "curated"))
    write_policy(corpus, {"held": rule({"source": "fixture", "license": "cc-by"})})
    assert search_sources(corpus)["rows"][0]["eligible_candidate"]
    with pytest.raises(ValueError, match="ineligible"):
        read_source(corpus, row["id"])
    write_policy(corpus, {"fetch_only": rule({"source": "fixture"}, "deny", "allow")})
    assert read_source(corpus, row["id"])["text"]
    (corpus / "corpus" / f"{row['id']}.md").write_text("Changed bytes")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_source(corpus, row["id"])


def test_policy_change_during_selection_still_rejected(corpus):
    with pytest.raises(RuntimeError, match="eligibility changed"):
        with settled_corpus(corpus):
            write_policy(corpus, {})


def test_complete_loop_with_v2_collection_denied_default_allowed(setup_loop):
    _, service, campaign, engine = setup_loop
    from pathlib import Path
    write_policy(Path(campaign["config"]["corpus_path"]), {
        "fetch_only": rule({"source": "fixture"}, "deny", "allow"),
    })
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    assert service.store.one("SELECT COUNT(*) AS n FROM stage_runs WHERE stage='evaluate' AND status='complete'")["n"] == 2
