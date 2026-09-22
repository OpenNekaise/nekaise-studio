import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from nekaise_loop.author_config import AuthorPool, AuthorSpec
from nekaise_loop.material_jobs import initialize_jobs, recover_author_processes
from nekaise_loop.processes import process_start, stop_child
from test_material_authors import configured


def cli_author(**changes):
    return AuthorSpec(id="a", label="Fixture Opus author", transport="claude_code",
                      model="claude-opus-5", **changes)


def fake_cli(tmp_path, script):
    path = tmp_path/"fixture-claude"
    path.write_text(f"#!{sys.executable}\n" + script)
    path.chmod(0o700)
    return str(path)


def setup_cli(setup_loop, tmp_path, *, fault="", timeout=5):
    script = '''import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
payload = json.load(sys.stdin)
assert args[args.index('--model')+1] == 'claude-opus-5'
assert args[args.index('--tools')+1] == ''
assert args[args.index('--max-turns')+1] == '1'
assert '--safe-mode' in args and '--no-session-persistence' in args
assert args[args.index('--setting-sources')+1] == ''
assert args[args.index('--output-format')+1] == 'stream-json'
print(json.dumps({'type':'stream_event','event':{'type':'message_start'}}), flush=True)
assert os.environ['CLAUDE_CODE_MAX_OUTPUT_TOKENS'] == '256'
assert os.environ['CLAUDE_CODE_MAX_RETRIES'] == '0'
assert os.environ['MAX_STRUCTURED_OUTPUT_RETRIES'] == '1'
assert os.environ['MAX_THINKING_TOKENS'] == '0'
assert 'ANTHROPIC_MODEL' not in os.environ
task = payload['task']
row = {'id':'variant','kind':'sft','concept':'Resistance',
       'training_tokenization':'full_text','training_text':'Doubling resistance halves heat flow.',
       'source_keys':list(task['sources'])[:1], 'seed_ids':task['job']['seed_ids'],
       'rationale':'Fixture varied teaching example'}
result = {'type':'result','subtype':'success','is_error':False,'num_turns':2,
          'structured_output':{'rows':[row]},
          'modelUsage':{'claude-opus-5':{'inputTokens':10,'outputTokens':40,
             'cacheReadInputTokens':20,'cacheCreationInputTokens':30}}}
'''
    if fault == "quota":
        script += "result.update(is_error=True, subtype='error_during_execution', result='usage limit; retry-after: 61 seconds')\n"
    elif fault == "wrong_model":
        script += "result['modelUsage']['claude-sonnet-5'] = result['modelUsage'].pop('claude-opus-5')\n"
    elif fault == "turn_limit":
        script += "result.update(is_error=True, subtype='error_max_turns')\n"
    elif fault == "missing_usage":
        script += "result.pop('modelUsage')\n"
    elif fault == "output_overrun":
        script += "result['modelUsage']['claude-opus-5']['outputTokens']=513\n"
    elif fault == "hang":
        script += "time.sleep(30)\n"
    elif fault == "oversize":
        script += "print('x'*2000001,flush=True)\ntime.sleep(30)\n"
    elif fault == "json":
        script += "print('{',flush=True)\nsys.exit(0)\n"
    elif fault == "truncated":
        script += "print(json.dumps({'type':'stream_event','event':{'type':'message_delta','delta':{'stop_reason':'max_tokens'}}}),flush=True)\ntime.sleep(30)\n"
    elif fault == "continuation":
        script += "print(json.dumps({'type':'stream_event','event':{'type':'message_start'}}),flush=True)\ntime.sleep(30)\n"
    script += "print(json.dumps(result))\nsys.exit(1 if result['is_error'] else 0)\n"
    pool = AuthorPool(authors=[cli_author(timeout_seconds=timeout)])
    settings, service, campaign, engine = configured(setup_loop, pool=pool)
    settings.claude = fake_cli(tmp_path, script)
    return settings, service, campaign, engine


def test_cli_authors_train_only_selected_material_and_separate_cached_usage(setup_loop, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "do-not-use-inherited-model")
    settings, service, campaign, engine = setup_cli(setup_loop, tmp_path)
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    details = service.snapshot(campaign["id"])["round"]
    work = details["learning_work"]["material_author_work"]
    assert work["reported_input_tokens"] == 120  # Cache reads and creation counted once.
    assert work["reported_output_tokens"] == 80
    assert work["reserved_output_tokens_all_attempts"] == 1024
    assert work["calls_without_usage"] == 0
    assert work["by_author"]["a"]["reported_input_tokens"] == 120
    assert details["learning_work"]["teacher_efficiency"]["teacher_tokens"] is None
    assert all(m["material_origin"]["model"] == "claude-opus-5" for m in details["materials"])
    for call in service.store.query("SELECT * FROM material_calls"):
        assert call["process_pid"] is None and call["process_start"] is None
        raw = service.artifacts.get(call["artifact"])["response"]
        assert raw["execution"]["max_output_tokens_per_message"] == 256
        assert raw["envelope"]["structured_output"]["rows"]


@pytest.mark.parametrize("fault", ["quota", "wrong_model", "turn_limit", "output_overrun", "json"])
def test_cli_failure_preserves_evidence_usage_and_blocks_training(setup_loop, tmp_path, fault):
    _, service, campaign, engine = setup_cli(setup_loop, tmp_path, fault=fault)
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == ("waiting" if fault == "quota" else "failed")
    assert not service.store.query("SELECT * FROM stage_runs WHERE stage='train'")
    calls = service.store.query("SELECT * FROM material_calls WHERE artifact IS NOT NULL")
    assert calls
    reported = []
    for call in calls:
        assert call["process_pid"] is None
        evidence = service.artifacts.get(call["artifact"])
        assert evidence
        if call["status"] == "cancelled" and not json.loads(call["usage"]):
            assert evidence["response"] is None
            assert service.artifacts.get(evidence["request_artifact"])["request"]
            assert call["reserved_tokens"] > 0
            continue  # The request receipt is evidence; it is not reported usage.
        if fault != "json":
            assert json.loads(call["usage"])["prompt_tokens"] == 60
            reported.append(call["id"])
    if fault != "json":
        assert reported


def test_cli_missing_usage_stays_unknown(setup_loop, tmp_path):
    _, service, campaign, engine = setup_cli(setup_loop, tmp_path, fault="missing_usage")
    engine.run(campaign["id"])
    work = service.snapshot(campaign["id"])["round"]["learning_work"]["material_author_work"]
    assert work["calls_without_usage"] == 2


@pytest.mark.parametrize("fault", ["hang", "oversize", "truncated", "continuation"])
def test_cli_timeout_and_output_limit_join_and_clear_owned_children(setup_loop, tmp_path, fault):
    _, service, campaign, engine = setup_cli(setup_loop, tmp_path, fault=fault, timeout=1)
    started = time.monotonic()
    engine.run(campaign["id"])
    assert time.monotonic() - started < 8
    assert service.store.campaign(campaign["id"])["status"] == "failed"
    assert all(c["process_pid"] is None for c in service.store.query("SELECT * FROM material_calls"))


def test_cli_cancellation_joins_all_concurrent_owned_processes(setup_loop, tmp_path):
    _, service, campaign, engine = setup_cli(setup_loop, tmp_path, fault="hang", timeout=20)
    owned = []
    def cancelled():
        if not service.store.one("SELECT name FROM sqlite_master WHERE name='material_calls'"):
            return False
        calls = service.store.query("SELECT process_pid,process_start FROM material_calls WHERE process_pid IS NOT NULL")
        if len(calls) == 2:
            owned.extend(calls)
            return True
        return False
    engine.run(campaign["id"], controls=cancelled)
    assert owned and service.store.campaign(campaign["id"])["status"] == "stopped"
    assert all(process_start(p["process_pid"]) != p["process_start"] for p in owned)
    assert all(c["process_pid"] is None for c in service.store.query("SELECT * FROM material_calls"))


def test_worker_reconciles_recorded_author_orphans_and_legacy_columns(setup_loop, tmp_path):
    _, service, campaign, engine = setup_cli(setup_loop, tmp_path)
    engine.run(campaign["id"])
    service.store.execute("ALTER TABLE material_calls DROP COLUMN process_pid")
    service.store.execute("ALTER TABLE material_calls DROP COLUMN process_start")
    initialize_jobs(service.store)
    initialize_jobs(service.store)
    call = service.store.one("SELECT * FROM material_calls ORDER BY id LIMIT 1")
    child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True)
    try:
        service.store.execute("UPDATE material_calls SET status='running',process_pid=?,process_start=? WHERE id=?",
                              (child.pid, process_start(child.pid), call["id"]))
        service.store.execute("UPDATE material_jobs SET status='running' WHERE id=?", (call["job_id"],))
        recover_author_processes(service.store)
        child.wait(timeout=5)
        saved = service.store.one("SELECT * FROM material_calls WHERE id=?", (call["id"],))
        assert saved["status"] == "uncertain" and saved["process_pid"] is None
        assert saved["reserved_tokens"] == call["reserved_tokens"]
        assert saved["artifact"] == call["artifact"]
        assert service.store.one("SELECT status FROM material_jobs WHERE id=?", (call["job_id"],))["status"] == "uncertain"
    finally:
        stop_child(child)


@pytest.mark.parametrize("changes", [{"model":"opus"}, {"base_url":"https://example.org"},
    {"api_key_env":"ANTHROPIC_API_KEY"}, {"options":{"max_turns":99}},
    {"options":{"thinking":"false"}}, {"options":{"effort":"max"}}])
def test_cli_configuration_rejects_unbounded_or_ambiguous_overrides(changes):
    data = cli_author().model_dump()
    data.update(changes)
    with pytest.raises(ValueError):
        AuthorSpec.model_validate(data)
