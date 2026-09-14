"""Schema-validated CLI transport for a trusted teaching model."""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import sys
from ..artifacts import atomic_write, canonical
from ..config import ROOT
from ..teaching import Curriculum, Revisions, Evaluations, Grades, Reflection
from ..teacher_tools import latest_strategy
from ..storage import now, encode
from ..failures import TeacherUnavailable, quota_kind, retry_seconds


def parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


class CliTeacher:
    def __init__(self, config, settings, store, campaign_id, round_id, runner, directory):
        self.config, self.settings, self.store = config, settings, store
        self.campaign_id, self.round_id = campaign_id, round_id
        self.runner, self.directory = runner, directory

    def request(self, purpose, payload, model):
        campaign = self.store.campaign(self.campaign_id)
        since = campaign.get("teacher_budget_since") or campaign["created_at"]
        count = self.store.one("SELECT COUNT(*) AS n FROM teacher_calls WHERE campaign_id=? AND created_at>=?", (self.campaign_id, since))["n"]
        if self.config.max_teacher_calls != -1 and count >= self.config.max_teacher_calls:
            raise TeacherUnavailable("Teacher call allowance used. Resume renews the allowance and retries this stage.", "budget")
        template = (ROOT / "prompts" / f"{purpose}.txt").read_text()
        schema = model.model_json_schema()
        def strict(node):
            if isinstance(node, dict):
                node.pop("default", None)
                if node.get("type") == "object" and "properties" in node:
                    node["required"] = list(node["properties"])
                    node["additionalProperties"] = False
                for value in node.values():
                    strict(value)
            elif isinstance(node, list):
                for value in node:
                    strict(value)
        strict(schema)
        call_id = self.store.execute("INSERT INTO teacher_calls(campaign_id,round_id,purpose,status,usage,created_at) VALUES(?,?,?,'running','{}',?)", (self.campaign_id, self.round_id, purpose, now()))
        call_dir = self.directory / f"teacher-{call_id}"
        call_dir.mkdir(parents=True)
        context = {"workspace": str(self.settings.workspace), "corpus_path": str((self.settings.root/self.config.corpus_path).resolve()), "campaign_id": self.campaign_id, "round_id": self.round_id}
        context_path = call_dir / "context.json"
        atomic_write(context_path, canonical(context))
        command_hint = shlex.join([sys.executable, "-B", "-m", "nekaise_loop.teacher_tools", str(context_path)])
        inputs = {"current": context, "config_hints": self.config.model_dump(), "latest_strategy": latest_strategy(self.settings.workspace, self.campaign_id), "task": payload}
        common = (ROOT/"prompts/teacher.txt").read_text()
        handbook = (ROOT/"docs/COAPT.md").read_text()
        prompt = common + "\n\nTEACHING HANDBOOK:\n" + handbook + "\n\nTASK:\n" + template + "\n\nREAD-ONLY ARCHIVE TOOL:\n" + command_hint + " '<JSON query>'\nStart with {\"op\":\"help\"}. Every archive page is accessible; follow next_offset.\n\nRECORDED DATA:\n" + json.dumps(inputs, ensure_ascii=False)
        atomic_write(call_dir / "input.json", canonical({"purpose": purpose, "inputs": inputs, "prompt": prompt, "model": self.config.teacher_model, "schema": schema}))
        def run(command):
            # Keep WAL sidecars open in the owning worker. Read-only teacher tools
            # can then read SQLite without needing to create shared-memory files.
            with self.store.connect() as history_handle:
                history_handle.execute("SELECT id FROM campaigns LIMIT 1").fetchone()
                return self.runner.run(command, cwd=call_dir, log=call_dir/"provider.log", timeout=self.config.max_stage_seconds, stdin=prompt)
        try:
            if self.config.teacher_provider == "claude":
                command = [self.settings.claude, "-p", "--model", self.config.teacher_model, "--effort", "high", "--tools", "Read,Glob,Grep,Bash", "--allowedTools", "Read,Glob,Grep,Bash", "--permission-prompts", "none", "--strict-mcp-config", "--no-session-persistence", "--output-format", "json", "--json-schema", json.dumps(schema)]
                output = run(command)
                envelope = parse_json(output)
                if envelope.get("is_error"):
                    raise RuntimeError("Teacher rejected request: " + str(envelope.get("result", "unknown error"))[:600])
                response = envelope.get("structured_output") or parse_json(envelope.get("result", "{}"))
                usage = {"cost_usd": envelope.get("total_cost_usd"), "models": envelope.get("modelUsage", {})}
            else:
                schema_path, result_path = call_dir / "schema.json", call_dir / "response.json"
                atomic_write(schema_path, canonical(schema))
                command = [self.settings.codex, "exec", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check", "-C", str(call_dir), "-s", "read-only", "-m", self.config.teacher_model, "-c", 'model_reasoning_effort="high"', "--output-schema", str(schema_path), "--output-last-message", str(result_path), "--json", "--color", "never", "-"]
                run(command)
                response = parse_json(result_path.read_text())
                usage = {"cost_usd": None, "note": "CLI does not provide a cost estimate in this adapter"}
            result = model.model_validate(response).model_dump()
            if "rows" in result and len({r["id"] for r in result["rows"]}) != len(result["rows"]):
                raise ValueError("Teacher returned duplicate IDs")
            atomic_write(call_dir / "output.json", canonical(result))
            self.store.execute("UPDATE teacher_calls SET status='complete',usage=? WHERE id=?", (encode(usage), call_id))
            self.store.event(self.campaign_id, self.round_id, "teacher", f"Teacher completed {purpose}", {"call_id": call_id, "usage": usage})
            return result
        except BaseException as exc:
            kind = quota_kind(exc)
            self.store.execute("UPDATE teacher_calls SET status=? WHERE id=?", ("waiting" if kind else "failed", call_id))
            if kind:
                raise TeacherUnavailable(str(exc), kind, retry_seconds(exc)) from exc
            raise

    def curriculum(self, brief):
        return self.request("curriculum", brief, Curriculum)

    def revise(self, lessons):
        return self.request("revise", {"lessons": lessons}, Revisions)["rows"]

    def evaluate(self, curriculum, lessons):
        return self.request("evaluate", {"curriculum": curriculum, "lessons": lessons}, Evaluations)["rows"]

    def grade(self, items):
        return self.request("grade", {"items": items}, Grades)["rows"]

    def reflect(self, observations):
        return self.request("reflect", observations, Reflection)
