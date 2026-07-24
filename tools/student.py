#!/usr/bin/env python3
"""student.py — batch student-model inference for the CoAPT loop.

Three jobs, JSONL in / JSONL out, two engines (offline vLLM, or a persistent server):

    $NEKAISE_EVAL_PYTHON tools/student.py score  --run-id <id> --in docs.jsonl   --out nll.jsonl
    $NEKAISE_EVAL_PYTHON tools/student.py draft  --run-id <id> --in prompts.jsonl --out drafts.jsonl
    $NEKAISE_EVAL_PYTHON tools/student.py answer --run-id <id> --in questions.jsonl --out answers.jsonl

- `score`  reads rows with a `text` field and appends `nll` (mean per-token negative
  log-likelihood) + `scored_tokens`. The diagnosis signal: how foreign a document still
  is to the current student.
- `draft`  reads rows with a `prompt` field and appends `draft` — the student's raw
  continuation. CoAPT-CPT's r_S: the draft exists to EXPOSE the student's state, so it is
  generated greedily and never cleaned up here.
- `answer` reads rows with a `question` field, formats the frozen closed-book template
  (`Question: ...\nAnswer:`), and appends `answer`. CoAPT-SFT's a_S.

Persistent server workflow — pay the engine load ONCE per checkpoint instead of per call:

    $NEKAISE_EVAL_PYTHON tools/student.py serve --run-id <id>       # prints the target
    $NEKAISE_EVAL_PYTHON tools/student.py score --target server:student@http://127.0.0.1:8971 ...
    $NEKAISE_EVAL_PYTHON tools/eval_probes.py --run-id <id> --target server:student@http://127.0.0.1:8971 ...
    $NEKAISE_EVAL_PYTHON tools/student.py serve-stop

`serve` starts a managed `vllm serve` on this machine (state under the workspace scratch
dir; serving a new checkpoint replaces the previous server). Server-mode `score` uses the
server's /tokenize endpoint and scores exactly the same first `--max-len` tokens as the
offline path, so NLLs are comparable across engines. The default
`--gpu-memory-utilization 0.2` leaves room for 1B full-parameter training on a 48GB GPU;
stop the server before training bigger models.

All other input fields pass through untouched, so provenance survives the round trip.
The model resolves like eval_probes: --run-id via the RunStore, or --checkpoint as a
path/HF id. Runs in the isolated eval environment (vLLM); no transformers.generate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ["PATH"] = f"{Path(sys.executable).resolve().parent}:{os.environ.get('PATH', '')}"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lib"))

# The frozen closed-book format. The CoAPT mixer serializes training QA with the same
# words, so what the student practices is exactly what the diagnosis asks.
ANSWER_TEMPLATE = "Question: {question}\nAnswer:"
ANSWER_STOP = ["\nQuestion:", "\n\n"]
MIN_SCORE_TOKENS = 8
DEFAULT_PORT = 8971
SERVED_NAME = "student"
SERVER_STATE = "student_server.json"
SERVER_LOG = "student_server.log"


def read_rows(path: Path, required: str) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if required not in row or not str(row[required]).strip():
            raise SystemExit(f"{path}:{i}: row missing required field {required!r}")
        if "id" not in row:
            raise SystemExit(f"{path}:{i}: row missing required field 'id'")
        rows.append(row)
    if not rows:
        raise SystemExit(f"no usable rows in {path}")
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def mean_nll(token_ids, prompt_logprobs) -> tuple[float | None, int]:
    """Mean NLL over prompt tokens from vLLM prompt_logprobs (first entry is None)."""
    total, count = 0.0, 0
    for tid, entry in zip(token_ids, prompt_logprobs or []):
        if entry is None:
            continue
        lp = entry.get(tid)
        if lp is None:
            continue
        value = lp.logprob if hasattr(lp, "logprob") else float(lp)
        if not math.isfinite(value):
            continue
        total += -value
        count += 1
    return (round(total / count, 4) if count else None), count


def mean_nll_from_list(logprobs) -> tuple[float | None, int]:
    """Mean NLL over an OpenAI-style token_logprobs list (first entry is None)."""
    values = [lp for lp in (logprobs or []) if lp is not None and math.isfinite(lp)]
    if not values:
        return None, 0
    return round(-sum(values) / len(values), 4), len(values)


def answer_prompt(question: str) -> str:
    return ANSWER_TEMPLATE.format(question=question.strip())


# ---------------------------------------------------------------- engines

class Student:
    """Offline-vLLM engine: deterministic, logprob-capable, completion-only."""

    def __init__(self, checkpoint: str, *, max_model_len: int = 4096, seed: int = 3407):
        from vllm import LLM
        self.llm = LLM(model=checkpoint, max_model_len=max_model_len, seed=seed,
                       dtype="bfloat16")
        self.tokenizer = self.llm.get_tokenizer()
        self.max_model_len = max_model_len

    def score(self, texts: list[str], *, max_len: int) -> list[tuple[float | None, int]]:
        from vllm import SamplingParams
        params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
        prompts, index = [], []
        token_lists: list[list[int]] = []
        for i, text in enumerate(texts):
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            ids = ids[:min(max_len, self.max_model_len - 1)]
            if len(ids) < MIN_SCORE_TOKENS:
                token_lists.append([])
                continue
            token_lists.append(ids)
            prompts.append({"prompt_token_ids": ids})
            index.append(i)
        results: list[tuple[float | None, int]] = [(None, 0)] * len(texts)
        if prompts:
            outs = self.llm.generate(prompts, params)
            for slot, out in zip(index, outs):
                results[slot] = mean_nll(token_lists[slot], out.prompt_logprobs)
        return results

    def complete(self, prompts: list[str], *, max_tokens: int, temperature: float,
                 stop: list[str] | None = None) -> list[str]:
        from vllm import SamplingParams
        params = SamplingParams(max_tokens=max_tokens, temperature=temperature,
                                stop=stop or [])
        outs = self.llm.generate(prompts, params)
        return [out.outputs[0].text for out in outs]


class ServerStudent:
    """OpenAI-compatible server engine (a managed `student.py serve` or any vllm serve).

    Server-mode score tokenizes via the server's /tokenize endpoint and sends token
    arrays, so it scores exactly the same first max_len tokens as the offline engine.
    """

    def __init__(self, model: str, base_url: str, *, workers: int = 16,
                 max_model_len: int = 4096):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.workers = workers
        self.max_model_len = max_model_len

    def _post(self, route: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}{route}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode())

    def _map(self, fn, items):
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(fn, items))

    def _tokenize(self, text: str) -> list[int]:
        out = self._post("/tokenize", {
            "model": self.model, "prompt": text, "add_special_tokens": False})
        return out["tokens"]

    def score(self, texts: list[str], *, max_len: int) -> list[tuple[float | None, int]]:
        limit = min(max_len, self.max_model_len - 1)

        def one(text: str) -> tuple[float | None, int]:
            ids = self._tokenize(text)[:limit]
            if len(ids) < MIN_SCORE_TOKENS:
                return None, 0
            out = self._post("/v1/completions", {
                "model": self.model, "prompt": ids, "max_tokens": 1,
                "temperature": 0.0, "echo": True, "logprobs": 0})
            token_logprobs = out["choices"][0]["logprobs"]["token_logprobs"]
            return mean_nll_from_list(token_logprobs[:len(ids)])

        return self._map(one, texts)

    def complete(self, prompts: list[str], *, max_tokens: int, temperature: float,
                 stop: list[str] | None = None) -> list[str]:
        def one(prompt: str) -> str:
            out = self._post("/v1/completions", {
                "model": self.model, "prompt": prompt, "max_tokens": max_tokens,
                "temperature": temperature, "stop": stop or []})
            return out["choices"][0]["text"]

        return self._map(one, prompts)


def parse_server_target(target: str) -> tuple[str, str]:
    if not target.startswith("server:") or "@" not in target:
        raise SystemExit(f"invalid --target {target!r}; expected server:<model>@<url>")
    model, base_url = target.removeprefix("server:").split("@", 1)
    return model, base_url


# ---------------------------------------------------------------- serve manager

def scratch_dir():
    from workspace import Workspace
    return Workspace.resolve(REPO).apply_environment().scratch_dir


def server_ready(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=3) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


def read_state(scratch: Path) -> dict | None:
    path = scratch / SERVER_STATE
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def stop_server(scratch: Path, *, quiet: bool = False) -> bool:
    state = read_state(scratch)
    if state is None:
        return False
    pid = int(state["pid"])
    if pid_alive(pid):
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGTERM)
        for _ in range(60):
            if not pid_alive(pid):
                break
            time.sleep(0.5)
        if pid_alive(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
    (scratch / SERVER_STATE).unlink(missing_ok=True)
    if not quiet:
        print("SERVER_RESULT " + json.dumps(
            {"action": "stopped", "pid": pid, "checkpoint": state.get("checkpoint")},
            sort_keys=True))
    return True


def cmd_serve(args, checkpoint: str) -> None:
    scratch = scratch_dir()
    stop_server(scratch, quiet=True)          # a new checkpoint replaces the old server
    base_url = f"http://127.0.0.1:{args.port}"
    log_path = scratch / SERVER_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    vllm_bin = Path(sys.executable).resolve().parent / "vllm"
    command = [
        str(vllm_bin), "serve", checkpoint,
        "--host", "127.0.0.1", "--port", str(args.port),
        "--served-model-name", SERVED_NAME,
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--dtype", "bfloat16",
    ]
    with log_path.open("a") as log:
        log.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                  f"{' '.join(command)}\n")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    target = f"server:{SERVED_NAME}@{base_url}"
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        if process.poll() is not None:
            tail = "".join(log_path.read_text().splitlines(keepends=True)[-15:])
            raise SystemExit(f"vllm serve exited with {process.returncode}:\n{tail}")
        if server_ready(base_url):
            break
        time.sleep(1.0)
    else:
        process.terminate()
        raise SystemExit(f"server not ready within {args.timeout}s; see {log_path}")
    state = {"pid": process.pid, "port": args.port, "checkpoint": checkpoint,
             "target": target, "log": str(log_path),
             "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (scratch / SERVER_STATE).write_text(json.dumps(state, indent=2, sort_keys=True))
    print("SERVER_RESULT " + json.dumps({"action": "started", **state}, sort_keys=True))


def cmd_serve_status(args) -> None:
    scratch = scratch_dir()
    state = read_state(scratch)
    if state is None:
        print("SERVER_RESULT " + json.dumps({"action": "status", "running": False}))
        return
    alive = pid_alive(int(state["pid"]))
    ready = alive and server_ready(f"http://127.0.0.1:{state['port']}")
    print("SERVER_RESULT " + json.dumps(
        {"action": "status", "running": alive, "ready": ready, **state}, sort_keys=True))


# ---------------------------------------------------------------- main

def resolve_checkpoint(args) -> str:
    if args.run_id:
        from runstore import RunStore
        return str(RunStore(REPO).resolve_checkpoint(args.run_id))
    return args.checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("score", "draft", "answer",
                                     "serve", "serve-stop", "serve-status"))
    who = ap.add_mutually_exclusive_group()
    who.add_argument("--run-id", help="immutable training run whose checkpoint to load")
    who.add_argument("--checkpoint", help="checkpoint path or HF id")
    who.add_argument("--target", help="server:<model>@<base_url> — a running server "
                                      "(see the serve mode); skips engine load")
    ap.add_argument("--in", dest="input", metavar="JSONL")
    ap.add_argument("--out", dest="output", metavar="JSONL")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=2048,
                    help="score: max prompt tokens per row")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="draft/answer: generation budget (defaults: draft 300, answer 96)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="serve: listen port")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.2,
                    help="serve: fraction of VRAM for the server (default leaves room "
                         "for 1B training)")
    ap.add_argument("--timeout", type=int, default=240, help="serve: readiness timeout")
    args = ap.parse_args()

    from workspace import Workspace
    Workspace.resolve(REPO).apply_environment()

    if args.mode == "serve-stop":
        scratch = scratch_dir()
        if not stop_server(scratch):
            print("SERVER_RESULT " + json.dumps({"action": "stopped", "running": False}))
        return
    if args.mode == "serve-status":
        cmd_serve_status(args)
        return
    if args.mode == "serve":
        if not (args.run_id or args.checkpoint):
            raise SystemExit("serve requires --run-id or --checkpoint")
        cmd_serve(args, resolve_checkpoint(args))
        return

    if not (args.run_id or args.checkpoint or args.target):
        raise SystemExit(f"{args.mode} requires --run-id, --checkpoint, or --target")
    if not args.input or not args.output:
        raise SystemExit(f"{args.mode} requires --in and --out")

    required = {"score": "text", "draft": "prompt", "answer": "question"}[args.mode]
    rows = read_rows(Path(args.input), required)
    if args.limit:
        rows = rows[:args.limit]

    if args.target:
        model, base_url = parse_server_target(args.target)
        if not server_ready(base_url):
            raise SystemExit(f"no server responding at {base_url} "
                             "(start one with the serve mode)")
        student = ServerStudent(model, base_url, max_model_len=args.max_model_len)
        engine = args.target
    else:
        engine = resolve_checkpoint(args)
        print(f"[student] {args.mode} x{len(rows)} on {engine}")
        student = Student(engine, max_model_len=args.max_model_len, seed=args.seed)

    if args.mode == "score":
        scored = student.score([row["text"] for row in rows], max_len=args.max_len)
        out_rows = [{**{k: v for k, v in row.items() if k != "text"},
                     "nll": nll, "scored_tokens": n}
                    for row, (nll, n) in zip(rows, scored)]
        usable = [row["nll"] for row in out_rows if row["nll"] is not None]
        summary = {"mode": "score", "rows": len(out_rows), "scored": len(usable),
                   "mean_nll": round(sum(usable) / len(usable), 4) if usable else None}
    elif args.mode == "draft":
        max_tokens = args.max_tokens or 300
        drafts = student.complete([row["prompt"] for row in rows],
                                  max_tokens=max_tokens, temperature=args.temperature)
        out_rows = [{**row, "draft": draft} for row, draft in zip(rows, drafts)]
        summary = {"mode": "draft", "rows": len(out_rows), "max_tokens": max_tokens}
    else:
        max_tokens = args.max_tokens or 96
        answers = student.complete(
            [answer_prompt(row["question"]) for row in rows],
            max_tokens=max_tokens, temperature=args.temperature, stop=ANSWER_STOP)
        out_rows = [{**row, "answer": answer.strip()}
                    for row, answer in zip(rows, answers)]
        summary = {"mode": "answer", "rows": len(out_rows), "max_tokens": max_tokens}

    write_rows(Path(args.output), out_rows)
    print("STUDENT_RESULT " + json.dumps(
        {**summary, "engine": engine, "out": args.output},
        ensure_ascii=False, sort_keys=True))
    # The offline engine can SIGABRT during teardown AFTER all work is complete and
    # written; exit deterministically so drivers can trust the exit code.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
