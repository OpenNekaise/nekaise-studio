"""bench pack — nekaise-bench as a task pack: the FIXED referee of the ceiling phase. Do NOT edit.

Wraps the external, independently-authored corpus-mastery benchmark
(https://github.com/OpenNekaise/nekaise-bench) in the studio's standard pack contract, so
training, GRPO rewards, and eval tooling consume it like any other pack. Grading functions
are IMPORTED from the bench's own harness — never reimplemented — and the dev/test split is
defined HERE, canonically:

    dev  (~75%): the loop's keep/revert signal
    test (~25%): FROZEN — milestone checks only (tools/eval_bench.py enforces --milestone)

Needs a local clone of the bench; location from NEKAISE_BENCH_DIR (default: ../nekaise-bench
next to this repo).

Contract: load_split / is_correct / reward / extract_answer.
Row schema: {"question", "answer", "id", "track", "topic", "difficulty"} where `question` is
the full prompt text (mcq rows include lettered choices + instruction) and `answer` is a
JSON-encoded gold {"track", "answer", "choices"?, "aliases"?} passed verbatim to
is_correct/reward. `system_prompt(track)` gives the matching system message.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH_DIR = Path(os.environ.get("NEKAISE_BENCH_DIR", REPO.parent / "nekaise-bench"))

_harness = None


def harness():
    """The bench's own eval module (prompts + grading) — the single grading truth."""
    global _harness
    if _harness is None:
        runner = BENCH_DIR / "eval_ollama.py"
        if not runner.exists():
            raise FileNotFoundError(
                f"nekaise-bench not found at {BENCH_DIR} — clone "
                f"https://github.com/OpenNekaise/nekaise-bench there, or set NEKAISE_BENCH_DIR")
        spec = importlib.util.spec_from_file_location("nekaise_bench_eval", runner)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _harness = mod
    return _harness


def in_split(qid: str, split: str) -> bool:
    """Deterministic id-hash split. NEVER change this rule — every recorded score depends
    on it (tests pin a fingerprint)."""
    if split == "all":
        return True
    frozen = int(hashlib.md5(qid.encode()).hexdigest(), 16) % 4 == 0   # ~25% -> test
    return frozen if split == "test" else not frozen


def system_prompt(track: str) -> str:
    h = harness()
    return h.MCQ_SYSTEM if track == "mcq" else h.OPEN_SYSTEM


def load_split(split: str = "dev", n: int | None = None) -> list[dict]:
    """Rows {question, answer(JSON gold), id, track, topic, difficulty} for dev|test|all."""
    h = harness()
    rows = []
    for line in (BENCH_DIR / "questions.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        q = json.loads(line)
        if not in_split(q["id"], split):
            continue
        if q["track"] == "mcq":
            question = h.mcq_prompt(q)
            gold = {"track": "mcq", "answer": q["answer"], "choices": q["choices"]}
        else:
            question = q["question"]
            gold = {"track": "open", "answer": q["answer"], "aliases": q.get("aliases") or []}
        rows.append({"question": question, "answer": json.dumps(gold, ensure_ascii=False),
                     "id": q["id"], "track": q["track"], "topic": q.get("topic", ""),
                     "difficulty": q.get("difficulty", "")})
    return rows[:n] if n else rows


def extract_answer(text: str) -> str:
    return harness().strip_think(text).strip()


def reward(prediction: str, gold_answer: str) -> float:
    """1.0/0.0 by the bench's own grading (letter match for mcq; alias/numeric for open)."""
    h = harness()
    g = json.loads(gold_answer)
    ans = h.strip_think(prediction)
    if g["track"] == "mcq":
        pred = h.extract_letter(ans, len(g["choices"]), g["choices"])
        return 1.0 if pred == g["answer"] else 0.0
    return 1.0 if h.grade_open(ans, g["answer"], g.get("aliases")) else 0.0


def is_correct(prediction: str, gold_answer: str) -> bool:
    return reward(prediction, gold_answer) >= 0.999


def load_test(n: int | None = None) -> list[dict]:
    """Back-compat alias."""
    return load_split("test", n)
