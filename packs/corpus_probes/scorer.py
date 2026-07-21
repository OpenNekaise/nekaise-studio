"""corpus_probes pack — the studio-owned loop metric for pure-CPT phases. Do NOT edit.

Numeric cloze continuations minted once from the cleaned HVAC corpus (build_probes.py) and
committed: the model continues a sentence prefix, and is correct iff the FIRST number in its
continuation equals the gold value. Deliberately independent of nekaise-bench (which is a
milestone-only external referee since the decoupling reform) and deliberately dense: a small
model under CPT moves here long before it moves on a hardened exam.

Splits (id-hashed, deterministic):
    dev    (~80%) — the loop's keep/revert signal
    frozen (~20%) — milestone confirmation only; report paired flips, never tune on it

Row schema matches the pack contract: {"question","answer","id","track","topic"} where
`question` is the raw continuation prompt (NO chat template — CPT probes run in base-model
completion mode) and `answer` is a JSON-encoded gold {"value","answer"}.

Contract: load_split / is_correct / reward / extract_answer.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parent
_NUM = re.compile(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?")


def _rows() -> list[dict]:
    path = PACK_DIR / "probes.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — mint it once: python {PACK_DIR}/build_probes.py")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def in_split(probe_id: str, split: str) -> bool:
    frozen = int(hashlib.md5(probe_id.encode()).hexdigest(), 16) % 5 == 0
    return frozen if split == "frozen" else (not frozen) if split == "dev" else True


def load_split(split: str = "dev", n: int = 0) -> list[dict]:
    """dev | frozen | all; absorption AND transfer rows (filter by `track` downstream)."""
    rows = [{"question": p["prompt"],
             "answer": json.dumps({"value": p["value"], "answer": p["answer"]}),
             "id": p["id"], "track": p["kind"], "topic": p["topic"]}
            for p in _rows() if in_split(p["id"], split)]
    return rows[:n] if n else rows


def extract_answer(text: str) -> str | None:
    m = _NUM.search(str(text))
    return m.group(0).replace(",", "") if m else None


def is_correct(pred: str, gold: str) -> bool:
    g = json.loads(gold)
    first = extract_answer(str(pred)[:80])   # first number in the continuation window
    if first is None:
        return False
    try:
        return abs(float(first) - float(g["value"])) < 1e-9
    except ValueError:
        return False


def reward(pred: str, gold: str) -> float:
    return 1.0 if is_correct(pred, gold) else 0.0
