"""Evaluation orchestration: tasks × generator → records + summary. Pure plumbing —
the verdict per record comes from gym.verifiers only (the same functions training uses).
"""
from __future__ import annotations

from gym.tasks import Task, is_correct, score


def evaluate(rows: list[Task], gen, *, max_tokens: int = 128, temperature: float = 0.0,
             batch_size: int = 64, progress=None) -> dict:
    """Run every task through `gen` (VllmEngine/OpenAIServer/compatible) and verify.

    Returns {"records": [...], "n", "score" (mean), "acc" (fraction fully correct),
    "by_track": {track: acc}} — tracks come from tags["track"] when present.
    """
    records = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        completes = [t for t in batch if t.mode == "complete"]
        chats = [t for t in batch if t.mode != "complete"]
        responses: dict[str, str] = {}
        if completes:
            outs = gen.complete([t.prompt for t in completes],
                                max_tokens=max_tokens, temperature=temperature)
            responses.update({t.id: o for t, o in zip(completes, outs)})
        if chats:
            msgs = [([{"role": "system", "content": t.system}] if t.system else [])
                    + [{"role": "user", "content": t.prompt}] for t in chats]
            outs = gen.chat(msgs, max_tokens=max_tokens, temperature=temperature)
            responses.update({t.id: o for t, o in zip(chats, outs)})
        for t in batch:
            resp = responses[t.id]
            records.append({"id": t.id, "response": resp, "score": score(t, resp),
                            "correct": is_correct(t, resp), **t.tags})
        if progress:
            progress(len(records), len(rows), records)

    def acc(rs):
        return round(sum(r["correct"] for r in rs) / len(rs), 4) if rs else None

    tracks = sorted({r.get("track") for r in records if r.get("track")})
    return {"records": records, "n": len(records), "acc": acc(records),
            "score": round(sum(r["score"] for r in records) / len(records), 4)
            if records else None,
            "by_track": {tr: acc([r for r in records if r.get("track") == tr])
                         for tr in tracks}}
