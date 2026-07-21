"""gsm8k task set — the public bootstrap that proves the loop on a bare clone."""
from __future__ import annotations

from gym.tasks import Task


def load(split: str = "test", n: int = 0) -> list[Task]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if n:
        ds = ds.select(range(min(n, len(ds))))
    return [Task(id=f"gsm8k-{split}-{i}", prompt=r["question"], verifier="final_number",
                 meta={"gold": r["answer"]}, ref_solution=r["answer"], mode="chat")
            for i, r in enumerate(ds)]
