"""gym.tasks — task registry. A task is prompt + reference solution + verifier binding.

Intake standard (R1): all three pieces or no entry — and the reference solution MUST pass
its own verifier (`intake_check`); tests/test_gym_tasks_gold.py runs this smoke test for
every locally-available task set. A task whose gold can't score 1.0 is a broken referee,
not a hard task.

    from gym import tasks
    rows = tasks.load("corpus_probes", split="dev")
    score = tasks.score(rows[0], some_model_response)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from gym import verifiers


@dataclass
class Task:
    id: str
    prompt: str
    verifier: str                 # name in gym.verifiers.REGISTRY
    meta: dict                    # gold payload the verifier needs
    ref_solution: str             # reference solution; intake requires it to pass
    mode: str = "chat"            # "chat" (instruct) | "complete" (base-model continuation)
    system: str = ""              # optional system prompt (chat mode)
    tags: dict = field(default_factory=dict)   # track/topic/difficulty/max_steps/...


def score(task: Task, response: str) -> float:
    return verifiers.get(task.verifier)(task.prompt, response, task.meta)


def is_correct(task: Task, response: str) -> bool:
    return score(task, response) >= 0.999


def intake_check(rows: list[Task], threshold: float = 0.999, sample: int = 0) -> list[str]:
    """Gold-passes-verifier smoke test. Returns human-readable failures (empty = pass)."""
    failures = []
    for t in rows[:sample] if sample else rows:
        if not (t.prompt and t.ref_solution and t.verifier):
            failures.append(f"{t.id}: incomplete triple (prompt/ref_solution/verifier)")
            continue
        try:
            s = score(t, t.ref_solution)
        except Exception as e:                                    # noqa: BLE001
            failures.append(f"{t.id}: verifier raised {e!r}")
            continue
        if s < threshold:
            failures.append(f"{t.id}: gold scores {s:.3f} < {threshold}")
    return failures


def load(name: str, split: str = "dev", n: int = 0) -> list[Task]:
    """Load a registered task set. Loaders are lazy imports so optional deps stay optional."""
    if name == "corpus_probes":
        from gym.tasks import corpus_probes as m
    elif name == "gsm8k":
        from gym.tasks import gsm8k as m
    elif name == "bench":
        from gym.tasks import bench as m
    elif name == "building":
        from gym.tasks import building as m
    elif name == "toy_agentic":
        from gym.tasks import toy_agentic as m
    else:
        raise KeyError(f"unknown task set '{name}'")
    return m.load(split=split, n=n)
