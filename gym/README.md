# Nekaise Gym

**The examination hall for building-intelligence models: tasks, verifiers, runner. No
training code — ever.**

Give it any model — a local checkpoint behind `vllm serve`, Ollama, or a frontier API —
and it runs building/HVAC tasks against programmatic verifiers and returns scores. It has
no idea how (or whether) you train. That's the point: the same verdict logic serves as
training reward, data filter, and leaderboard judge, so "correct" means one thing
everywhere.

## Three parts, nothing else

```
tasks/       questions + gold answers + measured difficulty bands
             (dev / frozen / frontier splits, id-hashed)
verifiers/   verify(prompt, response, meta) -> float   # 8 pure functions:
             numeric_cloze · numeric_tolerance · final_number · ontology_qa
             anchor_recall · bench_qa · sparql_exec · sandbox_tests
runner/      point at any OpenAI-compatible endpoint → run tasks → scores
```

## Quick start

```bash
vllm serve openbmb/MiniCPM5-1B &            # or any endpoint / API
python -m gym.runner --base-url http://localhost:8000/v1 \
       --model MiniCPM5-1B --task corpus_probes --split dev
```

## The one rule

studio contains gym; gym never contains studio ([BOUNDARY.md](../BOUNDARY.md),
Article 0). If you found training code in here, that's a bug — file it.

Health check: someone who has never heard of Nekaise Studio should be able to use this
to evaluate GPT. If that stops being true, the boundary broke.

---

*Currently a component of [nekaise-studio](https://github.com/OpenNekaise/nekaise-studio);
becomes a standalone repo once its API is frozen and an external consumer exists. Part of
[OpenNekaise](https://github.com/OpenNekaise). MIT.*
