# Skill: coapt-sft

The **Personalized SFT branch** of nekaise-coapt: `d → q_T → a_S → T(d, q_T, a_S) → a*`.
You author questions from a real corpus document, the student answers them closed-book,
and you correct each answer **using only that document** into the final `a*`. The
training pair is `(q_T, a*)` — the mixer serializes it as
`Question: …\nAnswer: …`, the same frozen format `tools/student.py answer` uses, so what
the student practices is exactly what the diagnosis asks.

`<round_dir>` is the active round's directory; `frontier.jsonl` and `docs.jsonl` already
exist there.

## Procedure

1. **Author questions.** For each frontier doc (text in `docs.jsonl`), write 2 questions
   spanning different types from `{fact, explain, contrast, apply}` across the round.
   Each row carries `question`, `reference_answer`, `evidence` (a VERBATIM substring of
   the doc that settles the answer), and `qtype`:
   - The question must be answerable from the document alone, closed-book, by someone
     who studied it — no trivia about phrasing, no "according to the text".
   - Phrase the question in your own words, not the source sentence's — a question whose
     answer is copyable from its own wording tests copying, not knowledge.
   - `reference_answer` is short and factual (1–3 sentences).
   Write `<round_dir>/sft_questions.jsonl`:

```json
{"id": "<doc_id>::q0", "doc_id": "...", "topic": "...", "qtype": "fact",
 "question": "...", "reference_answer": "...", "evidence": "..."}
```

2. **Student answers closed-book** (GPU, eval env):
   `$NEKAISE_EVAL_PYTHON tools/student.py answer --run-id <S> --in
   <round_dir>/sft_questions.jsonl --out <round_dir>/sft_answers.jsonl`
   (at R0, `--checkpoint <base_model>`). The student never sees the document here —
   that is the point.
3. **Correct.** For each row, read the document, the question, and the student's
   `answer`; record `student_verdict` (`correct` | `partial` | `wrong`), then write the
   final `answer` `a*`:
   - Keep the student's correct words; fix only what is wrong or missing.
   - Every claim in `a*` must trace to `evidence` (or the surrounding document text).
   - Concise: 1–3 sentences, terminology from the source.
4. **Gate — a second pass over the finished file.** Check that `evidence` is a verbatim
   substring of the doc's text in `docs.jsonl` and that every claim in `answer` traces to
   the document. One unsupported claim fails the row. Failed rows stay in the file; the
   mixer excludes them.
5. **Write** `<round_dir>/sft_final.jsonl`:

```json
{"id": "<doc_id>::q0", "doc_id": "...", "topic": "...", "qtype": "fact",
 "question": "...", "reference_answer": "...", "evidence": "...",
 "student_answer": "...", "student_verdict": "wrong",
 "answer": "...", "gate": {"passed": true, "reason": "claims trace to evidence"}}
```

   The round's `student_verdict` rate is this round's closed-book baseline — next round's
   loop-closure check re-asks these questions and must beat it.

## Hard rules

- **The corpus is the only source of truth.** No outside knowledge in `a*`, even when
  correct. What the evidence cannot support does not enter the training set.
- Never open `gym/tasks/corpus_probes/probes.jsonl`. Questions come from the document;
  you author blind to the referee, and these questions are training data, never an exam.
  The only exam is the frozen probe referee.
- Question style, counts, and gate criteria are frozen across rounds of a campaign.
- The student's answer is context for YOU; the training pair the mixer builds is
  `(question, a*)` only.
