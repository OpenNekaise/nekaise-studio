# Skill: coapt-cpt

The **Adaptive CPT branch** of nekaise-coapt: `d → r_S → T(d, r_S) → d̃`. The student
drafts over a real corpus chunk `d`; you, the teacher-editor, correct the draft **using
only that chunk** into textbook text `d̃` the student trains on. The draft exists to
expose the student's state — where it confuses terms, garbles numbers, or invents facts —
so your correction lands exactly on those failures. The SOURCE decides what is true; the
DRAFT decides what needs saying.

`<round_dir>` is the active round's directory. Inputs `cpt_draft_prompts.jsonl` (from
`build_data.py make-drafts`) and `docs.jsonl` already exist there.

## Procedure

1. **Student drafts** (GPU, eval env):
   `$NEKAISE_EVAL_PYTHON tools/student.py draft --run-id <S> --in
   <round_dir>/cpt_draft_prompts.jsonl --out <round_dir>/cpt_drafts.jsonl`
   (at R0, `--checkpoint <base_model>` instead). Drafts are greedy and raw — never edit
   this file.
2. **Correct.** For every draft row, read `source_chunk` and `draft`, note the concrete
   errors (swapped terms, wrong values, unsupported claims, dropped key facts), then
   write `teacher_text`:
   - Every fact must be supported by `source_chunk`. Numbers, units, standards, and
     component names appear exactly as the source writes them.
   - Keep the student's correct sentences where they are usable; fix only what is wrong,
     add only the most important facts the draft missed.
   - Plain continuous prose, roughly between half and 1.5× the length of `source_chunk`.
     No bullet lists, headers, or meta commentary — "The student wrote…" is FORBIDDEN;
     the output IS training text.
   - Use the source's terminology and register. If the source's own phrasing is fine,
     keep it verbatim — the student must learn the domain's distribution, not your voice.
3. **Gate — a second, separate pass over the finished file.** Re-read each
   `teacher_text` against `source_chunk` claim by claim: every number, name, and causal
   statement must trace to the source. Set `gate.passed` accordingly; be strict — a
   single unsupported claim fails the row. Failed rows STAY in the file (evidence); the
   mixer excludes them.
4. **Write** `<round_dir>/cpt_teacher.jsonl`, one row per draft:

```json
{"id": "<draft id>", "doc_id": "...", "topic": "...", "probe_type": "continuation",
 "source_chunk": "...", "student_draft": "...",
 "student_errors": ["confused __enter__ with __exit__", "invented a 45 min figure"],
 "teacher_text": "...",
 "gate": {"passed": true, "reason": "all claims traced to source"}}
```

5. **Pre-training NLL baseline** for next round's loop-closure check:
   `jq -c '{id, doc_id, text: .teacher_text}' <round_dir>/cpt_teacher.jsonl >
   <round_dir>/teacher_texts.jsonl`, then
   `$NEKAISE_EVAL_PYTHON tools/student.py score --run-id <S> --in
   <round_dir>/teacher_texts.jsonl --out <round_dir>/teacher_nll_pre.jsonl`.

## Hard rules

- **The corpus is the only source of truth.** Never import outside knowledge into
  `teacher_text`, even when you know it is correct — an unsupported-but-true fact still
  fails the gate. What the source does not say, the round does not teach.
- Never open `gym/tasks/corpus_probes/probes.jsonl`. You teach from the document, blind
  to the referee.
- Prompts, length bounds, and gate criteria are frozen across rounds of a campaign.
- Every row keeps full provenance (`source_chunk`, `student_draft`, `student_errors`);
  analysis reads those fields, training reads only `teacher_text`.
