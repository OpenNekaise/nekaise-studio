import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { diffWords, lossChart, safeURL, iterationMetrics, modelLabel } from "../dist/lib.js";
import { recoveryNotice, overview, teachingPlan, tokenLedger, iterationView, historyView } from "../dist/views.js";

const lesson = { id: "l1", kind: "sft", concept: "Heat transfer", prompt: "Task", student_prompt: "Exact task context", student: "Student draft <script>", teacher: "Corrected answer <img>", training_text: "Teacher-chosen <sequence>", errors: ["Wrong unit"], evidence: [], sources: [], document: { id: "", text: "", selection_reason: "Teacher choice", replay: false }, gate: { passed: true, mode: "trusted_teacher", reason: "Teach" } };
const question = { id: "e1", question: "Which unit?", student: "Student evaluation answer", reference: "Reference evaluation answer", rubric: ["Name the unit"], grade: { score: .25, verdict: "partial", feedback: "Teacher evaluation feedback", needs_practice: true } };
const round = { id: "r1", number: 1, status: "complete", lessons: [lesson], evaluations: [question], metrics: [{ step: 1, loss: 2.7 }], score: .25, model_before: "/private/checkpoint", created_at: "2026-09-14T12:00:00Z" };
const snapshot = { campaign: { id: "c1", status: "complete", config: {} }, rounds: [round], round, iterations: { r1: round }, stages: [], events: [], teacher_usage: { calls: 5 } };

test("untrusted student and teacher text cannot become HTML", () => {
  const result = diffWords("<script>alert(1)</script> watts", "<img src=x onerror=alert(1)> kelvin per watt");
  assert.ok(!result.before.includes("<script>"));
  assert.ok(!result.after.includes("<img"));
  assert.ok(result.after.includes("<mark>"));
  assert.equal(safeURL("javascript:alert(1)"), "#");
});
test("diff preserves unchanged passages", () => {
  assert.deepEqual(diffWords("The heat flows.", "The heat flows."), { before: "The heat flows.", after: "The heat flows." });
});
test("loss chart uses measured steps and handles empty and single-step runs", () => {
  assert.match(lossChart([]), /when training starts/);
  const chart = lossChart([{ step: 1, loss: 2.7 }]);
  assert.ok(!chart.includes("NaN"));
  assert.match(chart, /Step 1: loss 2.7000/);
  assert.match(lossChart([{ step: 1, loss: NaN }]), /when training starts/);
});
test("iteration loss preserves measured values when round step counters restart", () => {
  const rounds = [{ number: 2, metrics: [{ step: 1, loss: 6 }, { step: 2, loss: 5 }] }, { number: 1, metrics: [{ step: 1, loss: 3 }, { step: 2, loss: 2 }] }];
  const result = iterationMetrics(rounds);
  assert.deepEqual(result.map(m => [m.step, m.round_number, m.round_step, m.loss]), [[1, 1, 1, 3], [2, 1, 2, 2], [3, 2, 1, 6], [4, 2, 2, 5]]);
  assert.equal(rounds[0].metrics[0].step, 1);
  assert.match(lossChart(result), /Iteration 2, step 1: loss 6.0000/);
});
test("waiting remains visible with the next retry and escaped error", () => {
  const html = recoveryNotice({ campaign: { status: "waiting" }, recovery: { error: "<script>quota</script>", retry_at: "2026-09-15T12:00:00Z" } });
  assert.match(html, /Next automatic attempt/);
  assert.ok(!html.includes("<script>"));
  assert.equal(recoveryNotice(snapshot), "");
});
test("overview groups every supplied iteration and keeps only the requested working panels", () => {
  const html = overview({ ...snapshot, rounds: [{ ...round, id: "r2", number: 2 }, round] });
  assert.match(html, /Training loss/);
  assert.match(html, /Happening in the studio/);
  assert.match(html, /Inside the teaching room/);
  assert.match(html, /data-round="r1"/);
  assert.match(html, /data-round="r2"/);
  assert.ok(!/CURRENT ROUND|Prepare dataset|Learning, one step|Teacher's plan|stat-card/.test(html));
  assert.ok(!html.includes("/private/checkpoint"));
});
test("opening an iteration includes student work, revisions and evaluation results together", () => {
  const html = iterationView(snapshot, { selectedRound: "r1", diff: false });
  for (const content of ["Student draft &lt;script&gt;", "Corrected answer &lt;img&gt;", "Student evaluation answer", "Reference evaluation answer", "Teacher evaluation feedback", "25%", "Exact training text", "Teacher-chosen &lt;sequence&gt;", "no corpus source declared"]) assert.ok(html.includes(content), content);
  assert.ok(!html.includes("<script>"));
  assert.ok(!html.includes("<img>"));
});
test("old iteration details do not show the current iteration's lesson or assessment", () => {
  const old = { ...round, id: "old", number: 7, lessons: [{ ...lesson, student: "Older attempt" }], evaluations: [{ ...question, student: "Older assessment" }] };
  const html = iterationView({ ...snapshot, iterations: { r1: round, old }, rounds: [round, old] }, { selectedRound: "old" });
  assert.match(html, /Older attempt/);
  assert.match(html, /Older assessment/);
  assert.ok(!html.includes("Student evaluation answer"));
});
test("empty diagnostic iterations do not invent teaching or training results", () => {
  const diagnostic = { ...round, lessons: [], evaluations: [], metrics: [], score: null, curriculum: { lessons: [], readings: [], replay: [], train_epochs: 0, notes: "Assess later", evaluation_instructions: "" } };
  const html = iterationView({ ...snapshot, iterations: { r1: diagnostic } }, { selectedRound: "r1" });
  assert.match(html, /No lessons recorded/);
  assert.match(html, /No assessment recorded/);
  assert.match(html, /No weight updates requested/);
  assert.ok(!/NaN|undefined/.test(html));
});
test("token details distinguish prepared tokens from measured training exposure", () => {
  const html = tokenLedger({ round: { metrics: [], token_ledger: { total_tokens: 80, streams: { teacher: { prepared_tokens: 60 }, corpus: { prepared_tokens: 20 } }, missing_streams: ["replay"] } } });
  assert.match(html, /75%/);
  assert.match(html, /25%/);
  assert.match(html, /Trained/);
  assert.ok(!html.includes("NaN"));
});
test("teacher plan and durable notes remain available within an iteration", () => {
  const plan = { lessons: [{}], readings: [], replay: [], train_epochs: 2, notes: "<script>plan</script>", evaluation_instructions: "Choose a diagnostic" };
  const html = teachingPlan({ round: { curriculum: plan, teaching_strategy: { student_notes: "<img src=x>", next_round_instructions: "Review older lessons", action: "continue", reason: "Teacher decision" } } });
  assert.match(html, /1 task/);
  assert.match(html, /2 passes/);
  assert.match(html, /Review older lessons/);
  assert.ok(!html.includes("<script>"));
  assert.ok(!html.includes("<img"));
});
test("previous runs show real outcomes and open the corresponding run", () => {
  const campaign = { id: "c1", name: "Historical <run>", status: "failed", completed_rounds: 2, created_at: "2026-09-14T12:00:00Z", config: { focus: "Heat transfer", teacher_model: "Teacher", student_model: "/private/model/checkpoint" } };
  const html = historyView([campaign], { c1: snapshot });
  assert.match(html, /data-campaign="c1"/);
  assert.match(html, /Historical &lt;run&gt;/);
  assert.match(html, /2.700/);
  assert.match(html, /Heat transfer/);
  assert.match(html, /failed/);
  assert.ok(!html.includes("/private/model/checkpoint"));
  assert.equal(modelLabel("/private/model/checkpoint"), "Saved checkpoint");
});
test("the shell uses the mountain logo and has no sidebar or removed page copy", () => {
  const html = readFileSync(new URL('../dist/index.html', import.meta.url), 'utf8');
  assert.match(html, /assets\/nekaise-mountain.png/);
  assert.match(html, /<title>Nekaise Studio<\/title>/);
  assert.ok(!/LEARNING STUDIO|THE LEARNING LOOP|A little better|<aside|CAMPAIGNS|data-view="evaluation"|THE ROUND JOURNAL/.test(html));
  assert.match(html, /data-view="history"/);
});
