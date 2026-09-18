import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { diffWords, lossChart, safeURL, iterationMetrics, modelLabel } from "../dist/lib.js";
import { recoveryNotice, overview, teachingPlan, tokenLedger, iterationView, historyView, reportsView, learningWork } from "../dist/views.js";

test("reports distinguish pending decisions from applied outcomes and escape every narrative", () => {
  const data = { current: { campaign: { status: "running" }, summary: "Currently learning", next_action: "Automatic next step" }, items: [{ id: 1, kind: "failure", reason: "<img src=x>", status: "decided" }] };
  const detail = { recovery: { id: 1, status: "decided", kind: "failure", decision: { action: "continue", reason: "Fixture choice", report: "<script>proposed work</script>" } }, turns: [{ attempt: 1, turn: 1, decision: { report: "<img>Earlier proposal" }, checks: { status: "unsuccessful", error: "<svg>Fixture check failure" } }], events: [{ message: "<iframe>Recorded activity" }] };
  const html = reportsView(data, detail, 1);
  assert.match(html, /Currently learning/);
  assert.match(html, /Automatic application is pending/);
  assert.match(html, /unsuccessful/);
  assert.ok(!/<script>|<img|<svg>|<iframe>/.test(html));
  assert.ok(!/data-action=/.test(html));
  assert.match(reportsView(data, { ...detail, recovery: { ...detail.recovery, status: "resolved", continuation_id: "new-run" } }, 1), /data-campaign="new-run"/);
});

test("history filters keep archived records, lineage and retention reasons accessible", () => {
  const config = { focus: "Heat", teacher_model: "Teacher", student_model: "Student" };
  const rows = [
    { id: "live", name: "Repeated continued", display_name: "Current lessons", parent_campaign_id: "old", status: "running", config, completed_rounds: 3 },
    { id: "old", name: "Old", retention: "archive", retention_reason: "<script>retained evidence</script>", status: "failed", config, completed_rounds: 0 }
  ];
  const html = historyView(rows);
  assert.match(html,/Current lessons/);assert.match(html,/1 archived/);
  assert.match(html,/Continues/);assert.match(html,/data-campaign="old"/);
  assert.match(html,/&lt;script&gt;retained evidence/);assert.ok(!html.includes("<script>"));
  const archived=historyView(rows,{filter:"archived"});
  assert.match(archived,/data-campaign="old"/);assert.ok(!archived.includes('data-campaign="live"'));
  assert.match(historyView(rows,{query:"CURRENT"}),/data-campaign="live"/);
  assert.match(historyView(rows,{query:"absent"}),/No runs match/);
});

const lesson = { id: "l1", kind: "sft", concept: "Heat transfer", prompt: "Task", student_prompt: "Exact task context", student: "Student draft <script>", teacher: "Corrected answer <img>", training_text: "Teacher-chosen <sequence>", errors: ["Wrong unit"], evidence: [], sources: [], document: { id: "", text: "", selection_reason: "Teacher choice", replay: false }, gate: { passed: true, mode: "trusted_teacher", reason: "Teach" } };
const question = { id: "e1", question: "Which unit?", student: "Student evaluation answer", reference: "Reference evaluation answer", rubric: ["Name the unit"], grade: { score: .25, verdict: "partial", feedback: "Teacher evaluation feedback", needs_practice: true } };
const round = { id: "r1", number: 1, status: "complete", lessons: [lesson], evaluations: [question], metrics: [{ step: 1, loss: 2.7 }], score: .25, model_before: "/private/checkpoint", created_at: "2026-09-14T12:00:00Z" };
const snapshot = { campaign: { id: "c1", status: "complete", config: {} }, rounds: [round], round, iterations: { r1: round }, stages: [], events: [], teacher_usage: { calls: 5 } };

test("paired assessment exposes both answers and dimensions without claiming unmatched gain", () => {
  const paired = { ...question, grade: { ...question.grade, dimensions: [{ name: "Meaning <script>", score: .8, feedback: "Equivalent <img>" }] },
    comparison: { weights_changed: true, generation: { text: "Before <svg>" }, grade: { score: .2, feedback: "Before feedback", dimensions: [{ name: "Meaning <script>", score: .2 }] }, conditions: { comparable: false } } };
  const local = { ...round, evaluations: [paired] };
  const html = iterationView({ ...snapshot, iterations: { r1: local } }, { selectedRound: "r1" });
  assert.match(html, /Same-question comparison/);
  assert.match(html, /Before &lt;svg&gt;/);
  assert.match(html, /Teacher-defined criteria/);
  assert.match(html, /do not establish a training gain/);
  assert.ok(!/<script>|<img>|<svg>/.test(html));
  paired.comparison.weights_changed = false;
  assert.match(iterationView({ ...snapshot, iterations: { r1: local } }, { selectedRound: "r1" }), /same recorded answer and grade were reused/);
});

test("work accounting labels timer limits and tolerates missing evidence", () => {
  const html = learningWork({ id: "r", number: 4, measured_work_all_attempts: { tokens: 800, updates: 1, elapsed_seconds: 1 }, finished_stage_seconds: 120, teacher_stage_seconds: 100, checkpoint_bytes_present: 1024 ** 3, stage_seconds: { train: 20, grade: 100 } });
  assert.match(html, /800/);
  assert.match(html, /Parameter update loop/);
  assert.match(html, /not GPU utilization/);
  assert.match(html, /excludes between-iteration reviews/);
  assert.equal(learningWork(null), "");
  assert.match(learningWork({ error: "Unavailable <script>" }), /&lt;script&gt;/);
});

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
test("Studio opens with all five charts and lessons remain accessible", () => {
  const html = overview({ ...snapshot, rounds: [{ ...round, id: "r2", number: 2 }, round] });
  for (const title of ["Teacher tokens", "Tokens trained", "Teacher assessment", "Independent eval", "Training loss", "Session time"]) assert.ok(html.includes(title), title);
  assert.equal((html.match(/class="panel usage-card /g) || []).length, 5);
  assert.match(html, /data-round="r1"/); assert.match(html, /data-round="r2"/);
  assert.ok(!html.includes("/private/checkpoint"));
  const lessons = overview(snapshot, "teaching");
  assert.match(lessons, /Inside the lesson/); assert.match(lessons, /Student draft &lt;script&gt;/);
  assert.ok(!lessons.includes("studio-charts"));
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
  const html = historyView([campaign]);
  assert.match(html, /data-campaign="c1"/);
  assert.match(html, /Historical &lt;run&gt;/);
  assert.match(html, /<strong>2<\/strong>/);
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


test("HTTP errors remain useful when servers return plaintext or HTML", async () => {
  const { readAPIResponse } = await import("../dist/lib.js");
  for (const body of ["Internal Server Error", "<html>Bad Gateway</html>"]) {
    await assert.rejects(readAPIResponse(new Response(body, { status: 500 })), /HTTP 500/);
  }
  await assert.rejects(readAPIResponse(new Response('{"detail":"Disk unavailable"}', { status: 503 })), /HTTP 503: Disk unavailable/);
  await assert.rejects(readAPIResponse(new Response('broken', { status: 200 })), /invalid JSON response.*HTTP 200/);
  assert.deepEqual(await readAPIResponse(new Response('{"status":"ok"}')), { status: "ok" });
});
test("work plans remain estimates and shared batch clocks are labeled", () => {
  const html = learningWork({ id: "r", number: 1, measured_work_all_attempts: { tokens: 100, updates: 2, elapsed_seconds: 1 }, stage_seconds: {},
    teacher_work_plan: { estimated_targets_per_pass: 1000, material_strategy: "<script>Coverage", dose_rationale: "<img>Dose" },
    preparation: { targets_per_pass: 50, selected_rows: 2, expected_target_exposure: 100, expected_updates: 2, streams: { corpus: { available_targets: 400, prepared_targets: 20, unused_targets: 380, repeated_targets: 0 } } },
    update_work: { target_capacity: 64, updates_with_known_size: 2, short_updates: 2, mean_targets: 50, mean_fill: 50/64 },
    generation_work: { answers: 4, batches: 1, max_batch_size: 4, generated_tokens: 80, generation_seconds: 2 } });
  assert.match(html, /Teacher estimate/);
  assert.match(html, /not actual training/);
  assert.match(html, /Unused/);
  assert.match(html, /counting each shared batch clock once/);
  assert.match(html, /Mean targets per update/);
  assert.ok(!/<script>|<img>/.test(html));
});


test("a chart-selected older iteration stays in the picker beyond the snapshot window", () => {
  assert.match(iterationView(snapshot, { selectedRound: "not-loaded" }), /Loading iteration/);
  const html = iterationView({ ...snapshot, rounds: [{ ...round, id: "new", number: 150 }], iterations: { r1: round } }, { selectedRound: "r1", diff: false });
  assert.match(html, /value="r1" selected/);
  assert.match(html, /Student draft &lt;script&gt;/);
});
