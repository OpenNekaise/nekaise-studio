import { gpqaCard } from "./gpqa.js?v=ebb687a0fdd7";
import { efficiencyCard, ratioLabel } from "./efficiency.js?v=ebb687a0fdd7";
import { escapeHTML as e, number, duration, time, percent, safeURL, diffWords, lossChart, iterationMetrics, dateLabel, modelLabel } from "./lib.js?v=ebb687a0fdd7";
import { usageCharts, elapsedLabel, phaseFor, phases } from "./telemetry.js?v=ebb687a0fdd7";
import { benchmarkCard, benchmarkHistory } from "./benchmark.js?v=ebb687a0fdd7";
import { teacherAssessmentCard } from "./teacher-assessment.js?v=ebb687a0fdd7";
import { experimentCard, experimentsView } from "./experiments.js?v=ebb687a0fdd7";

export const badge = (status, small = false) => `<span class="status ${e(status)} ${small ? "small" : ""}">${e(status || "pending")}</span>`;
const empty = (text) => `<div class="empty-inline">${e(text)}</div>`;
const count = (n, noun) => `${number(n)} ${n === 1 ? noun : noun === "pass" ? "passes" : noun === "match" ? "matches" : noun + "s"}`;
const recorded = (text, fallback) => text ? e(text) : `<span class="muted">${e(fallback)}</span>`;
const detailsFor = (s, r) => s.iterations?.[r.id] || (s.round?.id === r.id ? s.round : null);

export function recoveryNotice(s) {
  if (!["waiting", "recovering"].includes(s.campaign.status)) return "";
  const recovery = s.recovery;
  const title = recovery?.kind === "history_review" ? "Orchestrator is reviewing run history and logs" : s.campaign.status === "recovering" ? "Orchestrator is handling an interruption" : "Automatic execution is waiting";
  const next = recovery?.retry_at ? `Next automatic attempt: ${new Date(recovery.retry_at).toLocaleString("en-GB")}.` : "Progress is saved. Execution details are in Report.";
  return `<div class="notice recovery-notice" role="status"><div><strong>${title}</strong><span>${e(next)}</span></div><button class="text-button" data-view="reports">Read report ↗</button><details data-detail="recovery-error"><summary>Interruption details</summary><p>${e(recovery?.error || s.campaign.error || "Progress is saved.")}</p></details></div>`;
}

export function tokenLedger(s) {
  const ledger = s.round?.token_ledger;
  if (!ledger) return "";
  const consumed = s.round.metrics?.at(-1)?.stream_tokens || {};
  const labels = { teacher: "Teacher lessons", corpus: "Source text", replay: "Review material" };
  return `<div class="table-scroll"><table><caption>Mid-training recipe · training target tokens</caption><thead><tr><th>Material</th><th>Prepared</th><th>Share</th><th>Trained</th></tr></thead><tbody>${Object.entries(ledger.streams).map(([key, value]) => `<tr><td>${e(labels[key] || key)}</td><td>${number(value.prepared_tokens)}</td><td>${percent(ledger.total_tokens ? value.prepared_tokens / ledger.total_tokens : 0)}</td><td>${number(consumed[key])}</td></tr>`).join("")}</tbody></table></div>${ledger.missing_streams?.length ? '<p class="quiet">Unavailable material shares were redistributed across available streams.</p>' : ""}`;
}

export function teachingPlan(s) {
  const plan = s.round?.curriculum, strategy = s.round?.teaching_strategy;
  if (!plan && !strategy) return "";
  return `<details class="panel detail-panel" data-detail="teacher-plan"><summary>Teacher plan & next steps</summary><div class="detail-body">${plan ? `<p class="quiet">${count(plan.lessons.length, "task")} · ${count(plan.readings.length, "reading")} · ${count(plan.replay.length, "review lesson")} · ${count(plan.train_epochs, "pass")}</p><p class="prose">${e(plan.notes)}</p><h3>Assessment plan</h3><p class="prose">${e(plan.evaluation_instructions)}</p>` : ""}${strategy ? `<h3>Student notes</h3><p class="prose">${e(strategy.student_notes)}</p><h3>Next iteration · ${e(strategy.action)}</h3><p class="prose">${e(strategy.next_round_instructions)}</p><p class="quiet">${e(strategy.reason)}</p>` : ""}</div></details>`;
}

function lessonSources(row) {
  const sources = row.sources?.length ? row.sources : row.document?.id ? [row.document] : [];
  return `<details class="source-details" data-detail="sources-${e(row.id)}"><summary>Recipe material, sources & exact training text</summary><div class="detail-body">${row.kind ? `<p class="quiet">Recipe material: ${e(row.kind.toUpperCase())}</p>` : ""}${sources.length ? sources.map(d => `<a href="${safeURL(d.url)}" target="_blank" rel="noopener noreferrer">${e(d.title)}</a><blockquote>${e(d.text)}</blockquote><p class="quiet">${e(d.license)} · ${e(d.topic)}${d.source_sha256 ? ` · <code title="${e(d.source_sha256)}">${e(d.source_sha256.slice(0, 12))}</code>` : ""}</p>`).join("") : '<p class="quiet">Teacher-authored material · no corpus source declared.</p>'}${row.evidence?.length ? `<h3>Teacher evidence</h3>${row.evidence.map(x => `<blockquote>${e(x)}</blockquote>`).join("")}` : ""}${row.training_text ? `<h3>Exact training text</h3><div class="prose">${e(row.training_text)}</div>` : ""}</div></details>`;
}

export function eventsView(events, expanded = false) {
  if (!events?.length) return empty("No activity recorded yet.");
  return `<ul class="activity-feed">${events.slice(0, expanded ? 100 : 6).map(item => `<li class="activity-item"><span class="event-dot ${e(item.kind)}" aria-hidden="true"></span><div>${item.round_id ? `<button class="event-link" data-round="${e(item.round_id)}">${e(item.message)}</button>` : e(item.message)}</div><time datetime="${e(item.created_at)}" title="${e(dateLabel(item.created_at))}">${time(item.created_at)}</time></li>`).join("")}</ul>`;
}

function chartPanel(s) {
  const metrics = iterationMetrics(s.rounds.map(r => detailsFor(s, r) || r));
  const last = metrics.at(-1);
  return `<section class="panel loss-panel"><div class="panel-heading"><div><h2>Training loss</h2><p class="quiet">Selected run only · ${count(s.rounds.length, "iteration")}</p></div>${last ? `<span class="last-loss">${number(last.loss, 3)}<small>latest</small></span>` : ""}</div><div class="chart">${lossChart(metrics)}</div><div class="panel-footer"><span>${count(metrics.length, "recorded update")}</span><span>${last ? `Iteration ${last.round_number} · step ${last.round_step}` : "No weight updates recorded"}</span></div></section>`;
}

function iterationList(s) {
  if (!s.rounds.length) return empty("No iterations yet. Start this run to begin.");
  return `<div class="iteration-list">${s.rounds.map(r => {
    const detail = detailsFor(s, r), metrics = detail?.metrics || [], lessons = detail?.lessons || [];
    const evaluations = detail?.evaluations || [], assessed = evaluations.filter(item => item.grade).length;
    const topics = [...new Set(lessons.map(l => l.concept).filter(Boolean))];
    const summary = topics.join(" · ") || detail?.curriculum?.notes || (detail ? "No lessons recorded" : "Loading iteration details…");
    return `<button class="iteration-row" data-round="${e(r.id)}"><span class="iteration-number">${String(r.number).padStart(2, "0")}</span><span class="iteration-description"><strong>Iteration ${number(r.number)}</strong><span class="iteration-topics">${e(summary)}</span><span class="iteration-meta">${detail ? `${count(lessons.length, "lesson")} · ${count(assessed, "assessed question")} · ${count(metrics.length, "recorded update")}` : "Details available on opening"}</span></span><span class="iteration-result">${badge(r.status, true)}<time datetime="${e(r.created_at)}">${e(dateLabel(r.created_at))}</time></span><span class="row-arrow" aria-hidden="true">↗</span></button>`;
  }).join("")}</div>`;
}

const stageLabels = { select: "Teacher is planning", plan: "Preparing lesson prompts", draft: "Student is practising", revise: "Teacher is revising answers", expand: "Material authors are generating", material_select: "Teacher is selecting material", gate: "Recording teacher choices", freeze: "Preparing the training dataset", train: "Student weights are updating", evaluate: "Teacher is preparing assessment", answer: "Student is answering assessment", grade: "Teacher is grading answers", adapt: "Teacher is reflecting" };
const runLabel = c => c?.display_name || c?.name || c?.id || "No run selected";
const shortId = id => id?.replace(/^campaign_/, "") || "";
const phaseLabel = (campaign, round) => ["running", "pausing", "stopping"].includes(campaign?.status) ? stageLabels[round?.stage] || "Preparing the next iteration" : phaseFor(campaign?.status, round?.stage).name;

export function studioNavigation(section = "overview") {
  return `<nav class="workspace-tabs" aria-label="Studio sections">${[["overview", "Overview"], ["teaching", "Lessons & assessments"], ["experiments", "Experiments"], ["activity", "Activity"]].map(([id, name]) => `<button data-studio-section="${id}" class="workspace-tab ${section === id ? "active" : ""}" ${section === id ? 'aria-current="page"' : ""}>${name}</button>`).join(" ")}</nav>`;
}

export function studioScope(s) {
  const historical = s.currentCampaign && s.currentCampaign.id !== s.campaign.id;
  return `<div class="workspace-heading studio-heading"><div><h1>CoAPT Mid-training</h1><p>${e(s.campaign.config.focus)}</p></div><div class="scope-tools"><span class="run-chip" title="${e(s.campaign.id)}">${historical ? "Previous run" : "Run"} <code>${e(shortId(s.campaign.id))}</code></span>${historical ? `<button class="text-button" data-campaign="${e(s.currentCampaign.id)}">Back to current run ↗</button>` : ""}</div></div>`;
}

function currentIteration(s) {
  const round = s.round, phase = phaseFor(s.campaign.status, round?.stage);
  const detail = round ? detailsFor(s, round) || round : null;
  const lessons = detail?.lessons || [], assessments = detail?.evaluations || [];
  const latest = detail?.metrics?.at(-1);
  return `<section class="panel teaching-current"><div class="teaching-current-top"><div><div class="eyebrow">Current iteration</div><h2>${round ? `Iteration ${number(round.number)}` : "Not started"}</h2></div>${round ? `<button class="button secondary" data-round="${e(round.id)}">Open lessons ↗</button>` : ""}</div><ol class="phase-steps" aria-label="Teaching phases">${phases.map((p, i) => `<li class="phase-${p.tone} ${phase.moving && p.tone === phase.tone && s.campaign.status !== "recovering" ? "current" : ""}" ${phase.moving && p.tone === phase.tone && s.campaign.status !== "recovering" ? 'aria-current="step"' : ""}><span class="phase-step-number">${i + 1}</span><span>${p.name}</span></li>`).join(" ")}</ol><dl class="iteration-facts"><div><dt>Seed lessons</dt><dd>${number(lessons.length)}</dd></div><div><dt>Authored candidates</dt><dd>${number(detail?.material_count ?? 0)}</dd></div><div><dt>Questions graded</dt><dd>${assessments.filter(q => q.grade).length}<small> / ${assessments.length}</small></dd></div><div><dt>Trained targets</dt><dd>${Number.isFinite(latest?.tokens) ? number(latest.tokens) : "—"}</dd><span class="fact-note">Latest attempt</span></div></dl></section>`;
}

function sessionStrip(s) {
  const phase = phaseFor(s.campaign.status, s.round?.stage), t = s.telemetry;
  return `<section class="session-strip panel"><div class="session-status"><span class="status-emblem phase-${phase.tone}" aria-hidden="true">${statusIcon(s.campaign.status)}</span><div><span class="eyebrow">${s.round ? `Iteration ${number(s.round.number)}` : "Run status"}</span><h2>${e(phaseLabel(s.campaign, s.round))}</h2></div></div><div class="session-clock"><span>Session time</span><strong data-elapsed="${e(s.campaign.id)}">${elapsedLabel(t?.elapsed_seconds)}</strong><small>Across continuations · pauses excluded</small></div></section>`;
}

function statusIcon(status) {
  const path = status === "recovering" ? '<path d="M19 7a8 8 0 1 0 1 9M19 3v5h-5"/>' : ["waiting", "paused", "stopped"].includes(status) ? '<path d="M9 7v10M15 7v10"/>' : status === "complete" ? '<path d="m6 12 4 4 8-8"/>' : '<path d="m4 13 4-5 4 8 4-10 4 7"/>';
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">${path}</svg>`;
}

export function overview(s, section = "overview") {
  const heading = studioScope(s);
  if (section === "experiments") return heading + studioNavigation(section) + experimentsView(s.experiments);
  if (section === "activity") return heading + studioNavigation(section) + `<section class="panel"><div class="panel-heading"><h2>Execution activity</h2><span class="quiet">Selected run only</span></div>${eventsView(s.events, true)}</section>`;
  const round = s.round, detail = round ? detailsFor(s, round) || round : null;
  const lessons = detail?.lessons || [];
  const preview = lessons.find(row => row.teacher) || lessons[0];
  const iterations = `<section class="panel teaching-room"><div class="panel-heading"><h2>Iterations</h2><span class="quiet">${count(s.rounds.length, "iteration")} in this run</span></div>${iterationList(s)}</section>`;
  if (section === "teaching") return heading + studioNavigation(section) + recoveryNotice(s) + currentIteration(s) + (preview ? `<div class="section-heading"><h2>Inside the lesson</h2><button class="text-button" data-round="${e(round.id)}">All lessons & materials →</button></div>${lessonCard(preview, 0, false)}` : `<div class="teaching-placeholder"><h2>Teaching material</h2><p>Lessons and student attempts appear here when they are recorded.</p></div>`) + iterations;
  return heading + sessionStrip(s) + `<div class="usage-grid studio-charts" aria-label="Training charts">${usageCharts(s.telemetry)}${efficiencyCard(s.telemetry)}${teacherAssessmentCard(s.teacherAssessment, s.round)}${benchmarkCard(s.benchmark, s.benchmarkBrowser)}${gpqaCard(s.gpqa)}</div><div class="chart-scope"><span>${s.telemetry ? `Usage across ${count(s.telemetry.campaign_count, "run")} in this continuation lineage` : "Loading recorded session usage"}</span><span>Independent eval observes checkpoints separately</span></div>${benchmarkHistory(s.benchmarkBrowser?.snapshot || s.benchmark, s.benchmarkBrowser)}${studioNavigation(section)}${recoveryNotice(s)}<div class="studio-detail-grid">${currentIteration(s)}${chartPanel(s)}</div>${iterations}${learningWork(s.round?.learning_work)}`;
}

function lessonCard(row, index, diff, selected) {
  const changes = diff ? diffWords(row.student, row.teacher) : { before: e(row.student), after: e(row.teacher) };
  return `<details class="panel lesson-card" data-detail="lesson-${e(row.id)}" ${index === 0 || selected === row.id ? "open" : ""}><summary><span class="tag">LESSON</span><strong>${e(row.concept || row.prompt || `Lesson ${index + 1}`)}</strong>${row.gate ? `<span class="lesson-choice ${row.gate.passed ? "selected" : "omitted"}">${row.gate.passed ? "Selected for training" : "Omitted"}</span>` : ""}</summary><div class="lesson-body"><div class="prompt-box"><h3>Student prompt</h3><div class="prose">${e(row.student_prompt || row.prompt)}</div></div><div class="revision-columns"><section class="revision-pane"><h3>Student attempt</h3><div class="prose student">${changes.before || '<span class="muted">No student attempt recorded yet.</span>'}</div></section><section class="revision-pane teacher-pane"><h3>Teacher revision</h3><div class="prose">${changes.after || '<span class="muted">No teacher revision recorded yet.</span>'}</div></section></div>${row.errors?.length ? `<div class="teacher-feedback"><h3>Teacher feedback</h3><ul>${row.errors.map(error => `<li>${e(error)}</li>`).join("")}</ul></div>` : ""}${row.gate?.reason ? `<p class="inclusion-reason">${e(row.gate.reason)}</p>` : ""}${lessonSources(row)}</div></details>`;
}

export function materialList(round) {
  if (!round.material_count && !round.materials?.length) return "";
  return `<section class="iteration-materials"><div class="section-heading"><h2>Authored teaching material</h2><span class="quiet">${number(round.material_count ?? round.materials.length)} candidates</span></div>${(round.materials || []).map(row => `<details class="panel lesson-card" data-detail="material-${e(row.id)}"><summary><span class="tag">MATERIAL</span><strong>${e(row.concept)}</strong><span class="lesson-choice ${row.use_for_training ? "selected" : "omitted"}">${row.use_for_training ? "Selected for training" : "Omitted"}</span></summary><div class="lesson-body"><p class="quiet">${e(row.material_origin?.author_id)} · ${e(row.material_origin?.model)}${row.material_origin?.teacher_edited ? " · Edited by teacher" : ""}. Synthetic teaching material; no student attempt was requested.</p>${row.student_prompt ? `<h3>Training prompt</h3><p class="prose">${e(row.student_prompt)}</p>` : ""}<h3>Teaching text</h3><p class="prose">${e(row.teacher)}</p><h3>Teacher selection</h3><p class="prose">${e(row.reason)}</p><p class="quiet">${e(row.material_origin?.review_scope)}</p>${lessonSources(row)}</div></details>`).join("")}${round.material_next_offset != null ? `<button class="button secondary" data-material-next="${e(round.id)}">Show more materials</button>` : ""}</section>`;
}

function authorWork(work) {
  const authors = work.material_author_work, sources = work.material_sources;
  if (!authors && !sources?.targets_by_origin) return "";
  const summary = authors ? `<h3>Material authors</h3><p class="quiet">${Object.entries(authors.jobs).map(([status, n]) => `${number(n)} ${e(status)}`).join(" · ")} · ${number(authors.calls)} requests across attempts.</p><p class="quiet">Reported API tokens: ${number(authors.reported_input_tokens)} input · ${number(authors.reported_output_tokens)} output. ${number(authors.calls_without_usage)} requests have incomplete usage. ${number(authors.reserved_output_tokens_all_attempts)} output tokens reserved across attempts.</p><p class="quiet">${e(authors.limitations)}</p>` : "";
  const breakdown = sources ? `<div class="table-scroll"><table><caption>Prepared targets by material origin · per pass</caption><tbody>${Object.entries(sources.targets_by_origin).map(([origin, n]) => `<tr><th>${e(origin)}</th><td>${number(n)}</td></tr>`).join("")}</tbody></table></div>` : "";
  return summary + breakdown;
}

function assessmentDimensions(item) {
  const dimensions = item.grade?.dimensions || [];
  if (!dimensions.length) return "";
  const before = item.comparison?.grade?.dimensions;
  return `<div class="table-scroll"><table><caption>Teacher-defined criteria</caption><thead><tr><th>Criterion</th>${before ? "<th>Before</th>" : ""}<th>Current</th><th>Feedback</th></tr></thead><tbody>${dimensions.map(d => `<tr><td>${e(d.name)}</td>${before ? `<td>${percent(before.find(b => b.name === d.name)?.score)}</td>` : ""}<td>${percent(d.score)}</td><td>${e(d.feedback)}</td></tr>`).join("")}</tbody></table></div>`;
}

function assessmentComparison(item) {
  const pair = item.comparison;
  if (!pair) return "";
  return `<details class="context-details" data-detail="assessment-comparison-${e(item.id)}"><summary>Same-question comparison${pair.grade && item.grade ? ` · ${percent(pair.grade.score)} → ${percent(item.grade.score)}` : ""}</summary><h3>Answer from the input checkpoint</h3><p class="prose">${recorded(pair.generation?.text, "Empty recorded answer.")}</p>${pair.grade ? `<p class="prose">${e(pair.grade.feedback)}</p>` : ""}<p class="quiet">${!pair.weights_changed ? "Weights were unchanged. The same recorded answer and grade were reused; no training gain was measured." : pair.conditions?.comparable ? "Same recorded prompt tokens, decoding settings and runtime. This is online teaching feedback on this question." : "Generation conditions could not be matched. These scores do not establish a training gain."}</p></details>`;
}

export function learningWork(work) {
  if (!work) return "";
  if (work.error) return `<p class="quiet">${e(work.error)}</p>`;
  const measured = work.measured_work_all_attempts;
  const plan = work.teacher_work_plan, prepared = work.preparation, updates = work.update_work, generated = work.generation_work;
  const dose = prepared ? `<h3>Material prepared per pass</h3><p class="quiet">${number(prepared.targets_per_pass)} targets · ${number(prepared.selected_rows)} selected rows · ${number(prepared.expected_target_exposure)} planned exposure across ${number(prepared.expected_updates)} updates. Prepared and planned counts are not actual training.</p><div class="table-scroll"><table><thead><tr><th>Material</th><th>Available</th><th>Prepared</th><th>Unused</th><th>Repeated</th></tr></thead><tbody>${Object.entries(prepared.streams).map(([name, s]) => `<tr><th>${e(name)}</th><td>${number(s.available_targets)}</td><td>${number(s.prepared_targets)}</td><td>${number(s.unused_targets)}</td><td>${number(s.repeated_targets)}</td></tr>`).join("")}</tbody></table></div><p class="quiet">Target occurrences in selected material, before repeated passes; these counts do not measure distinct knowledge.</p>` : "";
  const intention = plan ? `<h3>Teacher work plan</h3><p class="quiet">Teacher estimate: ${number(plan.estimated_targets_per_pass)} targets per pass.</p><p class="prose">${e(plan.material_strategy)}</p><p class="prose">${e(plan.dose_rationale)}</p>` : "";
  const updateDetails = updates?.updates_with_known_size ? `<div><dt>Mean targets per update</dt><dd>${number(updates.mean_targets, 1)} / ${number(updates.target_capacity)} · ${percent(updates.mean_fill)}</dd></div><div><dt>Partial updates</dt><dd>${number(updates.short_updates)} / ${number(updates.updates_with_known_size)} observed</dd></div>` : "";
  const ratio = work.teacher_efficiency;
  const efficiencyDetails = ratio ? `<h3>Training efficiency</h3><p class="quiet">${ratioLabel(ratio.reported_ratio)} · ${number(ratio.trained_tokens)} trained / ${number(ratio.teacher_tokens)} primary Teacher tokens${ratio.status !== "complete" || !ratio.final ? " · provisional usage" : ""}. Author usage is separate; exposure includes passes and retries.</p>` : "";
  const inference = generated?.batches ? `<h3>Recorded batch inference</h3><p class="quiet">${number(generated.answers)} answers in ${number(generated.batches)} batches · largest batch ${number(generated.max_batch_size)} · ${number(generated.generated_tokens)} generated tokens · ${duration(generated.generation_seconds)} generation time, counting each shared batch clock once. Loading and audit time are excluded.</p>` : "";
  return `<details class="panel detail-panel" data-detail="learning-work-${e(work.id)}"><summary>Measured training work · iteration ${number(work.number)}</summary><div class="detail-body"><dl class="training-details"><div><dt>Recorded training targets</dt><dd>${number(measured.tokens)}</dd></div><div><dt>Recorded updates</dt><dd>${number(measured.updates)}</dd></div>${updateDetails}<div><dt>Parameter update loop</dt><dd>${duration(measured.elapsed_seconds)}</dd></div><div><dt>Finished stage time</dt><dd>${duration(work.finished_stage_seconds)}</dd></div><div><dt>Teacher stage time</dt><dd>${duration(work.teacher_stage_seconds)}</dd></div><div><dt>Checkpoint bytes still present</dt><dd>${work.checkpoint_bytes_present == null ? "—" : `${number(work.checkpoint_bytes_present / 1024 ** 3, 2)} GiB`}</dd></div></dl><p class="quiet">Recorded work includes retries. Stage time excludes between-iteration reviews and unfinished stages. Update-loop time excludes loading, inference and saving; it is not GPU utilization.</p>${intention}${dose}${efficiencyDetails}${inference}${authorWork(work)}<div class="table-scroll"><table><caption>Finished stage time</caption><tbody>${Object.entries(work.stage_seconds).map(([name, seconds]) => `<tr><th>${e(name)}</th><td>${duration(seconds)}</td></tr>`).join("")}</tbody></table></div></div></details>`;
}

function assessmentCard(item, index) {
  return `<details class="panel assessment-card" data-detail="assessment-${e(item.id)}" ${index === 0 ? "open" : ""}><summary><span class="question-number">${index + 1}</span><strong>${e(item.question)}</strong>${item.grade ? badge(item.grade.verdict, true) : '<span class="quiet">Pending</span>'}</summary><div class="detail-body">${item.student_prompt && item.student_prompt !== item.question ? `<details class="context-details" data-detail="question-context-${e(item.id)}"><summary>Full student prompt</summary><p class="prose">${e(item.student_prompt)}</p></details>` : ""}<div class="revision-columns"><section class="revision-pane"><h3>Student answer</h3><div class="prose student">${recorded(item.student, "No answer recorded yet.")}</div></section><section class="revision-pane teacher-pane"><h3>Reference answer</h3><div class="prose">${e(item.reference)}</div></section></div>${assessmentComparison(item)}${assessmentDimensions(item)}${item.grade ? `<div class="assessment-feedback"><strong>${percent(item.grade.score)} · ${e(item.grade.needs_practice ? "Needs practice" : item.grade.verdict)}</strong><p class="prose">${e(item.grade.feedback)}</p></div>` : ""}<details class="context-details" data-detail="rubric-${e(item.id)}"><summary>Scoring criteria & sources</summary><ul>${(item.rubric || []).map(rule => `<li>${e(rule)}</li>`).join("")}</ul>${item.source_url ? `<a href="${safeURL(item.source_url)}" target="_blank" rel="noopener noreferrer">${e(item.source_title)}</a>` : '<p class="quiet">Teacher-authored assessment</p>'}</details></div></details>`;
}

export function iterationView(s, state = {}) {
  const round = s.iterations?.[state.selectedRound] || (s.round?.id === state.selectedRound ? s.round : null);
  const rounds = !round || s.rounds.some(r => r.id === round.id) ? s.rounds : [...s.rounds, round];
  const navigation = `<div class="iteration-toolbar"><button class="text-button" data-view="overview">← All iterations</button><label class="sr-only" for="round-picker">Select iteration</label><select id="round-picker" class="round-picker">${rounds.map(r => `<option value="${e(r.id)}" ${r.id === state.selectedRound ? "selected" : ""}>Iteration ${r.number} · ${e(r.status)}</option>`).join("")}</select></div>`;
  if (!round) return navigation + empty("Loading iteration…");
  const lessons = round.lessons || [], items = round.evaluations || [], metrics = round.metrics || [], last = metrics.at(-1);
  const local = { ...s, round };
  return `${navigation}<div class="iteration-heading"><div><h1>Iteration ${number(round.number)}</h1><p class="quiet">${e(dateLabel(round.created_at))} · ${count(lessons.length, "lesson")} · ${count(metrics.length, "recorded update")}${last ? ` · last loss ${number(last.loss, 3)}` : ""}</p></div>${badge(round.status)}</div><div class="section-heading"><h2>Lessons & revisions</h2><button class="button secondary small-button" data-diff="${!state.diff}" aria-pressed="${!!state.diff}">${state.diff ? "Hide changes" : "Highlight changes"}</button></div>${lessons.length ? lessons.map((row, i) => lessonCard(row, i, state.diff, state.lessonId)).join("") : empty("No lessons recorded in this iteration.")}${materialList(round)}<section class="iteration-assessment" tabindex="-1"><div class="section-heading"><h2>Assessment</h2><span class="quiet">${items.filter(item => item.grade).length} / ${items.length} reviewed${Number.isFinite(round.score) ? ` · ${number(round.score * 100, 1)}%` : ""}</span></div>${items.length ? items.map(assessmentCard).join("") : empty("No assessment recorded in this iteration.")}</section>${experimentCard(round.experiment, round.learning_work)}${teachingPlan(local)}${learningWork(round.learning_work)}<details class="panel detail-panel" data-detail="training-details"><summary>CoAPT Mid-training · recipe & details</summary><div class="detail-body">${round.curriculum?.train_epochs === 0 ? '<p>No weight updates requested for this iteration.</p>' : ""}${tokenLedger(local)}<dl class="training-details"><div><dt>Learning rate</dt><dd>${last?.learning_rate?.toExponential(1) || "—"}</dd></div><div><dt>Elapsed training</dt><dd>${duration(last?.elapsed_seconds)}</dd></div><div><dt>Parent checkpoint</dt><dd><code>${e(round.model_before)}</code></dd></div><div><dt>Saved checkpoint</dt><dd><code>${e(round.checkpoint || "Not saved yet")}</code></dd></div>${round.checkpoint_retention ? `<div><dt>Checkpoint availability</dt><dd>${e({keep:"Full training state",weights:"Inference weights only; optimizer retired",summary:"Records only; model bytes retired"}[round.checkpoint_retention.disposition])}</dd></div>` : ""}</dl></div></details>`;
}

export function historyView(campaigns, options = {}) {
  const query = (options.query || "").trim().toLowerCase(), filter = options.filter || "all";
  const names = new Map(campaigns.map(c => [c.id, runLabel(c)]));
  const rows = campaigns.filter(c => (filter === "all" || (filter === "archived" ? c.retention === "archive" : c.retention !== "archive")) && (!query || [c.id, runLabel(c), c.config?.focus, c.config?.student_model, c.parent_campaign_id, names.get(c.parent_campaign_id)].some(v => String(v || "").toLowerCase().includes(query))));
  const archived = campaigns.filter(c => c.retention === "archive").length;
  return `<div class="workspace-heading"><div><h1>Previous runs</h1><p>Browse continuations, teaching history and archived work.</p></div><span class="scope-label">${count(campaigns.length, "run")} · ${number(archived)} archived</span></div><div class="history-filters"><label for="history-search">Find a run<input id="history-search" type="search" value="${e(options.query || "")}" placeholder="Name, model, focus or run ID"></label><label for="history-filter">Show<select id="history-filter"><option value="all" ${filter === "all" ? "selected" : ""}>All runs</option><option value="visible" ${filter === "visible" ? "selected" : ""}>Kept in view</option><option value="archived" ${filter === "archived" ? "selected" : ""}>Archived runs</option></select></label><span class="quiet" role="status">${count(rows.length, "match")}</span></div><div class="panel history-register"><div class="history-columns" aria-hidden="true"><span>Run & lineage</span><span>Status</span><span>Iterations</span><span></span></div>${rows.length ? rows.map(c => `<article class="history-row"><div class="history-identity"><button class="history-run-title" data-campaign="${e(c.id)}">${e(runLabel(c))}</button><p><time datetime="${e(c.created_at)}">${e(dateLabel(c.created_at))}</time> · <span title="${e(c.id)}">${e(shortId(c.id))}</span>${c.retention === "archive" ? ' · <span class="archive-label">Archived</span>' : ""}</p>${c.parent_campaign_id ? `<p class="lineage-parent">↳ Continues ${names.has(c.parent_campaign_id) ? `<button class="text-button" data-campaign="${e(c.parent_campaign_id)}">${e(names.get(c.parent_campaign_id))}</button>` : `<span>${e(shortId(c.parent_campaign_id))}</span>`}</p>` : '<p class="lineage-parent">Starting run</p>'}<p class="history-focus">${e(c.config?.focus)} · ${e(modelLabel(c.config?.student_model || ""))}</p>${c.retention_reason ? `<details class="retention-note" data-detail="retention-${e(c.id)}"><summary>Retention decision</summary><p>${e(c.retention_reason)}</p></details>` : ""}</div><div class="history-state">${badge(c.status, true)}</div><div class="history-iterations"><strong>${number(c.completed_rounds)}</strong><span>completed</span></div><button class="text-button history-inspect" data-campaign="${e(c.id)}" aria-label="Inspect ${e(runLabel(c))}">Inspect ↗</button></article>`).join("") : empty("No runs match these filters.")}</div><p class="history-footnote">Archived runs retain their teaching records and lineage. Checkpoint availability follows the recorded storage decisions. <button class="text-button" data-view="reports">Read operational reports →</button></p>`;
}

export function emptyStudio() {
  return `<div class="empty-workspace"><h1>No runs yet</h1><button class="button primary" id="empty-create">Create a run</button></div>`;
}

export function reportsView(data, detail, selectedId, before) {
  if (!data) return empty("Loading reports…");
  const current = data.current || {}, recovery = detail?.recovery;
  const titles = { history_review: "Run history & logs", failure: "Execution recovery", status_review: "Status review", quota: "Teacher availability", rate_limit: "Teacher availability", budget: "Execution allowance" };
  const outcome = recovery ? {
    pending: "Orchestrator review is queued.", running: "The orchestrator is investigating. Its decision is still being prepared.",
    decided: "Decision recorded. Automatic application is pending.", waiting: recovery.retry_at ? "Training is waiting; the next automatic review is scheduled below." : "Training is waiting. No retry time was recorded for this incident.",
    resolved: recovery.continuation_id ? "Recovery resolved. A continuation run was created." : "Recovery resolved. See the recorded execution activity below.",
    cancelled: "Recovery was cancelled. This report's proposals are no longer pending.",
  }[recovery.status] || recovery.status : "";
  const decision = recovery?.decision;
  const report = decision?.report;
  const history = detail?.history;
  const checks = detail?.turns.filter(t => t.checks) || [];
  const paragraphs = report?.split(/\n\s*\n/) || [];
  const content = recovery ? `<article class="panel report-document">
    <div class="panel-heading"><div><div class="eyebrow">${before || selectedId !== data.items[0]?.id ? "Earlier report" : "Latest report"} · ${e(recovery.id)}</div><h2>${e(titles[recovery.kind] || recovery.kind)}</h2><p class="quiet">${e(dateLabel(recovery.created_at))} · updated ${e(dateLabel(recovery.updated_at))}</p></div>${badge(recovery.status)}</div>
    <div class="detail-body"><div class="report-decision"><span class="eyebrow">${decision ? `Decision · ${e(decision.action)}` : ["pending", "running"].includes(recovery.status) ? "Investigation in progress" : "Recorded incident"}</span><p>${e(decision?.reason || outcome)}</p>${!decision && recovery.error ? `<details class="context-details" data-detail="incident-details"><summary>Incident details</summary><div class="prose">${e(recovery.error)}</div></details>` : ""}</div><div class="report-outcome"><strong>Recorded outcome</strong><p>${e(outcome)}</p></div>${recovery.retry_at ? `<p>Next automatic review: <strong>${e(dateLabel(recovery.retry_at))}</strong></p>` : ""}${recovery.status === "waiting" && !recovery.retry_at ? '<p class="quiet">No retry time is recorded. See the current system status above.</p>' : ""}
    ${report ? `<section class="report-investigation"><h3>Investigation</h3><p class="prose report-prose">${e(paragraphs[0])}</p>${paragraphs.length > 1 ? `<details class="context-details" data-detail="full-investigation"><summary>Full investigation & evidence · ${paragraphs.length - 1} more sections</summary><div class="prose report-prose">${e(paragraphs.slice(1).join("\n\n"))}</div></details>` : ""}</section>` : '<p class="quiet">No final agent narrative was recorded for this incident. Available observations appear below.</p>'}
    ${checks.length ? `<h3>Host validation</h3>${checks.map(t => `<p>Attempt ${t.attempt}, turn ${t.turn}: <strong>${e(t.checks.status)}</strong> · ${e(dateLabel(t.checks.finished_at))}${t.checks.error ? `<br>${e(t.checks.error)}` : ""}</p>`).join("")}` : ""}
    ${history || recovery.continuation_id ? `<details class="context-details" data-detail="report-applied-operations"><summary>Applied run & storage decisions</summary>
    ${history ? `<p>History review applied ${e(dateLabel(history.applied_at))}: ${count(history.result.run_retention.length, "run")} reviewed; ${count(history.result.log_removals.length, "log")} moved to recoverable trash.</p>` : ""}
    ${history?.result.checkpoint_retention?.length ? `<p>Checkpoint storage: ${history.result.checkpoint_retention.length} decisions applied · ${number(history.result.checkpoint_retention.reduce((sum, x) => sum + x.bytes_released, 0) / 1024 ** 3, 1)} GiB released.</p><details class="context-details" data-detail="checkpoint-retention"><summary>Retained findings and checkpoint availability</summary>${history.result.checkpoint_retention.map(x => `<h3>${e({keep:"Full training state retained",weights:"Inference weights retained",summary:"Experiment records retained"}[x.disposition])}</h3><p><code>${e(x.path)}</code></p><p>${e(x.summary)}</p><p class="quiet">${e(x.reason)}</p>`).join("")}</details>` : ""}
    ${recovery.continuation_id ? `<button class="text-button" data-campaign="${e(recovery.continuation_id)}">Open continuation ↗</button>` : ""}
    </details>` : ""}
    <details class="context-details" data-detail="report-turns"><summary>All decision turns · ${detail.turns.length}</summary>${detail.turns.map(t => `<h3>Attempt ${t.attempt} · turn ${t.turn} · ${e(t.decision?.action || "Observation")}</h3><p>${e(t.decision?.reason || t.decision_error || "")}</p><div class="prose report-prose">${e(t.decision?.report || "No narrative recorded.")}</div>${t.checks_error ? `<p>${e(t.checks_error)}</p>` : ""}`).join("")}</details>
    <details class="context-details" data-detail="report-events"><summary>Recorded execution activity · ${detail.events.length}</summary>${detail.events.map(event => `<p><time>${e(dateLabel(event.created_at))}</time><br>${e(event.message)}</p>`).join("")}</details>
    </div><div class="panel-footer"><span>Decisions and observations are retained.</span><button class="text-button" data-campaign="${e(recovery.campaign_id)}">Open run ↗</button></div></article>` : empty(selectedId ? "Loading report…" : "No orchestrator reports yet. Current execution status is shown above.");
  const currentRun = current.campaign;
  const phase = phaseFor(currentRun?.status, current.round?.stage);
  const status = `<section class="panel operations-now"><div class="operations-main"><div class="operation-title"><span class="status-emblem phase-${phase.tone}" aria-hidden="true">${statusIcon(currentRun?.status)}</span><div><div class="eyebrow">Current execution</div><h2>${currentRun ? e(phaseLabel(currentRun, current.round)) : "No active run"}</h2></div></div><div class="operation-meta">${currentRun ? badge(currentRun.status, true) : ""}${current.round ? `<span>Iteration ${number(current.round.number)}</span>` : ""}${currentRun ? `<code title="${e(currentRun.id)}">${e(shortId(currentRun.id))}</code>` : ""}</div>${currentRun ? `<button class="button secondary" data-campaign="${e(currentRun.id)}">Open Studio <span aria-hidden="true">↗</span></button>` : ""}</div><div class="operations-next"><div class="next-action-label"><span class="next-action-icon" aria-hidden="true">→</span><h3>Next action</h3></div><p class="next-action-text">${e(current.next_action || "No next action recorded.")}</p><div class="review-schedule">${current.recovery?.retry_at ? `<span>Scheduled for ${e(dateLabel(current.recovery.retry_at))}</span>` : current.review_in_rounds > 0 ? `<span>Routine review in ${count(current.review_in_rounds, "completed iteration")}</span>` : !current.recovery && current.review_in_rounds === 0 ? "<span>Routine review due at the next completed-round boundary</span>" : ""}</div><details class="context-details" data-detail="current-status-evidence"><summary>Execution details</summary><p>${e(current.summary)}</p></details></div></section>`;
  return `<div class="workspace-heading report-heading"><div><h1>Report</h1><p>Execution & orchestrator decisions</p></div><span class="observation-chip"><i aria-hidden="true"></i>Live <span>· ${e(time(current.observed_at))}</span></span></div>${status}<div class="reports-layout">${content}<aside class="report-index" aria-label="Report history"><h2>Report history</h2><p class="quiet">Dated decisions across runs</p><div class="report-index-list" data-scroll="reports-${before || "latest"}">${data.items.map(r => `<button class="report-link ${r.id === selectedId ? "selected" : ""}" data-report="${r.id}" ${r.id === selectedId ? 'aria-current="true"' : ""}><strong>${e(titles[r.kind] || r.kind)} <span>#${r.id}</span></strong><time>${e(dateLabel(r.created_at))}</time><span>${e(r.reason)}</span>${badge(r.status, true)}</button>`).join("")}</div><div class="report-pages">${before ? '<button class="text-button" data-report-page="latest">Latest reports</button>' : ""}${data.next_before ? `<button class="text-button" data-report-page="${data.next_before}">Older reports →</button>` : ""}</div></aside></div>`;
}
