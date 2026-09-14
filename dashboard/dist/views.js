import { escapeHTML as e, number, duration, time, percent, safeURL, diffWords, lossChart, iterationMetrics, dateLabel, modelLabel } from "./lib.js";

export const badge = (status, small = false) => `<span class="status ${e(status)} ${small ? "small" : ""}">${e(status || "pending")}</span>`;
const empty = (text) => `<div class="empty-inline">${e(text)}</div>`;
const count = (n, noun) => `${number(n)} ${n === 1 ? noun : noun === "pass" ? "passes" : noun + "s"}`;
const recorded = (text, fallback) => text ? e(text) : `<span class="muted">${e(fallback)}</span>`;
const detailsFor = (s, r) => s.iterations?.[r.id] || (s.round?.id === r.id ? s.round : null);

export function recoveryNotice(s) {
  if (!["waiting", "recovering"].includes(s.campaign.status)) return "";
  const recovery = s.recovery;
  const title = s.campaign.status === "recovering" ? "Orchestrator is handling an interruption" : "Waiting for the teacher";
  const next = recovery?.retry_at ? `Next automatic attempt: ${new Date(recovery.retry_at).toLocaleString("en-GB")}. You can resume now.` : "Progress is saved. Resume when ready.";
  return `<div class="notice" role="status"><strong>${title}</strong><p>${e(recovery?.error || s.campaign.error || "Progress is saved.")}</p><span>${e(next)}</span></div>`;
}

export function tokenLedger(s) {
  const ledger = s.round?.token_ledger;
  if (!ledger) return "";
  const consumed = s.round.metrics?.at(-1)?.stream_tokens || {};
  const labels = { teacher: "Teacher lessons", corpus: "Source text", replay: "Review material" };
  return `<div class="table-scroll"><table><caption>Teaching mix · training target tokens</caption><thead><tr><th>Material</th><th>Prepared</th><th>Share</th><th>Trained</th></tr></thead><tbody>${Object.entries(ledger.streams).map(([key, value]) => `<tr><td>${e(labels[key] || key)}</td><td>${number(value.prepared_tokens)}</td><td>${percent(ledger.total_tokens ? value.prepared_tokens / ledger.total_tokens : 0)}</td><td>${number(consumed[key])}</td></tr>`).join("")}</tbody></table></div>${ledger.missing_streams?.length ? '<p class="quiet">Unavailable material shares were redistributed across available streams.</p>' : ""}`;
}

export function teachingPlan(s) {
  const plan = s.round?.curriculum, strategy = s.round?.teaching_strategy;
  if (!plan && !strategy) return "";
  return `<details class="panel detail-panel" data-detail="teacher-plan"><summary>Teacher plan & next steps</summary><div class="detail-body">${plan ? `<p class="quiet">${count(plan.lessons.length, "task")} · ${count(plan.readings.length, "reading")} · ${count(plan.replay.length, "review lesson")} · ${count(plan.train_epochs, "pass")}</p><p class="prose">${e(plan.notes)}</p><h3>Assessment plan</h3><p class="prose">${e(plan.evaluation_instructions)}</p>` : ""}${strategy ? `<h3>Student notes</h3><p class="prose">${e(strategy.student_notes)}</p><h3>Next iteration · ${e(strategy.action)}</h3><p class="prose">${e(strategy.next_round_instructions)}</p><p class="quiet">${e(strategy.reason)}</p>` : ""}</div></details>`;
}

function lessonSources(row) {
  const sources = row.sources?.length ? row.sources : row.document?.id ? [row.document] : [];
  return `<details class="source-details" data-detail="sources-${e(row.id)}"><summary>Sources & exact training text</summary><div class="detail-body">${sources.length ? sources.map(d => `<a href="${safeURL(d.url)}" target="_blank" rel="noopener noreferrer">${e(d.title)}</a><blockquote>${e(d.text)}</blockquote><p class="quiet">${e(d.license)} · ${e(d.topic)}${d.source_sha256 ? ` · <code title="${e(d.source_sha256)}">${e(d.source_sha256.slice(0, 12))}</code>` : ""}</p>`).join("") : '<p class="quiet">Teacher-authored material · no corpus source declared.</p>'}${row.evidence?.length ? `<h3>Teacher evidence</h3>${row.evidence.map(x => `<blockquote>${e(x)}</blockquote>`).join("")}` : ""}${row.training_text ? `<h3>Exact training text</h3><div class="prose">${e(row.training_text)}</div>` : ""}</div></details>`;
}

export function eventsView(events, expanded = false) {
  if (!events?.length) return empty("No activity recorded yet.");
  return `<ul class="activity-feed">${events.slice(0, expanded ? 100 : 6).map(item => `<li class="activity-item"><span class="event-dot ${e(item.kind)}" aria-hidden="true"></span><div>${item.round_id ? `<button class="event-link" data-round="${e(item.round_id)}">${e(item.message)}</button>` : e(item.message)}</div><time datetime="${e(item.created_at)}" title="${e(dateLabel(item.created_at))}">${time(item.created_at)}</time></li>`).join("")}</ul>`;
}

function chartPanel(s) {
  const metrics = iterationMetrics(s.rounds.map(r => detailsFor(s, r) || r));
  const last = metrics.at(-1);
  return `<section class="panel loss-panel"><div class="panel-heading"><h2>Training loss</h2>${last ? `<span class="last-loss">${number(last.loss, 3)}<small>latest</small></span>` : ""}</div><div class="chart">${lossChart(metrics)}</div><div class="panel-footer"><span>${count(metrics.length, "recorded update")}</span><span>${last ? `Iteration ${last.round_number} · step ${last.round_step}` : "No weight updates recorded"}</span></div></section>`;
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

export function overview(s) {
  return `<div class="overview-grid">${chartPanel(s)}<section class="panel activity-panel"><div class="panel-heading"><h2>Happening in the studio</h2>${s.campaign.status === "running" ? badge("running", true) : ""}</div>${eventsView(s.events)}<div class="panel-footer"><span>${count(s.teacher_usage?.calls || 0, "teacher call")}${s.teacher_usage?.cost_reported ? ` · $${number(s.teacher_usage.cost_usd, 2)}` : ""}</span><button class="text-button" data-expand-activity aria-expanded="${!!s.expandedActivity}">${s.expandedActivity ? "Less activity" : "All activity"}</button></div>${s.expandedActivity ? `<div class="expanded-activity">${eventsView(s.events?.slice(6), true)}</div>` : ""}</section></div><section class="panel teaching-room"><div class="panel-heading"><h2>Inside the teaching room</h2><span class="quiet">${count(s.rounds.length, "iteration")}</span></div>${iterationList(s)}</section>`;
}

function lessonCard(row, index, diff, selected) {
  const changes = diff ? diffWords(row.student, row.teacher) : { before: e(row.student), after: e(row.teacher) };
  return `<details class="panel lesson-card" data-detail="lesson-${e(row.id)}" ${index === 0 || selected === row.id ? "open" : ""}><summary><span class="tag">${e(row.kind?.toUpperCase() || "LESSON")}</span><strong>${e(row.concept || row.prompt || `Lesson ${index + 1}`)}</strong>${row.gate ? `<span class="lesson-choice ${row.gate.passed ? "selected" : "omitted"}">${row.gate.passed ? "Selected for training" : "Omitted"}</span>` : ""}</summary><div class="lesson-body"><div class="prompt-box"><h3>Student prompt</h3><div class="prose">${e(row.student_prompt || row.prompt)}</div></div><div class="revision-columns"><section class="revision-pane"><h3>Student attempt</h3><div class="prose student">${changes.before || '<span class="muted">No student attempt recorded yet.</span>'}</div></section><section class="revision-pane teacher-pane"><h3>Teacher revision</h3><div class="prose">${changes.after || '<span class="muted">No teacher revision recorded yet.</span>'}</div></section></div>${row.errors?.length ? `<div class="teacher-feedback"><h3>Teacher feedback</h3><ul>${row.errors.map(error => `<li>${e(error)}</li>`).join("")}</ul></div>` : ""}${row.gate?.reason ? `<p class="inclusion-reason">${e(row.gate.reason)}</p>` : ""}${lessonSources(row)}</div></details>`;
}

function assessmentCard(item, index) {
  return `<details class="panel assessment-card" data-detail="assessment-${e(item.id)}" ${index === 0 ? "open" : ""}><summary><span class="question-number">${index + 1}</span><strong>${e(item.question)}</strong>${item.grade ? badge(item.grade.verdict, true) : '<span class="quiet">Pending</span>'}</summary><div class="detail-body">${item.student_prompt && item.student_prompt !== item.question ? `<details class="context-details" data-detail="question-context-${e(item.id)}"><summary>Full student prompt</summary><p class="prose">${e(item.student_prompt)}</p></details>` : ""}<div class="revision-columns"><section class="revision-pane"><h3>Student answer</h3><div class="prose student">${recorded(item.student, "No answer recorded yet.")}</div></section><section class="revision-pane teacher-pane"><h3>Reference answer</h3><div class="prose">${e(item.reference)}</div></section></div>${item.grade ? `<div class="assessment-feedback"><strong>${percent(item.grade.score)} · ${e(item.grade.needs_practice ? "Needs practice" : item.grade.verdict)}</strong><p class="prose">${e(item.grade.feedback)}</p></div>` : ""}<details class="context-details" data-detail="rubric-${e(item.id)}"><summary>Scoring criteria & sources</summary><ul>${(item.rubric || []).map(rule => `<li>${e(rule)}</li>`).join("")}</ul>${item.source_url ? `<a href="${safeURL(item.source_url)}" target="_blank" rel="noopener noreferrer">${e(item.source_title)}</a>` : '<p class="quiet">Teacher-authored assessment</p>'}</details></div></details>`;
}

export function iterationView(s, state = {}) {
  const round = s.iterations?.[state.selectedRound] || (s.round?.id === state.selectedRound ? s.round : null);
  const navigation = `<div class="iteration-toolbar"><button class="text-button" data-view="overview">← All iterations</button><label class="sr-only" for="round-picker">Select iteration</label><select id="round-picker" class="round-picker">${s.rounds.map(r => `<option value="${e(r.id)}" ${r.id === state.selectedRound ? "selected" : ""}>Iteration ${r.number} · ${e(r.status)}</option>`).join("")}</select></div>`;
  if (!round) return navigation + empty("Loading iteration…");
  const lessons = round.lessons || [], items = round.evaluations || [], metrics = round.metrics || [], last = metrics.at(-1);
  const local = { ...s, round };
  return `${navigation}<div class="iteration-heading"><div><h1>Iteration ${number(round.number)}</h1><p class="quiet">${e(dateLabel(round.created_at))} · ${count(lessons.length, "lesson")} · ${count(metrics.length, "recorded update")}${last ? ` · last loss ${number(last.loss, 3)}` : ""}</p></div>${badge(round.status)}</div><div class="section-heading"><h2>Lessons & revisions</h2><button class="button secondary small-button" data-diff="${!state.diff}" aria-pressed="${!!state.diff}">${state.diff ? "Hide changes" : "Highlight changes"}</button></div>${lessons.length ? lessons.map((row, i) => lessonCard(row, i, state.diff, state.lessonId)).join("") : empty("No lessons recorded in this iteration.")}<section class="iteration-assessment"><div class="section-heading"><h2>Assessment</h2><span class="quiet">${items.filter(item => item.grade).length} / ${items.length} reviewed${Number.isFinite(round.score) ? ` · ${percent(round.score)}` : ""}</span></div>${items.length ? items.map(assessmentCard).join("") : empty("No assessment recorded in this iteration.")}</section>${teachingPlan(local)}<details class="panel detail-panel" data-detail="training-details"><summary>Training details</summary><div class="detail-body">${round.curriculum?.train_epochs === 0 ? '<p>No weight updates requested for this iteration.</p>' : ""}${tokenLedger(local)}<dl class="training-details"><div><dt>Learning rate</dt><dd>${last?.learning_rate?.toExponential(1) || "—"}</dd></div><div><dt>Elapsed training</dt><dd>${duration(last?.elapsed_seconds)}</dd></div><div><dt>Parent checkpoint</dt><dd><code>${e(round.model_before)}</code></dd></div><div><dt>Saved checkpoint</dt><dd><code>${e(round.checkpoint || "Not saved yet")}</code></dd></div></dl></div></details>`;
}

export function historyView(campaigns, snapshots = {}) {
  return `<div class="history-heading"><h1>Previous runs</h1><span class="quiet">${count(campaigns.length, "run")}</span></div>${campaigns.length ? `<div class="run-history">${campaigns.map(c => {
    const snapshot = snapshots[c.id], round = snapshot?.round, metrics = round?.metrics || [], items = round?.evaluations || [];
    const topics = [...new Set((round?.lessons || []).map(l => l.concept).filter(Boolean))].slice(0, 3);
    const latestLoss = metrics.at(-1)?.loss;
    return `<button class="panel run-card" data-campaign="${e(c.id)}"><span class="run-card-heading"><strong>${e(c.name)}</strong>${badge(c.status, true)}</span><span class="run-date">${e(dateLabel(c.created_at))}${c.parent_campaign_id ? " · Continued run" : ""}</span><span class="run-focus">${e(c.config.focus)}</span><span class="run-outcomes"><span><small>Completed iterations</small><strong>${number(c.completed_rounds)}</strong></span><span><small>Latest iteration loss</small><strong>${number(latestLoss, 3)}</strong></span><span><small>Questions assessed · latest</small><strong>${snapshot ? number(items.filter(i => i.grade).length) : "…"}</strong></span></span>${topics.length ? `<span class="run-topics">${topics.map(topic => `<span>${e(topic)}</span>`).join("")}</span>` : `<span class="quiet">${snapshot ? (round ? "No lessons recorded in the latest iteration" : "No iterations started") : "Loading recorded results…"}</span>`}<span class="run-card-footer"><span>${e(c.config.teacher_model)} · ${e(modelLabel(c.config.student_model))}</span><span>Open run ↗</span></span></button>`;
  }).join("")}</div>` : empty("No saved runs yet.")}`;
}

export function emptyStudio() {
  return `<div class="empty-workspace"><h1>No runs yet</h1><button class="button primary" id="empty-create">Create a run</button></div>`;
}
