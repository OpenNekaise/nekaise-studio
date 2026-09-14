import {
  escapeHTML as e,
  number,
  duration,
  time,
  relativeTime,
  percent,
  safeURL,
  diffWords,
  lossChart,
} from "./lib.js";

export const badge = (status, small = false) =>
  `<span class="status ${e(status)} ${small ? "small" : ""}"><span class="dot ${status === "running" ? "running-pulse" : ""}"></span>${e(status)}</span>`;
const empty = (text) => `<div class="empty-inline">${text}</div>`;
const activeStage = (s) => s.round?.stage || "select";

export function recoveryNotice(s) {
  if (!["waiting", "recovering"].includes(s.campaign.status)) return "";
  const r = s.recovery;
  const title = s.campaign.status === "recovering" ? "Orchestrator is handling an interruption" : "Learning is waiting";
  const next = r?.retry_at ? `Next automatic attempt: ${new Date(r.retry_at).toLocaleString("en-GB")}. You can resume now.` : "Resume when ready, or pause automatic recovery.";
  return `<div class="notice" role="status"><strong>${title}</strong><p>${e(r?.error || s.campaign.error || "Progress is saved.")}</p><span>${e(next)}</span></div>`;
}

function tokenLedger(s) {
  const ledger = s.round?.token_ledger;
  if (!ledger) return "";
  const consumed = s.round.metrics?.at(-1)?.stream_tokens || {};
  const labels = {teacher: "Teacher lessons", corpus: "Source text", replay: "Review material"};
  return `<section class="panel"><div class="panel-heading"><h2>Teaching mix</h2><span class="quiet">Effective training tokens · includes EOS</span></div><div class="table-scroll"><table><thead><tr><th>MATERIAL</th><th>PREPARED</th><th>SHARE</th><th>TRAINED</th></tr></thead><tbody>${Object.entries(ledger.streams).map(([key, value]) => `<tr><td>${e(labels[key] || key)}</td><td>${number(value.prepared_tokens)}</td><td>${percent(value.prepared_tokens / ledger.total_tokens)}</td><td>${Number.isFinite(consumed[key]) ? number(consumed[key]) : "—"}</td></tr>`).join("")}</tbody></table></div>${ledger.missing_streams?.length ? '<p class="quiet panel-body">Unavailable material shares are distributed across the available streams.</p>' : ""}</section>`;
}

export function teachingPlan(s) {
  const plan = s.round?.curriculum, strategy = s.round?.teaching_strategy;
  if (!plan && !strategy) return "";
  return `<section class="panel"><div class="panel-heading"><h2>Teacher's plan</h2><span class="quiet">Teaching decisions & memory</span></div><div class="panel-body">${plan ? `<p class="quiet">${number(plan.lessons.length)} tasks · ${number(plan.readings.length)} readings · ${number(plan.replay.length)} review lessons · ${number(plan.train_epochs)} passes</p><p class="prose">${e(plan.notes)}</p><details data-detail="teacher-assessment-plan"><summary>Assessment plan</summary><p class="prose">${e(plan.evaluation_instructions)}</p></details>` : ""}${strategy ? `<details data-detail="teacher-memory"><summary>Student notes & next round · ${e(strategy.action)}</summary><h3>STUDENT NOTES</h3><p class="prose">${e(strategy.student_notes)}</p><h3>NEXT ROUND</h3><p class="prose">${e(strategy.next_round_instructions)}</p><p class="quiet">${e(strategy.reason)}</p></details>` : ""}</div></section>`;
}

function lessonSources(row) {
  const sources = row.sources?.length ? row.sources : row.document.id ? [row.document] : [];
  return `<details class="source-details" data-detail="source"><summary>Sources, evidence & training text</summary><div class="source-body">${sources.length ? sources.map(d => `<a href="${safeURL(d.url)}" target="_blank" rel="noopener noreferrer">${e(d.title)} ↗</a><blockquote>${e(d.text)}</blockquote><div class="source-meta"><span>${e(d.license)}</span><span>${e(d.topic)}</span><code title="${e(d.source_sha256)}">SHA ${e(d.source_sha256.slice(0,12))}</code></div>`).join("") : '<p class="quiet">Teacher-authored material · no corpus source declared.</p>'}${row.evidence.length ? `<h3 class="quiet">TEACHER EVIDENCE</h3>${row.evidence.map(x => `<blockquote>${e(x)}</blockquote>`).join("")}` : ""}${row.training_text ? `<h3 class="quiet">EXACT TRAINING TEXT</h3><div class="prose">${e(row.training_text)}</div>` : ""}</div></details>`;
}
const stageGroup = {
  select: 0,
  plan: 0,
  draft: 0,
  revise: 1,
  gate: 1,
  freeze: 1,
  train: 2,
  evaluate: 3,
  answer: 3,
  grade: 3,
  adapt: 3,
};

function pipeline(s) {
  const r = s.round,
    current = stageGroup[activeStage(s)];
  return `<div class="pipeline"><div class="pipeline-title"><span class="round-number">${String(r?.number || 1).padStart(2, "0")}</span><div><small>CURRENT ROUND</small><strong>${e(r ? s.stages.find((x) => x.id === r.stage)?.label || r.status : "Ready to begin")}</strong></div></div>${[
    "Prepare dataset",
    "Teacher revision",
    "Train student",
    "Evaluate & adapt",
  ]
    .map((label, i) => {
      const done = r?.status === "complete" || i < current;
      return `<div class="pipeline-step ${done ? "done" : i === current && r ? "current" : ""}"><span class="step-circle">${done ? "✓" : i + 1}</span><div>${label}<small>${done ? "Completed" : i === current && r ? e(s.campaign.status) : "Up next"}</small></div></div>`;
    })
    .join("")}</div>`;
}

function stats(s) {
  const r = s.round,
    m = r?.metrics || [],
    last = m.at(-1),
    lessons = r?.lessons || [],
    accepted = lessons.filter((x) => x.gate?.passed).length,
    complete = lessons.filter((x) => x.gate).length;
  const cards = [
    [
      "Training loss",
      last ? number(last.loss, 3) : "—",
      "",
      last
        ? `Step ${last.step} of ${last.total_steps}`
        : r?.curriculum?.train_epochs === 0 || r?.token_ledger?.total_tokens === 0 ? "Teacher chose no weight updates" : "Waiting for training",
      "blue",
    ],
    [
      "Lessons prepared",
      number(accepted),
      `/ ${r?.curriculum ? lessons.length : s.campaign.config.lessons_per_round}`,
      complete
        ? `${complete - accepted} excluded · ${s.campaign.config.rounds === -1 ? "continuous learning" : `${s.campaign.config.rounds} rounds`}`
        : "Teacher corrections",
      "positive",
    ],
    [
      "Online evaluation",
      percent(r?.score),
      "",
      r?.evaluations?.some((x) => x.grade)
        ? `${r.evaluations.filter((x) => x.grade).length} teacher-created questions`
        : "Created fresh in each round",
      "",
    ],
    [
      "Training tokens",
      last ? number(last.tokens) : "—",
      "",
      last
        ? `${number(last.tokens_per_second)} tokens / second`
        : "Recorded by the trainer",
      "",
    ],
  ];
  return `<div class="stats-grid">${cards.map(([label, value, unit, note, tone]) => `<article class="stat-card"><div class="stat-label">${label}<span aria-hidden="true">${label === "Training loss" ? "↘" : label === "Online evaluation" ? "◎" : "·"}</span></div><div class="stat-value">${value}<span class="stat-unit">${unit}</span></div><div class="stat-note ${tone}">${note}</div></article>`).join("")}</div>`;
}

function chartPanel(s) {
  const m = s.round?.metrics || [],
    last = m.at(-1);
  return `<section class="panel"><div class="panel-heading"><div><h2>Learning, one step at a time</h2><p class="panel-subtitle">Training loss · current round</p></div><span class="legend"><i class="legend-line"></i>Actual loss</span></div><div class="chart">${lossChart(m)}</div><div class="chart-summary"><span>Learning rate<strong>${last ? last.learning_rate.toExponential(1) : s.campaign.config.learning_rate.toExponential(1)}</strong></span><span>Elapsed<strong>${duration(last?.elapsed_seconds)}</strong></span><span>Peak GPU memory<strong>${last ? `${number(last.gpu_memory_gb, 1)} GB` : "—"}</strong></span></div></section>`;
}

export function eventsView(events, full = false) {
  return events.length
    ? `<ul class="${full ? "timeline-list" : "activity-feed"}">${events
        .slice(0, full ? 100 : 5)
        .map(
          (x) =>
            `<li class="activity-item"><span class="event-dot ${e(x.kind)}"></span><div>${e(x.message)}${x.data?.stage ? `<small>${e(x.data.stage)}${x.data.attempt ? ` · attempt ${x.data.attempt}` : ""}</small>` : ""}</div><time class="event-time" datetime="${e(x.created_at)}" title="${e(x.created_at)}">${time(x.created_at)}</time></li>`,
        )
        .join("")}</ul>`
    : empty("The first event will appear when the campaign starts.");
}

function lessonTable(s) {
  const lessons = s.round?.lessons || [];
  return `<section class="panel">${lessons.length ? `<div class="table-scroll"><table><thead><tr><th>LESSON / CONCEPT</th><th>BRANCH</th><th>SOURCE</th><th>TEACHER LESSON</th><th></th></tr></thead><tbody>${lessons.map((x) => `<tr class="interactive" data-lesson="${e(x.id)}"><td><button class="table-title" data-lesson="${e(x.id)}">${e(x.concept)}</button><div class="table-subtitle">${e(x.document.selection_reason)}</div></td><td><span class="tag ${x.kind === "sft" ? "blue" : ""}">${x.kind.toUpperCase()}</span></td><td><span class="table-subtitle">${e(x.document.topic.replaceAll("_", " "))}</span>${x.document.replay ? ' <span class="tag clay">replay</span>' : ""}</td><td>${x.gate ? badge(x.gate.passed ? (x.gate.mode === "trusted_teacher" ? "accepted" : "verified") : (x.gate.mode === "trusted_teacher" ? "omitted" : "rejected"), true) : `<span class="muted">${x.teacher ? "Teacher deciding" : x.student ? "Awaiting teacher" : "Student pending"}</span>`}</td><td><button class="icon-button" data-lesson="${e(x.id)}" aria-label="Inspect ${e(x.concept)}">↗</button></td></tr>`).join("")}</tbody></table></div>` : empty("Selected passages, student attempts, and teacher revisions will appear here.")}</section>`;
}

export function roundPicker(s) {
  return `<select class="round-picker" id="round-picker" aria-label="Select round"><option value="">Latest round</option>${s.rounds.map((r) => `<option value="${e(r.id)}" ${s.selectedRound === r.id ? "selected" : ""}>Round ${String(r.number).padStart(2, "0")} · ${e(r.status)}</option>`).join("")}</select>`;
}

export function overview(s) {
  return `${s.rounds.length ? `<div class="view-toolbar"><span class="quiet">${e(s.campaign.config.focus)}</span>${roundPicker(s)}</div>` : ""}${pipeline(s)}${stats(s)}${teachingPlan(s)}${tokenLedger(s)}<div class="content-grid">${chartPanel(s)}<section class="panel"><div class="panel-heading"><div><h2>Happening in the studio</h2><p class="panel-subtitle">Updates from the learning loop</p></div>${s.campaign.status === "running" ? '<span class="status small running">Live</span>' : ""}</div>${eventsView(s.events)}<div class="panel-footer"><span>${s.teacher_usage.calls} teacher calls${s.teacher_usage.cost_reported ? ` · $${number(s.teacher_usage.cost_usd, 2)}` : ""}</span><button data-view="activity">All activity ↗</button></div></section></div><div class="section-heading"><div><h2>Inside the teaching room <span class="count">${s.round?.lessons.length || 0} lessons</span></h2><p>A source, a student attempt, and a better explanation.</p></div><button class="text-button" data-view="lessons">Inspect lessons ↗</button></div>${lessonTable(s)}`;
}

export function lessonsView(s, state) {
  const lessons = s.round?.lessons || [],
    row = lessons.find((r) => r.id === state.lessonId) || lessons[0];
  if (!row)
    return empty("Lessons will appear after the first passages are selected.");
  const diffs = state.diff
    ? diffWords(row.student, row.teacher)
    : { before: e(row.student), after: e(row.teacher) };
  return `<div class="view-toolbar"><div class="view-tabs" role="group" aria-label="Revision display"><button data-diff="false" class="${!state.diff ? "active" : ""}">Side by side</button><button data-diff="true" class="${state.diff ? "active" : ""}">Highlight changes</button></div>${roundPicker(s)}</div><div class="teaching-layout"><div class="lesson-list" role="list" aria-label="Lessons">${lessons.map((r) => `<button class="lesson-option ${row.id === r.id ? "selected" : ""}" data-select-lesson="${e(r.id)}" aria-pressed="${row.id === r.id}"><div class="lesson-option-top"><span class="tag ${r.kind === "sft" ? "blue" : ""}">${r.kind.toUpperCase()}</span><span class="quiet">${e(r.id.toUpperCase())}</span></div><span class="lesson-concept">${e(r.concept)}</span><span class="quiet">${r.gate ? (r.gate.passed ? "✓ Teacher lesson accepted" : "Teacher chose to omit") : r.teacher ? "Awaiting preparation" : r.student ? "Awaiting revision" : "Awaiting student"}</span></button>`).join("")}</div><article class="panel"><div class="inspector-heading"><div><div class="eyebrow">${row.kind.toUpperCase()} · ROUND ${s.round.number}</div><h2>${e(row.concept)}</h2><p class="panel-subtitle">${e(row.document.selection_reason)}</p></div>${row.gate ? badge(row.gate.passed ? (row.gate.mode === "trusted_teacher" ? "accepted" : "verified") : (row.gate.mode === "trusted_teacher" ? "omitted" : "rejected")) : ""}</div><div class="prompt-box"><strong>STUDENT PROMPT</strong>${e(row.student_prompt || row.prompt || "The teacher is preparing this lesson.")}</div><div class="inspector-columns"><div class="inspection-pane"><div class="pane-label">STUDENT ATTEMPT <span>${row.generation ? `${row.generation.tokens} tokens` : ""}</span></div><div class="prose student">${diffs.before || '<span class="muted">Waiting for the student…</span>'}</div></div><div class="inspection-pane"><div class="pane-label teacher-label">TEACHER REVISION <span>CoAPT</span></div><div class="prose">${diffs.after || '<span class="muted">The teacher will revise the student’s attempt against the source.</span>'}</div></div></div>${row.errors.length ? `<div class="error-list"><h3>WHAT THE TEACHER NOTICED</h3><ul>${row.errors.map((x) => `<li>${e(x)}</li>`).join("")}</ul></div>` : ""}${row.gate ? `<div class="gate-box ${!row.gate.passed ? "rejected" : ""}"><span>${row.gate.passed ? "✓" : "×"}</span><div><strong>${row.gate.passed ? "Accepted for training" : "Teacher omitted from training"}</strong><p>${e(row.gate.reason)}</p></div></div>` : ""}${lessonSources(row)}</article></div>`;
}

export function evaluationView(s) {
  const items = s.round?.evaluations || [],
    gaps = s.round?.gaps || [];
  return `<div class="view-toolbar"><div><h2>What does the student understand?</h2><p class="panel-subtitle">Round ${s.round?.number || 1} · ${items.length} questions · ${percent(s.round?.score)} rubric score</p></div>${roundPicker(s)}</div><div class="notice">These questions are created online by the teacher to guide the next round. Each round has a different evaluation; scores are not a fixed-benchmark trend.</div><div class="eval-grid"><section>${items.length ? items.map((r, i) => `<details class="eval-item" data-detail="eval-${e(r.id)}" ${i === 0 ? "open" : ""}><summary class="eval-question"><span>${e(r.question)}</span>${r.grade ? badge(r.grade.verdict, true) : '<span class="tag">Awaiting answer</span>'}</summary><div class="eval-content"><span class="tag">${e(r.concept)}</span> ${r.novel_source ? '<span class="tag blue">New source</span>' : ""}<h3>STUDENT ANSWER</h3><p>${e(r.student) || '<span class="muted">Waiting for the student…</span>'}</p><h3>REFERENCE ANSWER</h3><p>${e(r.reference)}</p><h3>SCORING CRITERIA</h3><ul class="quiet">${r.rubric.map((x) => `<li>${e(x)}</li>`).join("")}</ul>${r.grade ? `<div class="eval-feedback"><strong>${percent(r.grade.score)} · ${e(r.grade.gap_type === "none" ? "Understood" : r.grade.gap_type + " gap")}</strong><br>${e(r.grade.feedback)}</div>` : ""}<p class="quiet" style="margin-top:15px">Source: <a href="${safeURL(r.source_url)}" target="_blank" rel="noopener noreferrer">${e(r.source_title)}</a></p></div></details>`).join("") : empty("After training, the teacher will create a fresh diagnostic evaluation.")}</section><aside class="panel" style="align-self:start"><div class="panel-heading"><div><h2>The next learning priorities</h2><p class="panel-subtitle">Concepts to revisit in the next dataset</p></div></div><div class="panel-body">${gaps.length ? gaps.map((g) => `<div class="gap-item"><div class="gap-label"><span>${e(g.concept)}</span><span class="tag clay">${e(g.type)}</span></div><div class="gap-bar"><span style="width:${Math.round(g.priority * 100)}%"></span></div><p>${g.type === "knowledge" ? "Select supporting corpus passages" : "Practice applying the concept in new questions"}</p></div>`).join("") : empty(items.some((x) => x.grade) ? "No outstanding gaps recorded." : "Priorities appear after the teacher assesses the answers.")}</div></aside></div>`;
}

export function activityView(s) {
  return `<div class="section-heading"><div><h2>Every round, accounted for</h2><p>Checkpoint lineage and the steps behind each result.</p></div></div><section class="panel"><div class="table-scroll"><table><thead><tr><th>ROUND</th><th>STATE</th><th>ONLINE SCORE</th><th>PARENT CHECKPOINT</th><th>CREATED</th></tr></thead><tbody>${s.rounds.map((r) => `<tr class="interactive" data-round="${e(r.id)}"><td><button data-round="${e(r.id)}">Round ${String(r.number).padStart(2, "0")} ↗</button></td><td>${badge(r.status, true)}</td><td>${percent(r.score)}</td><td><code title="${e(r.model_before)}">${e(r.model_before.length > 54 ? "…" + r.model_before.slice(-54) : r.model_before)}</code></td><td class="quiet">${time(r.created_at)}</td></tr>`).join("") || '<tr><td colspan="5">No rounds have started.</td></tr>'}</tbody></table></div></section><div class="section-heading"><h2>Activity log</h2></div><section class="panel">${eventsView(s.events, true)}</section>`;
}
