import { escapeHTML as e, number, dateLabel } from "./lib.js?v=05be6631cf7e";
import { scoreTrend, trendLegend } from "./score-trend.js?v=05be6631cf7e";

// Read-only display of Bench's aggregate projection. Observations only: nothing here is
// treated as an acceptance decision. A fitted display trend leaves observed values unchanged.
const finite = x => Number.isFinite(x);
const pct = x => finite(x) ? `${number(x * 100, 1)}%` : "—";
const pp = x => finite(x) ? `${x > 0 ? "+" : ""}${number(x * 100, 2)} pp` : "—";
const count = x => finite(x) ? number(x) : "—";
const fraction = (correct, n) => finite(correct) && finite(n) && n > 0 ? `${number(correct)} / ${number(n)}` : null;
const short = id => typeof id === "string" ? id.slice(0, 12) : "";
const labels = { not_ready: "Evaluation data unavailable", unavailable: "Evaluation unavailable", no_baseline: "Waiting for baseline", pending: "Queued", running: "Evaluating", ok: "Latest snapshot evaluated", stale: "Earlier snapshot", incompatible: "Incompatible evaluation", failed_infra: "Evaluation failed" };

// Practical metric names. Rates are fractions of the evaluated question set.
export const METRICS = {
  score: { label: "Overall", axis: "share of questions answered correctly", value: p => p.score },
  numerical: { label: "Numerical", axis: "share of numerical questions answered correctly", value: p => p.numerical },
  choice: { label: "Choice", axis: "share of choice questions answered correctly", value: p => p.choice },
  invalid_rate: { label: "Invalid answers", axis: "share of questions with an invalid answer", value: p => finite(p.invalid) && finite(p.n) && p.n > 0 ? p.invalid / p.n : null },
  budget_rate: { label: "Output limit", axis: "share of questions that exhausted the output budget", value: p => finite(p.budget_exhausted) && finite(p.n) && p.n > 0 ? p.budget_exhausted / p.n : null },
};
const metricOf = name => Object.hasOwn(METRICS, name) ? name : "score";
// Difference intervals only. A zero-width interval is stated as such rather than shown as a bare pair.
const intervalText = ci => !Array.isArray(ci) || !finite(ci[0]) || !finite(ci[1]) ? "no interval" : ci[0] === ci[1] ? `${pp(ci[0])} (zero-width interval)` : `${pp(ci[0])} to ${pp(ci[1])}`;
const kindLabel = p => p.observation_kind === "baseline" || (!p.observation_kind && p.round_number === 0) ? "Starting model" : p.observation_kind === "milestone" ? "Declared milestone" : "Sample";
const pointName = p => p?.round_number ? `Iteration ${number(p.round_number)}` : "Starting model";

export function scoreChart(points, metric = "score", options = {}) {
  const m = METRICS[metricOf(metric)];
  const valid = (points || []).filter(p => p && finite(m.value(p)) && finite(p.retained_tokens));
  if (!valid.length) return `<div class="token-chart-empty">${options.emptyText ? e(options.emptyText) : metric === "score" || !metric ? "No evaluation results available for this run." : `No ${e(m.label.toLowerCase())} measurements in the displayed observations.`}</div>`;
  const w = 560, h = 200, left = 48, right = 22, top = 16, bottom = 38;
  const max = Math.max(1, ...valid.map(p => p.retained_tokens));
  const x = tokens => left + tokens / max * (w - left - right);
  const y = value => top + (1 - value) * (h - top - bottom);
  const grid = [0, .5, 1].map(r => `<line x1="${left}" x2="${w-right}" y1="${y(r)}" y2="${y(r)}"/><text x="${left-8}" y="${y(r)+4}" text-anchor="end">${r*100}%</text>`).join("");
  // A reference line is only drawn from a real baseline observation of the displayed metric.
  const baseline = options.baseline && finite(m.value(options.baseline)) ? m.value(options.baseline) : null;
  const reference = baseline === null ? "" : `<line class="eval-reference" x1="${left}" x2="${w-right}" y1="${y(baseline)}" y2="${y(baseline)}"><title>Starting model ${e(m.label.toLowerCase())} ${pct(baseline)}</title></line>`;
  const selected = options.selected || null;
  const trend = scoreTrend(valid.map(p => ({ x: p.retained_tokens, y: m.value(p) })), x, y);
  // The caller scopes observations to one compatible weight lineage and displayed history.
  const dots = valid.map(p => {
    const title = `${pointName(p)} · ${kindLabel(p)} · ${m.label} ${pct(m.value(p))} · ${number(p.retained_tokens)} retained-weight tokens · ${dateLabel(p.evaluated_at)}`;
    const classes = `eval-point${p.observation_kind === "milestone" ? " milestone" : ""}${p.observation_kind === "baseline" ? " baseline" : ""}${selected && p.model_id === selected ? " selected" : ""}`;
    const circle = `<circle class="${classes}" cx="${x(p.retained_tokens)}" cy="${y(m.value(p))}" r="${p.observation_kind === "milestone" ? 6 : 5}"><title>${e(title)}</title></circle>`;
    return options.interactive && typeof p.model_id === "string" ? `<a href="#benchmark-history" role="button" data-benchmark-point="${e(p.model_id)}" aria-label="${e(`Show details for ${title}`)}" aria-pressed="${selected === p.model_id}">${circle}</a>` : circle;
  }).join("");
  return `<svg class="${trend ? "has-score-trend" : ""}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${e(`Independent evaluation observations and smoothed trend by retained-weight training tokens: ${m.axis}`)}">${grid}${reference}${trend}${dots}<text x="${left}" y="${h-10}">0</text><text x="${w-right}" y="${h-10}" text-anchor="end">${number(max)} retained-weight tokens</text></svg>${trend ? trendLegend : ""}`;
}

function freshness(data, p) {
  if (!p?.evaluated_at) return "";
  const generated = Date.parse(data?.generated_at), evaluated = Date.parse(p.evaluated_at);
  if (!finite(generated) || !finite(evaluated) || generated < evaluated) return `Measured ${e(dateLabel(p.evaluated_at))}`;
  const hours = (generated - evaluated) / 3.6e6;
  return `Measured ${e(dateLabel(p.evaluated_at))} · ${hours < 1 ? "under an hour" : hours < 48 ? `${Math.floor(hours)} h` : `${Math.floor(hours / 24)} d`} old at the last update`;
}

export function benchmarkCard(data, browser = {}) {
  const p = data?.latest;
  const label = !data ? "Loading evaluation…" : data?.service_stale ? "Evaluator not reporting" : labels[data?.status] || labels.not_ready;
  const invalid = (data?.issues || []).filter(x => x.status !== "skipped").length;
  const skipped = (data?.issues || []).filter(x => x.status === "skipped").length;
  const history = data?.history && finite(data.history.count) ? data.history : null;
  const baseline = data?.comparisons?.baseline || null;
  const points = data?.points || [];
  const baselinePoint = points.find(q => q?.observation_kind === "baseline") || null;
  const gain = p ? (baseline && finite(baseline.delta) ? baseline.delta : p.delta) : null;
  const gainText = p ? (finite(gain) ? `${pp(gain)} vs matching baseline${finite(baseline?.gained) && finite(baseline?.lost) ? ` · ${number(baseline.gained)} gained, ${number(baseline.lost)} lost` : ""}` : "No matching baseline difference") : "Baseline and trained model use the same protocol";
  const subs = p ? [fraction(p.numerical_correct, p.numerical_n) ? `Numerical ${pct(p.numerical)} (${fraction(p.numerical_correct, p.numerical_n)})` : `Numerical ${pct(p.numerical)}`, fraction(p.choice_correct, p.choice_n) ? `Choice ${pct(p.choice)} (${fraction(p.choice_correct, p.choice_n)})` : `Choice ${pct(p.choice)}`].join(" · ") : "";
  const details = p ? `<details class="eval-details" data-detail="benchmark-details"><summary>Evaluation details</summary><div><p>${subs}</p><p>${count(p.invalid)} invalid answers · ${count(p.budget_exhausted)} output-budget failures</p>${p.delta_ci95 ? `<p>Paired difference 95% interval: ${pp(p.delta_ci95[0])} to ${pp(p.delta_ci95[1])} · ${count(p.clusters)} source/family clusters${p.interval_reliable ? "" : " · limited evidence"}</p>` : ""}<p>Monitoring observations, not a general intelligence score or a training decision.</p><p>${e(p.protocol)} · ${e(p.release_name)} · ${e(short(p.model_id))}</p>${invalid || skipped ? `<p>${invalid} failed/incompatible observations · ${skipped} superseded samples in the latest ${count(data.issues.length)} recorded issues</p>` : ""}</div></details>` : "";
  const historyLine = history ? `<span class="eval-history-line">${count(history.count)} observation${history.count === 1 ? "" : "s"} recorded${points.length && history.count > points.length ? ` · ${count(points.length)} shown` : ""} <button type="button" class="text-button" data-benchmark-open aria-expanded="${browser?.open ? "true" : "false"}" aria-controls="benchmark-history">${browser?.open ? "Full history open" : "Open full history →"}</button></span>` : "";
  return `<section class="panel usage-card eval-usage"><div class="usage-heading"><h2>Independent eval</h2><span class="usage-live"><i></i>${e(label)}</span></div><strong class="usage-total">${pct(p?.score)}</strong><p class="usage-description">Building energy${p ? ` · ${count(p.correct)} / ${count(p.n)} correct · ${e(p.release_name)}` : " · awaiting verified results"}</p><div class="token-chart">${scoreChart(points, "score", { baseline: baselinePoint })}</div><div class="usage-footer"><span>${gainText}</span><span>${p ? `${p.round_number ? `Scored iteration ${number(p.round_number)}` : "Scored starting model"} · ${e(dateLabel(p.evaluated_at))}` : "Evaluation runs independently of training"}</span>${p && data?.generated_at ? `<span>${freshness(data, p)}</span>` : ""}${data?.stale ? `<span>Earlier weights · ${number(data.lag_tokens)} retained-weight tokens behind</span>` : ""}${invalid ? `<span>${invalid} failed/incompatible observation(s) among recent issues</span>` : ""}${historyLine}</div>${details}</section>`;
}

function comparisonBlock(kind, comparison, reason) {
  const title = kind === "baseline" ? "Difference from starting model" : "Difference from declared milestone";
  if (!comparison) {
    const reasons = { no_baseline: "The matching starting-model evaluation is unavailable.",
      no_earlier_milestone: "No declared earlier milestone has been evaluated. The previous observation is not used as a substitute.",
      same_checkpoint: "This observation is the reference checkpoint itself.",
      incompatible: "The recorded evaluation conditions do not support this comparison." };
    const text = reason ? e(Object.hasOwn(reasons, reason) ? reasons[reason] : reason) : kind === "milestone" ? reasons.no_earlier_milestone : "No matching baseline comparison is available.";
    return `<div class="eval-comparison"><h3>${title}</h3><p class="muted">${text}</p></div>`;
  }
  const c = comparison;
  const ciText = intervalText(c.delta_ci95);
  const reference = c.reference_round_id ? `reference iteration ${e(String(c.reference_round_id))}` : `reference ${e(short(c.reference_model_id))}`;
  return `<div class="eval-comparison"><h3>${title}</h3><dl><div><dt>Paired difference</dt><dd>${pp(c.delta)} over ${count(c.n)} paired questions</dd></div><div><dt>95% interval</dt><dd>${ciText}${c.interval_reliable ? "" : " · limited evidence"}</dd></div><div><dt>Gained / lost</dt><dd>${count(c.gained)} gained · ${count(c.lost)} lost</dd></div><div><dt>Validity changes</dt><dd>${count(c.invalid_to_correct)} invalid → correct · ${count(c.correct_to_invalid)} correct → invalid</dd></div><div><dt>Reference</dt><dd>${reference} · ${pct(c.reference_score)} at ${count(c.reference_tokens)} retained-weight tokens · ${count(c.clusters)} clusters</dd></div></dl><p class="muted">Descriptive comparison of paired answers. It is not an acceptance, success or regression verdict.</p></div>`;
}

function pointDetail(p) {
  if (!p) return "";
  const num = fraction(p.numerical_correct, p.numerical_n), cho = fraction(p.choice_correct, p.choice_n);
  const rows = [
    ["Observation", `${pointName(p)} · ${kindLabel(p)}`],
    ["Source", `${p.round_id ? `round ${e(String(p.round_id))} · ` : ""}run ${e(String(p.run_id ?? "—"))}`],
    ["Version", `${e(p.protocol ?? "—")} · ${e(p.release_name ?? "—")}`],
    ["Checkpoint", `${e(short(p.model_id))} · ${count(p.retained_tokens)} retained-weight tokens · saved ${e(dateLabel(p.checkpoint_at))}`],
    ["Measured", e(dateLabel(p.evaluated_at))],
    ["Overall", `${pct(p.score)} · ${fraction(p.correct, p.n) || "—"} correct`],
    ["Numerical", num ? `${pct(p.numerical)} · ${num}` : `${pct(p.numerical)} · counts not recorded`],
    ["Choice", cho ? `${pct(p.choice)} · ${cho}` : `${pct(p.choice)} · counts not recorded`],
    ["Invalid answers", count(p.invalid)],
    ["Output limit", count(p.budget_exhausted)],
    ["Paired difference vs starting model", p.delta_ci95 ? `${pp(p.delta)} · 95% interval ${intervalText(p.delta_ci95)} · ${count(p.clusters)} clusters${p.interval_reliable ? "" : " · limited evidence"}` : finite(p.delta) ? pp(p.delta) : "not computed"],
    ["Gained / lost vs starting model", finite(p.gained) && finite(p.lost) ? `${number(p.gained)} gained · ${number(p.lost)} lost` : "not recorded"],
  ];
  return `<div class="eval-selected" aria-live="polite"><h3>Selected observation</h3><dl>${rows.map(([k, v]) => `<div><dt>${e(k)}</dt><dd>${v}</dd></div>`).join("")}</dl></div>`;
}

function observationTable(points, metricName) {
  const m = METRICS[metricName];
  if (!points.length) return `<p class="empty-inline">No observations on this page.</p>`;
  const extra = metricName === "score" ? "" : `<th scope="col">${e(m.label)} (shown)</th>`;
  return `<div class="table-scroll"><table class="eval-table"><thead><tr><th scope="col">Observation</th><th scope="col">Kind</th><th scope="col">Retained tokens</th>${extra}<th scope="col">Overall</th><th scope="col">Numerical</th><th scope="col">Choice</th><th scope="col">Invalid</th><th scope="col">Output limit</th><th scope="col">vs starting model</th><th scope="col">Measured</th></tr></thead><tbody>${points.map(p => `<tr><td><button type="button" class="text-button" data-benchmark-point="${e(p.model_id)}" aria-label="${e(`Show details for ${pointName(p)} measured ${dateLabel(p.evaluated_at)}`)}">${pointName(p)}</button></td><td>${kindLabel(p)}</td><td>${count(p.retained_tokens)}</td>${metricName === "score" ? "" : `<td>${pct(m.value(p))}</td>`}<td>${pct(p.score)}</td><td>${pct(p.numerical)}</td><td>${pct(p.choice)}</td><td>${count(p.invalid)}</td><td>${count(p.budget_exhausted)}</td><td>${pp(p.delta)}</td><td>${e(dateLabel(p.evaluated_at))}</td></tr>`).join("")}</tbody></table></div>`;
}

export function benchmarkHistory(data, browser = {}) {
  if (!browser?.open) return "";
  // A schema v1 projection carries no history identity, so a panel left open from another campaign renders nothing.
  const history = data?.history && finite(data.history.count) ? data.history : null;
  if (!history) return "";
  const metricName = metricOf(browser.metric);
  const m = METRICS[metricName];
  const overview = (data?.points || []).filter(Boolean);
  const page = browser.data;
  const pageValid = page && page.schema_version === 2 && (!history || page.history_id === history.id) && Array.isArray(page.points);
  const showingOverview = browser.page === null || browser.page === undefined || !pageValid;
  const shown = showingOverview ? overview : page.points.slice(0, page.page_size || 100);
  const baselinePoint = overview.find(q => q.observation_kind === "baseline") || shown.find(q => q.observation_kind === "baseline") || null;
  const selectedId = typeof browser.selected_model_id === "string" ? browser.selected_model_id : null;
  const selected = selectedId ? shown.find(q => q.model_id === selectedId) || overview.find(q => q.model_id === selectedId) || null : null;
  const heading = `<div class="panel-heading"><div><h2>Independent eval history</h2><p class="quiet">${count(history.count)} recorded observation${history.count === 1 ? "" : "s"} · Same questions and evaluation protocol</p></div><button type="button" class="button secondary small-button" data-benchmark-close aria-label="Close evaluation history">Close</button></div>`;
  const metricButtons = `<div class="eval-metrics" role="group" aria-label="Displayed measurement">${Object.entries(METRICS).map(([key, def]) => `<button type="button" class="workspace-tab${key === metricName ? " active" : ""}" data-benchmark-metric="${key}" aria-pressed="${key === metricName}">${def.label}</button>`).join("")}</div>`;
  const pages = Math.max(1, history.pages || 1);
  const current = showingOverview ? null : page.page;
  const pageControls = `<nav class="eval-pages" aria-label="Observation pages"><button type="button" class="workspace-tab${showingOverview ? " active" : ""}" data-benchmark-overview aria-pressed="${showingOverview}">Overview</button><button type="button" class="workspace-tab" data-benchmark-page="0" ${current === 0 ? "disabled" : ""} aria-label="Oldest page">Oldest</button><button type="button" class="workspace-tab" data-benchmark-page="${current === null ? 0 : Math.max(0, current - 1)}" ${current === null || current === 0 ? "disabled" : ""} aria-label="Previous page">Previous</button><span class="quiet">${showingOverview ? `Overview: ${count(shown.length)} of ${count(history?.count ?? overview.length)} sampled across the whole history by position` : `Page ${number(current + 1)} of ${number(pages)} · oldest first`}</span><button type="button" class="workspace-tab" data-benchmark-page="${current === null ? pages - 1 : Math.min(pages - 1, current + 1)}" ${current === null || current >= pages - 1 ? "disabled" : ""} aria-label="Next page">Next</button><button type="button" class="workspace-tab" data-benchmark-page="${pages - 1}" ${current === pages - 1 ? "disabled" : ""} aria-label="Newest page">Newest</button></nav>`;
  const status = browser.error ? `<p class="form-error" role="alert">${e(browser.error)}</p>` : browser.loading ? `<p class="quiet" aria-live="polite">Loading observation page…</p>` : "";
  const chart = scoreChart(shown, metricName, { baseline: baselinePoint, interactive: true, selected: selectedId, emptyText: showingOverview && !overview.length ? "No evaluation results available in this history." : undefined });
  const scope = `<div class="chart-scope"><span>${e(m.label)} · ${e(m.axis)}</span><span>${showingOverview ? "Dots are sampled observations spanning the whole history; the starting model and declared milestones are always kept" : "Dots are the observations on this page only"}${baselinePoint ? " · dashed line marks the starting model" : ""}</span></div>`;
  const comparisons = `<div class="eval-comparisons">${comparisonBlock("baseline", data?.comparisons?.baseline || null, data?.comparison_reasons?.baseline)}${comparisonBlock("milestone", data?.comparisons?.milestone || null, data?.comparison_reasons?.milestone)}</div>`;
  const update = browser.newerAvailable ? '<div class="eval-history-update"><span>New evaluations are available. This view keeps the history you opened.</span><button type="button" class="text-button" data-benchmark-refresh>Refresh history</button></div>' : "";
  return `<section class="panel eval-history" id="benchmark-history" aria-label="Independent evaluation history">${heading}${update}<div class="eval-history-controls">${metricButtons}${pageControls}</div>${status}<div class="token-chart">${chart}</div>${scope}${pointDetail(selected)}<p class="eval-comparison-scope">Comparisons below describe the latest observation in this history snapshot: ${e(pointName(data.latest))} · ${e(dateLabel(data.latest?.evaluated_at))}.</p>${comparisons}${observationTable(shown, metricName)}<div class="panel-footer"><span>Monitoring observations of building-energy transfer, not a general intelligence score or a training decision.</span><span>Intervals describe paired differences on these tasks. Repeated observations do not establish a causal effect.</span></div></section>`;
}
