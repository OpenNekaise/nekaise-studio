import { escapeHTML as e, number, dateLabel } from "./lib.js?v=9a2009180d1e";

const pct = value => Number.isFinite(value) ? `${number(value * 100, 1)}%` : "—";
import { SFT_BASELINE_ID } from "./gpqa.js?v=9a2009180d1e";
const isBaseline = row => row?.model_id === SFT_BASELINE_ID;
const model = row => `${isBaseline(row) ? "MiniCPM5-1B-SFT · starting point" : row.model_label}${row.round_number ? ` · iteration ${row.round_number}` : ""} · ${row.model_id?.slice(0, 8) || "pending identity"}`;
const compatible = (a, b) => a.protocol_id === b.protocol_id && a.dataset_sha256 === b.dataset_sha256 &&
  a.max_input_tokens === b.max_input_tokens;

function comparison(results, latest) {
  const references = results.filter(isBaseline);
  const anchor = latest || references[0];
  const baseline = anchor && references.find(row => compatible(row, anchor));
  const points = anchor ? results.filter(row => !isBaseline(row) && compatible(row, anchor)).reverse() : [];
  const excluded = results.filter(row => !isBaseline(row)).length - points.length;
  const note = baseline ? `MiniCPM5-1B-SFT baseline: ${pct(baseline.score)} · ${baseline.correct}/12,032 · 95% source-cluster interval ${pct(baseline.ci95[0])}–${pct(baseline.ci95[1])}` :
    references.length ? "Baseline uses a different protocol; no comparison line is drawn." : "Starting-point baseline awaiting a completed, verified evaluation.";
  if (!anchor) return `<p class="muted gpqa-baseline-note">${e(note)}</p>`;
  const left = 64, right = 674, top = 18, bottom = 206;
  const y = score => bottom - score * (bottom - top);
  const x = i => points.length < 2 ? (left + right) / 2 : left + i * (right - left) / (points.length - 1);
  const grid = [0, .25, .5, .75, 1].map(value => `<line class="gpqa-grid" x1="${left}" x2="${right}" y1="${y(value)}" y2="${y(value)}"/><text x="${left - 9}" y="${y(value) + 4}" text-anchor="end">${value * 100}%</text>`).join("");
  const referenceLine = baseline ? `<line class="gpqa-baseline-line" x1="${left}" x2="${right}" y1="${y(baseline.score)}" y2="${y(baseline.score)}"><title>${e(note)}</title></line>` : "";
  const dots = points.map((row, i) => `<circle class="gpqa-score-point" cx="${x(i)}" cy="${y(row.score)}" r="5"><title>${e(model(row))}: ${pct(row.score)} · ${row.correct}/12,032 · ${e(dateLabel(row.updated_at))}</title></circle>`).join("");
  const labels = points.length ? [0, ...(points.length > 1 ? [points.length - 1] : [])].map(i => `<text x="${x(i)}" y="228" text-anchor="middle">Evaluation ${i + 1}</text>`).join("") : "";
  return `<div class="gpqa-comparison"><svg viewBox="0 0 700 245" role="img" aria-label="MMLU-Pro accuracy from zero to 100 percent. ${e(note)}. ${points.length} compatible model evaluations.">
    ${grid}<line class="gpqa-chance-line" x1="${left}" x2="${right}" y1="${y(anchor.chance_score)}" y2="${y(anchor.chance_score)}"/>${referenceLine}${dots}${labels}</svg>
    <div class="gpqa-legend"><span class="gpqa-legend-model">Evaluated models</span>${baseline ? `<span class="gpqa-legend-baseline">SFT starting point · ${pct(baseline.score)}</span>` : ""}<span class="gpqa-legend-chance">Random choice · ${pct(anchor.chance_score)}</span></div>
    <p class="gpqa-baseline-note">${e(note)}</p>
    ${baseline && latest ? `<p class="gpqa-baseline-note">Difference from starting point: ${latest.score > baseline.score ? "+" : ""}${number((latest.score - baseline.score) * 100, 1)} percentage points · matching protocol.</p>` : ""}
    ${excluded ? `<p class="gpqa-baseline-note">${number(excluded)} result(s) under other protocols are listed below and excluded from this chart.</p>` : ""}</div>`;
}

export function mmluCard(data) {
  const runs = data?.status === "ok" && Array.isArray(data.runs) ? [...data.runs].sort((a, b) =>
    (b.started_at || b.updated_at).localeCompare(a.started_at || a.updated_at)) : [];
  const attempt = runs[0];
  const results = runs.filter(row => row.status === "complete");
  const latest = results.find(row => !isBaseline(row));
  const detail = latest || results[0];
  const status = data?.status === "unavailable" ? "Unavailable" : !attempt ? "Not evaluated" :
    attempt.status === "running" ? (data.stale ? "No recent progress" : `Evaluating${isBaseline(attempt) ? " baseline" : ""} · ${number(attempt.completed)} / 12,032`) :
    attempt.status === "failed" ? "Latest attempt failed" : attempt.status === "interrupted" ? "Latest attempt interrupted" : "Completed";
  const progress = attempt?.status === "running" ? `<p class="muted">${e(model(attempt))} · ${number(attempt.completed)} / 12,032 answered${data.stale ? " · Check the independent evaluator" : ""}</p>` : "";
  const details = detail ? `<details class="eval-details" data-detail="mmlu-details"><summary>Evaluation details &amp; recent results</summary><div>
    <p>95% source-cluster interval: ${pct(detail.ci95[0])}–${pct(detail.ci95[1])} · random-choice reference: ${pct(detail.chance_score)}.</p>
    <p>Zero-shot answer-choice likelihood · one forward pass per question · CPU FP32. All 12,032 test questions; 14 subjects.</p>
    <p>${e(detail.protocol)} · protocol ${e(detail.protocol_id.slice(0, 12))} · dataset ${e(detail.dataset_sha256.slice(0, 12))}</p>
    <p>This measures preference among answer letters. It is different from the published five-shot reasoning protocol; scores are not leaderboard comparisons.</p>
    <p>The interval resamples source clusters. Public benchmark exposure during pretraining is unknown; a change in score alone does not establish a training effect.</p>
    <h3>Subjects · ${e(model(detail))}</h3>
    <div class="table-scroll"><table><thead><tr><th>Subject</th><th>Accuracy</th><th>Correct / total</th></tr></thead><tbody>${detail.subjects.map(row => `<tr><td>${e(row.category)}</td><td>${pct(row.score)}</td><td>${number(row.correct)} / ${number(row.n)}</td></tr>`).join("")}</tbody></table></div>
    <div class="table-scroll"><table><thead><tr><th>Evaluated model</th><th>Score</th><th>Protocol</th><th>Evaluated</th></tr></thead><tbody>${results.map(row => `<tr><td>${e(model(row))}</td><td>${pct(row.score)} · ${row.correct}/12,032</td><td>${e(row.protocol_id.slice(0, 8))}</td><td>${e(dateLabel(row.updated_at))}</td></tr>`).join("")}</tbody></table></div>
    </div></details>` : "";
  return `<section class="panel usage-card eval-usage mmlu-card"><div class="usage-heading"><h2>MMLU-Pro</h2><span class="usage-live">${e(status)}</span></div>
    <strong class="usage-total">${pct(latest?.score)}</strong>
    <p class="usage-description">14 subjects · full test set · weekly${latest ? ` · ${latest.correct} / 12,032 correct` : " · 12,032 questions"}</p>
    <p class="muted">Zero-shot · answer-choice likelihood</p>${progress}${comparison(results, latest)}<div class="usage-footer"><span>${latest ? e(model(latest)) : "No completed model evaluation beyond the starting point yet."}</span>
    <span>${latest ? `${attempt !== latest ? "Last completed result · " : ""}${e(dateLabel(latest.updated_at))}` : attempt?.status === "running" ? "The score appears after all 12,032 answers are verified." : detail ? "Starting point evaluated; awaiting a model result under the same protocol." : attempt ? "Review the saved evaluator diagnostics before retrying." : "Bench runs the latest saved model weekly. The baseline is evaluated once per protocol."}</span>
    <span>Standalone evaluations across models · independent of the selected training run</span>${data?.rejected_entries ? `<span>${number(data.rejected_entries)} saved result(s) unavailable after verification.</span>` : ""}</div>${details}</section>`;
}
