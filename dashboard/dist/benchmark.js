import { escapeHTML as e, number, dateLabel } from "./lib.js?v=1f2d40bed9fc";

const pct = x => Number.isFinite(x) ? `${number(x * 100, 1)}%` : "—";
const pp = x => Number.isFinite(x) ? `${x > 0 ? "+" : ""}${number(x * 100, 2)} pp` : "—";
const labels = { not_ready: "Waiting for baseline", unavailable: "Evaluation unavailable", no_baseline: "Waiting for baseline", pending: "Queued", running: "Evaluating", ok: "Latest snapshot evaluated", stale: "Earlier snapshot", incompatible: "Incompatible evaluation", failed_infra: "Evaluation failed" };

export function scoreChart(points) {
  const valid = (points || []).filter(p => Number.isFinite(p.score) && Number.isFinite(p.retained_tokens));
  if (!valid.length) return '<div class="token-chart-empty">No completed independent evaluation yet.</div>';
  const w = 560, h = 200, left = 48, right = 22, top = 16, bottom = 38;
  const max = Math.max(1, ...valid.map(p => p.retained_tokens));
  const x = tokens => left + tokens / max * (w - left - right);
  const y = value => top + (1 - value) * (h - top - bottom);
  const grid = [0, .5, 1].map(r => `<line x1="${left}" x2="${w-right}" y1="${y(r)}" y2="${y(r)}"/><text x="${left-8}" y="${y(r)+4}" text-anchor="end">${r*100}%</text>`).join("");
  // Dots are observations. No interpolated capability, no line across weight branches.
  return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Independent evaluation observations by retained-weight training tokens">${grid}${valid.map(p => `<circle class="eval-point" cx="${x(p.retained_tokens)}" cy="${y(p.score)}" r="5"><title>${e(p.round_number ? `Iteration ${p.round_number}` : "Starting model")} · ${pct(p.score)} · ${number(p.retained_tokens)} retained-weight tokens · ${e(dateLabel(p.evaluated_at))}</title></circle>`).join("")}<text x="${left}" y="${h-10}">0</text><text x="${w-right}" y="${h-10}" text-anchor="end">${number(max)} retained-weight tokens</text></svg>`;
}

export function benchmarkCard(data) {
  const p = data?.latest;
  const label = !data ? "Loading evaluation…" : data?.service_stale ? "Evaluator not reporting" : labels[data?.status] || labels.not_ready;
  const invalid = (data?.issues || []).filter(x => x.status !== "skipped").length;
  const skipped = (data?.issues || []).filter(x => x.status === "skipped").length;
  const details = p ? `<details class="eval-details" data-detail="benchmark-details"><summary>Evaluation details</summary><div><p>Numerical ${pct(p.numerical)} · Choice ${pct(p.choice)}</p><p>${number(p.invalid)} invalid answers · ${number(p.budget_exhausted)} output-budget failures</p>${p.delta_ci95 ? `<p>Paired difference 95% interval: ${pp(p.delta_ci95[0])} to ${pp(p.delta_ci95[1])} · ${number(p.clusters)} source/family clusters${p.interval_reliable ? "" : " · limited evidence"}</p>` : ""}<p>Monitoring observations, not a general intelligence score or a training decision.</p><p>${e(p.protocol)} · ${e(p.release_name)} · ${e(p.model_id.slice(0, 12))}</p>${invalid || skipped ? `<p>${invalid} failed/incompatible observations · ${skipped} superseded samples</p>` : ""}</div></details>` : "";
  return `<section class="panel usage-card eval-usage"><div class="usage-heading"><h2>Independent eval</h2><span class="usage-live"><i></i>${e(label)}</span></div><strong class="usage-total">${pct(p?.score)}</strong><p class="usage-description">Building energy${p ? ` · ${number(p.correct)} / ${number(p.n)} correct · ${e(p.release_name)}` : " · awaiting verified results"}</p><div class="token-chart">${scoreChart(data?.points)}</div><div class="usage-footer"><span>${p ? `${pp(p.delta)} vs matching baseline` : "Baseline and trained model use the same protocol"}</span><span>${p ? `${p.round_number ? `Scored iteration ${number(p.round_number)}` : "Scored starting model"} · ${e(dateLabel(p.evaluated_at))}` : "Training continues while evaluation runs"}</span>${data?.stale ? `<span>Earlier weights · ${number(data.lag_tokens)} retained-weight tokens behind</span>` : ""}${invalid ? `<span>${invalid} failed/incompatible observation(s) retained</span>` : ""}</div>${details}</section>`;
}
