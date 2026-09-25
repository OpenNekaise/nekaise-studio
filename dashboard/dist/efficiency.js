import { escapeHTML as e, number } from "./lib.js?v=ebb687a0fdd7";
import { scoreTrend } from "./score-trend.js?v=ebb687a0fdd7";

export const ratioLabel = value => Number.isFinite(value) && value >= 0
  ? `${new Intl.NumberFormat("en-GB", { maximumSignificantDigits: 3 }).format(value)}×` : "—";

export function efficiencyChart(series) {
  const points = (series || []).filter(p => Number.isFinite(p.ratio) && p.ratio >= 0 && Number.isFinite(p.iteration));
  if (!points.length) return '<div class="token-chart-empty">Waiting for a completed iteration with reported Teacher tokens.</div>';
  const w = 560, h = 200, left = 64, right = 18, top = 16, bottom = 35;
  const first = Math.min(...points.map(p => p.iteration)), last = Math.max(first + 1, ...points.map(p => p.iteration));
  const high = Math.max(.001, ...points.map(p => p.ratio));
  const x = n => left + (n - first) / (last - first) * (w - left - right);
  const y = n => top + (1 - n / high) * (h - top - bottom);
  const grid = [0, .5, 1].map(f => `<line x1="${left}" x2="${w-right}" y1="${y(high*f)}" y2="${y(high*f)}"/><text x="${left-9}" y="${y(high*f)+4}" text-anchor="end">${ratioLabel(high*f)}</text>`).join("");
  // Normalize only for the shared bounded smoother; the ratio itself has no 1× ceiling.
  const trend = scoreTrend(points.map(p => ({ x: p.iteration, y: p.ratio/high })), x, n => y(n*high));
  const dots = points.map(p => `<circle class="efficiency-point" cx="${x(p.iteration)}" cy="${y(p.ratio)}" r="3.5"><title>${e(`Lineage iteration ${p.iteration} · Run ${p.campaign_id} · Iteration ${p.number}: ${ratioLabel(p.ratio)} · ${number(p.trained_tokens)} trained / ${number(p.teacher_tokens)} Teacher tokens`)}</title></circle>`).join("");
  return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Training tokens per Teacher token, by completed lineage iteration">${grid}${dots}${trend}<text x="${left}" y="${h-9}">${number(first)}</text>${points.at(-1).iteration !== first ? `<text x="${w-right}" y="${h-9}" text-anchor="end">${number(points.at(-1).iteration)}</text>` : ""}</svg>${trend ? '<div class="score-trend-legend"><span>Smoothed efficiency</span><span>Per iteration</span></div>' : ""}`;
}

export function efficiencyCard(telemetry) {
  const data = telemetry?.efficiency, usage = data?.teacher_usage;
  const partial = data && data.status !== "complete";
  const label = telemetry?.unavailable ? "Usage unavailable" : !data ? "Loading measurements…" : partial ? "Partial reported usage" : "Measured efficiency";
  const missing = (usage?.missing_calls || 0) + (usage?.pending_calls || 0);
  return `<section class="panel usage-card efficiency-usage"><div class="usage-heading"><h2>Training efficiency</h2><span class="usage-live">${e(label)}</span></div><strong class="usage-total">${ratioLabel(data?.reported_ratio)}</strong><p class="usage-description">Tokens trained / Teacher tokens</p><div class="token-chart">${efficiencyChart(data?.series)}</div><div class="usage-footer"><span>Lineage total: ${number(data?.trained_tokens)} trained / ${number(data?.teacher_tokens)} Teacher tokens</span><span>${missing ? `${number(missing)} calls have pending or missing usage; the displayed total ratio is provisional. ` : ""}${number((data?.series || []).filter(p => Number.isFinite(p.ratio)).length)} plotted / ${number(data?.completed_observations || 0)} completed iterations · ${number(data?.missing_observations || 0)} lack complete usage.</span><span>Primary Teacher input + output. Author usage is separate. Totals include passes, retries and unfinished attempts.</span></div></section>`;
}
