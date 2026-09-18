import { escapeHTML as e, number, time, dateLabel } from "./lib.js?v=45d8cb9d849a";

export const phases = [
  { name: "Planning", tone: "planning", stages: ["select", "plan"] },
  { name: "Practice", tone: "practice", stages: ["draft", "revise", "expand", "material_select", "gate"] },
  { name: "Learning", tone: "learning", stages: ["freeze", "train"] },
  { name: "Assessment", tone: "assessment", stages: ["evaluate", "answer", "grade"] },
  { name: "Reflection", tone: "reflection", stages: ["adapt"] },
];
export function phaseFor(status, stage) {
  if (status === "recovering") return { name: "Orchestrator review", tone: "practice", moving: true };
  if (status === "waiting") return { name: "Waiting", tone: "assessment", moving: false };
  if (!["running", "pausing", "stopping"].includes(status)) return { name: { complete: "Complete", paused: "Paused", stopped: "Stopped", queued: "Queued", ready: "Ready", failed: "Interrupted", interrupted: "Interrupted" }[status] || "Ready", tone: "idle", moving: false };
  const phase = phases.find(p => p.stages.includes(stage));
  return { ...(phase || { name: "Starting", tone: "planning" }), moving: true };
}
export function elapsedLabel(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  const n = Math.max(0, Math.floor(seconds)), days = Math.floor(n / 86400);
  return `${days ? `${days}d ` : ""}${String(Math.floor(n / 3600) % 24).padStart(2, "0")}:${String(Math.floor(n / 60) % 60).padStart(2, "0")}:${String(n % 60).padStart(2, "0")}`;
}
const compact = value => Number.isFinite(value) ? new Intl.NumberFormat("en-GB", { notation: "compact", maximumFractionDigits: 2 }).format(value) : "—";

export function tokenChart(series, label) {
  const points = (series || []).filter(p => Number.isFinite(p.tokens) && Number.isFinite(Date.parse(p.at)));
  if (!points.length) return '<div class="token-chart-empty">Waiting for reported token usage.</div>';
  const w = 560, h = 200, left = 52, right = 18, top = 16, bottom = 35;
  const start = Date.parse(points[0].at), end = Math.max(start + 1, Date.parse(points.at(-1).at));
  const high = Math.max(1, ...points.map(p => p.tokens));
  const x = at => left + (Date.parse(at) - start) / (end - start) * (w - left - right);
  const y = tokens => top + (1 - tokens / high) * (h - top - bottom);
  const path = points.map((p, i) => `${i ? "H" : "M"}${x(p.at).toFixed(2)}${i ? "V" : ","}${y(p.tokens).toFixed(2)}`).join(" ");
  const grid = [0, .5, 1].map(r => `<line x1="${left}" x2="${w - right}" y1="${y(high * r)}" y2="${y(high * r)}"/><text x="${left - 10}" y="${y(high * r) + 4}" text-anchor="end">${compact(high * r)}</text>`).join("");
  const ticks = [start, (start + end) / 2, end].map((at, i) => {
    const stamp = new Date(at).toISOString();
    return `<text x="${x(stamp)}" y="${h - 9}" text-anchor="${i === 0 ? "start" : i === 2 ? "end" : "middle"}">${e(new Date(start).toDateString() !== new Date(end).toDateString() ? new Date(at).toLocaleDateString("en-GB", { day: "2-digit", month: "short" }) + " " : "")}${e(time(stamp).slice(0, 5))}</text>`;
  }).join("");
  return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${e(label)}, cumulative tokens. Latest ${number(points.at(-1).tokens)}.">${grid}<path class="token-area" d="${path} L${x(points.at(-1).at)},${h - bottom} L${left},${h - bottom} Z"/><path class="token-line" d="${path}"/>${points.map(p => `<circle class="token-point" cx="${x(p.at)}" cy="${y(p.tokens)}" r="3"><title>${e(dateLabel(p.at))} · ${number(p.tokens)} tokens</title></circle>`).join("")}${ticks}</svg>`;
}

export function usageCharts(telemetry) {
  if (!telemetry || telemetry.unavailable) return ["Teacher tokens", "Tokens trained"].map((title, i) => `<section class="panel usage-card ${i ? "training-usage" : "teacher-usage"}"><div class="usage-heading"><h2>${title}</h2></div><strong class="usage-total">—</strong><p class="usage-description">${telemetry?.unavailable ? "Usage unavailable" : "Loading measurements…"}</p><div class="token-chart-empty">${telemetry?.unavailable ? "The next refresh will try again." : "Waiting for recorded usage."}</div></section>`).join("");
  const teacher = telemetry.teacher, trained = telemetry.training;
  return `<section class="panel usage-card teacher-usage"><div class="usage-heading"><h2>Teacher tokens</h2><span class="usage-live">Reported usage</span></div><strong class="usage-total" title="${number(teacher.total)} tokens">${compact(teacher.total)}</strong><p class="usage-description">Input + output · cache included once</p><div class="token-chart">${tokenChart(teacher.series, "Teacher usage")}</div><div class="usage-footer"><span>${teacher.total === null ? "Usage not reported yet" : `${compact(teacher.input)} input · ${compact(teacher.output)} output · ${compact(teacher.cached)} cached input`}</span><span>${teacher.pending_calls ? `${teacher.pending_calls} ${teacher.pending_calls === 1 ? "call" : "calls"} in progress` : "Updates after each call"}${teacher.missing_calls ? ` · ${teacher.missing_calls} ${teacher.missing_calls === 1 ? "call" : "calls"} without usage` : ""}</span></div></section><section class="panel usage-card training-usage"><div class="usage-heading"><h2>Tokens trained</h2><span class="usage-live">Measured updates</span></div><strong class="usage-total" title="${number(trained.total)} tokens">${compact(trained.total)}</strong><p class="usage-description">Actual training targets · all passes and retries</p><div class="token-chart">${tokenChart(trained.series, "Training exposure")}</div><div class="usage-footer"><span>${number(trained.updates)} optimizer updates · ${number(telemetry.completed_rounds)} completed iterations across this lineage</span><span>Consumed work includes updates that may not survive in the retained checkpoint.</span></div></section>`;
}
