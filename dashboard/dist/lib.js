export const escapeHTML = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
export const number = (value, digits = 0) =>
  Number.isFinite(value)
    ? new Intl.NumberFormat("en-GB", {
        maximumFractionDigits: digits,
        minimumFractionDigits: digits,
      }).format(value)
    : "—";
export const duration = (seconds) =>
  !Number.isFinite(seconds)
    ? "—"
    : seconds < 60
      ? `${Math.round(seconds)}s`
      : seconds < 3600
        ? `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
        : `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
export const time = (value) =>
  value
    ? new Date(value).toLocaleTimeString("en-GB", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      })
    : "—";
export const relativeTime = (value) => {
  const s = Math.max(0, (Date.now() - new Date(value).getTime()) / 1000);
  return s < 60
    ? "just now"
    : s < 3600
      ? `${Math.floor(s / 60)}m ago`
      : `${Math.floor(s / 3600)}h ago`;
};
export const percent = (value) =>
  Number.isFinite(value) ? `${Math.round(value * 100)}%` : "—";
export function safeURL(value) {
  try {
    const u = new URL(value);
    return ["https:", "http:"].includes(u.protocol) ? escapeHTML(u.href) : "#";
  } catch {
    return "#";
  }
}

// A bounded word diff. Only escaped text is rendered, including model outputs.
export function diffWords(before, after) {
  const a = String(before || "").split(/(\s+)/),
    b = String(after || "").split(/(\s+)/);
  if (a.length * b.length > 700000)
    return { before: escapeHTML(before), after: escapeHTML(after) };
  const rows = Array.from(
    { length: a.length + 1 },
    () => new Uint16Array(b.length + 1),
  );
  for (let i = a.length - 1; i >= 0; i--)
    for (let j = b.length - 1; j >= 0; j--)
      rows[i][j] =
        a[i] === b[j]
          ? rows[i + 1][j + 1] + 1
          : Math.max(rows[i + 1][j], rows[i][j + 1]);
  let left = "",
    right = "",
    i = 0,
    j = 0;
  while (i < a.length || j < b.length) {
    if (i < a.length && j < b.length && a[i] === b[j]) {
      left += escapeHTML(a[i++]);
      right += escapeHTML(b[j++]);
    } else if (
      j < b.length &&
      (i === a.length || rows[i][j + 1] >= rows[i + 1][j])
    )
      right += `<mark>${escapeHTML(b[j++])}</mark>`;
    else left += `<del>${escapeHTML(a[i++])}</del>`;
  }
  return { before: left, after: right };
}

export function lossChart(metrics) {
  const points = metrics.filter(
    (m) => Number.isFinite(m.loss) && Number.isFinite(m.step),
  );
  if (!points.length)
    return '<div class="chart-empty"><div>Loss will appear when training starts.<span>One point for every completed optimizer step.</span></div></div>';
  const w = 640,
    h = 250,
    left = 48,
    right = 18,
    top = 24,
    bottom = 52;
  const values = points.map((m) => m.loss),
    min = Math.min(...values),
    max = Math.max(...values),
    pad = Math.max((max - min) * 0.2, 0.12),
    lo = Math.max(0, min - pad),
    hi = max + pad;
  const first = points[0].step,
    last = points.at(-1).step;
  const x = (s) =>
      left +
      (last === first ? 0.5 : (s - first) / (last - first)) *
        (w - left - right),
    y = (v) => top + ((hi - v) / (hi - lo)) * (h - top - bottom);
  const path = points
    .map(
      (m, i) =>
        `${i ? "L" : "M"}${x(m.step).toFixed(2)},${y(m.loss).toFixed(2)}`,
    )
    .join(" ");
  const grid = Array.from({ length: 4 }, (_, i) => {
    const v = hi - ((hi - lo) * i) / 3,
      yy = y(v);
    return `<line class="grid-line" x1="${left}" x2="${w - right}" y1="${yy}" y2="${yy}"/><text x="${left - 10}" y="${yy + 4}" text-anchor="end">${number(v, 2)}</text>`;
  }).join("");
  const labels = [
    ...new Set([first, Math.round(first + (last - first) / 2), last]),
  ]
    .map(
      (s) => `<text x="${x(s)}" y="${h - 27}" text-anchor="middle">${s}</text>`,
    )
    .join("");
  const dots = points
    .map(
      (m) =>
        `<circle class="chart-point" cx="${x(m.step)}" cy="${y(m.loss)}" r="${points.length < 15 ? 3 : 1.5}"><title>${m.round_number ? `Iteration ${m.round_number}, step ${m.round_step}` : `Step ${m.step}`}: loss ${number(m.loss, 4)}</title></circle>`,
    )
    .join("");
  return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Training loss, ${points.length} recorded steps. Latest loss ${number(points.at(-1).loss, 4)}"><defs><linearGradient id="loss-fill" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#8eabb7" stop-opacity=".17"/><stop offset="1" stop-color="#8eabb7" stop-opacity="0"/></linearGradient></defs>${grid}<path d="${path} L${x(last)},${h - bottom} L${x(first)},${h - bottom} Z" fill="url(#loss-fill)"/><path class="chart-line" d="${path}"/>${dots}${labels}<text x="${w / 2}" y="${h - 6}" text-anchor="middle">Recorded optimizer updates</text></svg>`;
}


export function iterationMetrics(rounds) {
  const points = [];
  for (const round of [...rounds].sort((a, b) => a.number - b.number)) {
    for (const metric of [...(round.metrics || [])].sort((a, b) => a.step - b.step)) {
      if (!Number.isFinite(metric.loss) || !Number.isFinite(metric.step)) continue;
      points.push({ ...metric, round_number: round.number, round_step: metric.step, step: points.length + 1 });
    }
  }
  return points;
}

export function dateLabel(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString("en-GB", { day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function modelLabel(value = "") {
  if (value.endsWith("/checkpoint")) return "Saved checkpoint";
  if (value.includes("/snapshots/")) return "Cached base model";
  return value.split("/").filter(Boolean).at(-1) || "Student";
}

// HTTP errors may be plain text (server/proxy failures), not JSON API details.
export async function readAPIResponse(response) {
  let data;
  try {
    data = await response.json();
  } catch {
    if (!response.ok) throw new Error(`Server request failed (HTTP ${response.status}). Please try again shortly.`);
    throw new Error(`The server returned an invalid JSON response (HTTP ${response.status}).`);
  }
  if (!response.ok) {
    const detail = typeof data?.detail === "string" ? data.detail : JSON.stringify(data?.detail || data);
    throw new Error(`HTTP ${response.status}: ${detail || "Server request failed"}`);
  }
  return data;
}
