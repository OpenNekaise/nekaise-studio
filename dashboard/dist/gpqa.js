import { escapeHTML as e, number, dateLabel } from "./lib.js?v=d2803ff6ecfc";

const pct = value => Number.isFinite(value) ? `${number(value * 100, 1)}%` : "—";
const model = row => `${row.model_label}${row.round_number ? ` · iteration ${row.round_number}` : ""} · ${row.model_id?.slice(0, 8) || "pending identity"}`;

export function gpqaCard(data) {
  const runs = data?.status === "ok" && Array.isArray(data.runs) ? data.runs : [];
  const attempt = runs[0];
  const results = runs.filter(row => row.status === "complete");
  const latest = results[0];
  const status = data?.status === "unavailable" ? "Unavailable" : !attempt ? "Not evaluated" :
    attempt.status === "running" ? (data.stale ? "No recent progress" : `Evaluating · ${attempt.completed} / 198`) :
    attempt.status === "failed" ? "Latest attempt failed" : attempt.status === "interrupted" ? "Latest attempt interrupted" : "Completed";
  const progress = attempt?.status === "running" ? `<p class="muted">${e(model(attempt))} · ${number(attempt.completed)} / 198 answered${data.stale ? " · Check the independent evaluator" : ""}</p>` : "";
  const details = latest ? `<details class="eval-details" data-detail="gpqa-details"><summary>Evaluation details &amp; recent results</summary><div>
    <p>95% interval: ${pct(latest.ci95[0])}–${pct(latest.ci95[1])} · random-choice reference: 25%.</p>
    <p>${number(latest.invalid)} invalid answers · ${number(latest.budget_exhausted)} output-budget failures. Both count as incorrect.</p>
    <p>Zero-shot generation · one response per question · ${number(latest.max_new_tokens)} output tokens · ${number(latest.max_input_tokens)} input tokens · CPU FP32.</p>
    <p>${e(latest.protocol)} · protocol ${e(latest.protocol_id.slice(0, 12))} · dataset ${e(latest.dataset_sha256.slice(0, 12))}</p>
    <p>Intervals describe this fixed question set. Compare results only under matching protocols; published leaderboard settings may differ.</p>
    <div class="table-scroll"><table><thead><tr><th>Evaluated model</th><th>Score</th><th>Protocol</th><th>Evaluated</th></tr></thead><tbody>${results.map(row => `<tr><td>${e(model(row))}</td><td>${pct(row.score)} · ${row.correct}/198</td><td>${e(row.protocol_id.slice(0, 8))} · ${number(row.max_new_tokens)} tokens</td><td>${e(dateLabel(row.updated_at))}</td></tr>`).join("")}</tbody></table></div>
    </div></details>` : "";
  return `<section class="panel usage-card eval-usage gpqa-card"><div class="usage-heading"><h2>GPQA Diamond</h2><span class="usage-live">${e(status)}</span></div>
    <strong class="usage-total">${pct(latest?.score)}</strong>
    <p class="usage-description">Graduate-level science${latest ? ` · ${latest.correct} / 198 correct` : " · 198 questions"}</p>
    ${progress}<div class="usage-footer"><span>${latest ? e(model(latest)) : "No completed full evaluation yet."}</span>
    <span>${latest ? `${attempt !== latest ? "Last completed result · " : ""}${e(dateLabel(latest.updated_at))}` : "Run the GPQA command in Bench to evaluate a local model."}</span>
    <span>Standalone evaluations across models · independent of the selected training run</span></div>${details}</section>`;
}
