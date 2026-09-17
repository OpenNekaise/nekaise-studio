import test from "node:test";
import assert from "node:assert/strict";
import { benchmarkCard, benchmarkHistory, scoreChart, METRICS } from "../dist/benchmark.js";
import { usageCharts } from "../dist/telemetry.js";

const hex = c => c.repeat(64);
const point = (i, extra = {}) => ({ model_id: hex(String(i % 10)), root_id: hex("0"), round_id: i ? `round-${i}` : null, round_number: i, retained_tokens: i * 1000, checkpoint_at: `2026-09-${String(1 + i % 20).padStart(2, "0")}T10:00:00Z`, evaluated_at: `2026-09-${String(1 + i % 20).padStart(2, "0")}T12:00:00Z`, run_id: `run-${i}`, protocol_id: hex("1"), protocol: "chat-1", release_id: hex("2"), release_name: "core-v1", n: 80, correct: 40, score: .5, invalid: 3, budget_exhausted: 1, numerical: .4, choice: .6, delta: i ? .05 : null, delta_ci95: i ? [-.01, .11] : null, clusters: i ? 12 : null, interval_reliable: i > 0, observation_kind: i === 0 ? "baseline" : "sample", numerical_n: 40, numerical_correct: 16, choice_n: 40, choice_correct: 24, gained: i ? 7 : null, lost: i ? 3 : null, ...extra });
const comparison = (kind, extra = {}) => ({ kind, reference_model_id: hex("0"), reference_round_id: null, reference_tokens: 0, reference_score: .45, n: 80, delta: .05, delta_ci95: [-.01, .11], gained: 7, lost: 3, invalid_to_correct: 2, correct_to_invalid: 1, clusters: 12, interval_reliable: true, interpretation: "descriptive", ...extra });
const v2 = (overrides = {}) => ({ schema_version: 2, campaign_id: "c", generated_at: "2026-09-17T12:00:00Z", status: "ok", stale: false, issues: [], latest: point(9), points: [point(0), point(3, { observation_kind: "milestone" }), point(6), point(9)], history: { id: hex("a"), cohort_id: hex("b"), count: 950, pages: 10, page_size: 100, overview_count: 4 }, comparisons: { baseline: comparison("baseline"), milestone: null }, comparison_reasons: { baseline: null, milestone: "No declared milestone has completed evaluation" }, ...overrides });
const page = (n, overrides = {}) => ({ schema_version: 2, campaign_id: "c", history_id: hex("a"), cohort_id: hex("b"), count: 950, page_size: 100, page: n, pages: 10, points: Array.from({ length: n === 9 ? 50 : 100 }, (_, i) => point(n * 100 + i + 1, { model_id: String(n * 100 + i + 1).padStart(64, "f") })), ...overrides });
const open = extra => ({ metric: "score", open: true, history_id: hex("a"), page: null, data: null, loading: false, error: null, selected_model_id: null, ...extra });

test("missing eval is not zero; real zero remains a score", () => {
  assert.match(benchmarkCard(null), /usage-total">—/);
  const p = { score: 0, correct: 0, n: 80, model_id: "a".repeat(64), release_name: "<img>", round_number: 5, retained_tokens: 10 };
  const html = benchmarkCard({ latest: p, points: [p], status: "running", stale: true, lag_tokens: 20 });
  assert.match(html, /0\.0%/); assert.match(html, /Scored iteration 5/);
  assert.match(html, /Earlier weights/); assert.match(html, /Evaluating/);
  assert.ok(!html.includes("<img>"));
});
test("chart preserves observations and adds a labeled fit only with enough distinct positions", () => {
  const chart=scoreChart([{score:.5,retained_tokens:0},{score:.4,retained_tokens:10}]);
  assert.equal((chart.match(/<circle/g)||[]).length,2);
  assert.ok(!chart.includes("<path"));assert.ok(!/NaN|Infinity/.test(chart));
  const fitted = scoreChart([point(0), point(2), point(9)], "numerical", { interactive: true });
  assert.match(fitted, /class="score-trend"/);
  assert.match(fitted, /Smoothed trend/);
  assert.equal((fitted.match(/data-benchmark-point=/g) || []).length, 3);
  assert.equal((fitted.match(/<circle/g) || []).length, 3);
});
test("usage and independent evaluation preserve separate measurement semantics", () => {
  const t={elapsed_seconds:1,completed_rounds:1,teacher:{total:10,input:8,output:2,cached:0,series:[]},training:{total:15,updates:1,series:[]}};
  const usage=usageCharts(t,{id:"run"});
  assert.match(usage,/Teacher tokens/);assert.match(usage,/Tokens trained/);
  assert.ok(!usage.includes("Independent eval"));
  const evaluation=benchmarkCard({status:"failed_infra",issues:[{status:"failed_infra"}]});
  assert.match(evaluation,/Evaluation failed/);
  assert.ok(!evaluation.includes("Teacher tokens"));
});

test("schema v1 projection without history renders the compact card unchanged and no history panel", () => {
  const p = point(2); delete p.observation_kind; delete p.numerical_n; delete p.gained;
  const html = benchmarkCard({ schema_version: 1, latest: p, points: [point(0), p], status: "ok" });
  assert.match(html, /\+5\.00 pp vs matching baseline/);
  assert.ok(!html.includes("data-benchmark-open"));
  assert.ok(!html.includes("gained"));
  assert.equal(benchmarkHistory({ schema_version: 1, latest: p, points: [p] }, open()), "", "no history identity means no panel even if left open");
  assert.equal(benchmarkHistory(null, open()), "");
});

test("compact card shows baseline gain, freshness and long-history count with an open button", () => {
  const html = benchmarkCard(v2(), { open: false });
  assert.match(html, /\+5\.00 pp vs matching baseline · 7 gained, 3 lost/);
  assert.match(html, /950 observations recorded · 4 shown/);
  assert.match(html, /data-benchmark-open aria-expanded="false"/);
  assert.match(html, /Measured .*old at the last update/);
  assert.match(html, /Numerical 40\.0% \(16 \/ 40\)/);assert.match(html, /Choice 60\.0% \(24 \/ 40\)/);
  assert.match(html, /eval-reference/);
  assert.match(benchmarkCard(v2(), { open: true }), /aria-expanded="true"/);
  assert.ok(!html.includes("data-benchmark-metric"));
  assert.ok(!/general intelligence score(?! or)/.test(html));
});

test("history panel metadata, metric selector and overview sample count", () => {
  const html = benchmarkHistory(v2(), open());
  assert.match(html, /id="benchmark-history"/);assert.match(html, /950 recorded observations · Same questions and evaluation protocol/);
  for (const label of ["Overall", "Numerical", "Choice", "Invalid answers", "Output limit"]) assert.match(html, new RegExp(`data-benchmark-metric="[a-z_]+" aria-pressed="(true|false)">${label}<`));
  assert.match(html, /data-benchmark-metric="score" aria-pressed="true"/);
  assert.match(html, /Overview: 4 of 950 sampled across the whole history/);
  assert.match(html, /data-benchmark-overview aria-pressed="true"/);
  assert.match(html, /data-benchmark-close/);
  assert.equal((html.match(/<circle/g) || []).length, 4);
  assert.match(html, /eval-point milestone/);assert.match(html, /eval-point baseline/);
  assert.match(html, /class="score-trend"/);assert.ok(!/NaN|Infinity|undefined/.test(html));
  const visible = html.replace(/<[^>]+>/g, " ");
  for (const jargon of ["invalid_rate", "budget_rate", "observation_kind", "delta_ci95", "schema_version"]) assert.ok(!visible.includes(jargon), jargon);
  assert.equal((html.match(/<th scope="col">Overall<\/th>/g) || []).length, 1);
  assert.match(benchmarkHistory(v2(), open({ metric: "budget_rate" })), /<th scope="col">Output limit \(shown\)<\/th>/);
  assert.equal((html.match(/<tbody>[\s\S]*?<\/tbody>/)[0].match(/<tr>/g) || []).length, 4);
  assert.equal(benchmarkHistory(v2(), { open: false }), "");
});

test("metrics use displayed values, missing values are absent rather than zero, and errors show", () => {
  const data = v2({ points: [point(0, { numerical: null, numerical_n: null, numerical_correct: null }), point(4, { invalid: 8, n: 80, budget_exhausted: 4 })] });
  const invalid = benchmarkHistory(data, open({ metric: "invalid_rate" }));
  assert.match(invalid, /data-benchmark-metric="invalid_rate" aria-pressed="true"/);
  assert.match(invalid, /share of questions with an invalid answer/);
  assert.equal((invalid.match(/<circle/g) || []).length, 2);
  const numerical = benchmarkHistory(data, open({ metric: "numerical" }));
  assert.equal((numerical.match(/<circle/g) || []).length, 1);
  assert.match(numerical, /<td>—<\/td>/);
  assert.match(numerical, /Numerical<\/th>/);
  const none = benchmarkHistory(v2({ points: [point(0, { choice: null })] }), open({ metric: "choice" }));
  assert.match(none, /No choice measurements in the displayed observations/);
  assert.ok(!none.includes("0.0%</title>"));
  assert.equal(METRICS.budget_rate.value({ budget_exhausted: 2, n: null }), null);
  assert.equal(METRICS.budget_rate.value({ budget_exhausted: 2, n: 8 }), .25);
  const failed = benchmarkHistory(v2(), open({ error: "<b>fetch failed</b>", loading: true }));
  assert.match(failed, /role="alert">&lt;b&gt;fetch failed&lt;\/b&gt;/);
  assert.ok(!failed.includes("<b>"));
  assert.match(benchmarkHistory(v2(), open({ loading: true })), /Loading observation page/);
});

test("page navigation uses zero-based oldest-first pages and shows only the fetched page", () => {
  const newest = benchmarkHistory(v2(), open({ page: 9, data: page(9) }));
  assert.match(newest, /Page 10 of 10 · oldest first/);
  assert.match(newest, /data-benchmark-page="8" +aria-label="Previous page"/);
  assert.match(newest, /data-benchmark-page="9" disabled aria-label="Next page"/);
  assert.match(newest, /data-benchmark-page="9" disabled aria-label="Newest page"/);
  assert.match(newest, /data-benchmark-page="0" +aria-label="Oldest page"/);
  assert.match(newest, /data-benchmark-overview aria-pressed="false"/);
  assert.equal((newest.match(/<circle/g) || []).length, 50);
  assert.equal((newest.match(/<tbody>[\s\S]*?<\/tbody>/)[0].match(/<tr>/g) || []).length, 50);
  assert.match(newest, /Dots are the observations on this page only/);
  const oldest = benchmarkHistory(v2(), open({ page: 0, data: page(0) }));
  assert.match(oldest, /Page 1 of 10/);
  assert.match(oldest, /data-benchmark-page="0" disabled aria-label="Previous page"/);
  assert.match(oldest, /data-benchmark-page="1" +aria-label="Next page"/);
  assert.equal((oldest.match(/<tbody>[\s\S]*?<\/tbody>/)[0].match(/<tr>/g) || []).length, 100);
  const big = benchmarkHistory(v2(), open({ page: 0, data: page(0, { page_size: 100, points: Array.from({ length: 300 }, (_, i) => point(i + 1)) }) }));
  assert.equal((big.match(/<tbody>[\s\S]*?<\/tbody>/)[0].match(/<tr>/g) || []).length, 100);
  const mismatch = benchmarkHistory(v2(), open({ page: 3, data: page(3, { history_id: hex("c") }) }));
  assert.match(mismatch, /Overview: 4 of 950/);
  const overview = benchmarkHistory(v2(), open({ page: null, data: page(9) }));
  assert.match(overview, /data-benchmark-overview aria-pressed="true"/);
  assert.match(overview, /sampled across the whole history/);
});

test("comparisons are descriptive, label a zero-width interval and never invent a milestone", () => {
  const html = benchmarkHistory(v2({ comparisons: { baseline: comparison("baseline", { delta_ci95: [.02, .02], reference_round_id: "<round>" }), milestone: null } }), open());
  assert.match(html, /Difference from starting model/);
  assert.match(html, /\+2\.00 pp \(zero-width interval/);
  assert.match(html, /7 gained · 3 lost/);assert.match(html, /2 invalid → correct · 1 correct → invalid/);
  assert.match(html, /reference iteration &lt;round&gt;/);
  assert.match(html, /not an acceptance, success or regression verdict/);
  assert.match(html, /No declared milestone has completed evaluation/);
  const none = benchmarkHistory(v2({ comparison_reasons: { baseline: null, milestone: null } }), open());
  assert.match(none, /No declared earlier milestone has been evaluated\. The previous observation is not used as a substitute/);
  assert.ok(!/Difference from declared milestone<\/h3><dl>/.test(none));
  const milestone = benchmarkHistory(v2({ comparisons: { baseline: null, milestone: comparison("milestone", { reference_round_id: "round-3", reference_tokens: 3000, delta: -.02, delta_ci95: [-.08, .04], interval_reliable: false }) }, comparison_reasons: { baseline: "Baseline protocol differs", milestone: null } }), open());
  assert.match(milestone, /Baseline protocol differs/);
  assert.match(milestone, /-2\.00 pp over 80 paired questions/);
  assert.match(milestone, /-8\.00 pp to \+4\.00 pp · limited evidence/);
  assert.match(milestone, /reference iteration round-3 · 45\.0% at 3,000 retained-weight tokens/);
  for (const word of ["Success", "Regression", "Accepted", "Passed"]) assert.ok(!html.includes(word), word);
});

test("selected observation shows details, escapes labels and keeps a legacy point readable", () => {
  const data = v2({ points: [point(0), point(3, { observation_kind: "milestone", release_name: "<script>x</script>", round_id: "r<3>", delta_ci95: [.01, .01] })] });
  const html = benchmarkHistory(data, open({ selected_model_id: hex("3") }));
  assert.match(html, /Selected observation/);
  assert.match(html, /Iteration 3 · Declared milestone/);
  assert.match(html, /round r&lt;3&gt; · run run-3/);
  assert.match(html, /chat-1 · &lt;script&gt;x&lt;\/script&gt;/);
  assert.ok(!html.includes("<script>"));
  assert.match(html, /333333333333 · 3,000 retained-weight tokens · saved/);
  assert.match(html, /Numerical<\/dt><dd>40\.0% · 16 \/ 40/);
  assert.match(html, /95% interval \+1\.00 pp \(zero-width interval/);
  assert.match(html, /7 gained · 3 lost/);
  assert.match(html, /eval-point milestone selected/);
  assert.match(html, /data-benchmark-point="3{64}" aria-label="Show details for Iteration 3[^"]*" aria-pressed="true"/);
  const legacy = benchmarkHistory(v2({ points: [point(2, { numerical_n: undefined, numerical_correct: undefined, gained: undefined, lost: undefined, observation_kind: undefined })] }), open({ selected_model_id: hex("2") }));
  assert.match(legacy, /Iteration 2 · Sample/);
  assert.match(legacy, /40\.0% · counts not recorded/);
  assert.match(legacy, /Gained \/ lost vs starting model<\/dt><dd>not recorded/);
  assert.ok(!legacy.includes("0 gained"));
  assert.ok(!benchmarkHistory(data, open({ selected_model_id: hex("9") })).includes("Selected observation"));
  assert.ok(!benchmarkHistory(data, open()).includes("Selected observation"));
});
