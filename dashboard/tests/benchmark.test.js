import test from "node:test";
import assert from "node:assert/strict";
import { benchmarkCard, scoreChart } from "../dist/benchmark.js";
import { usageCharts } from "../dist/telemetry.js";

test("missing eval is not zero; real zero remains a score", () => {
  assert.match(benchmarkCard(null), /usage-total">—/);
  const p = { score: 0, correct: 0, n: 80, model_id: "a".repeat(64), release_name: "<img>", round_number: 5, retained_tokens: 10 };
  const html = benchmarkCard({ latest: p, points: [p], status: "running", stale: true, lag_tokens: 20 });
  assert.match(html, /0\.0%/); assert.match(html, /Scored iteration 5/);
  assert.match(html, /Earlier weights/); assert.match(html, /Evaluating/);
  assert.ok(!html.includes("<img>"));
});
test("chart shows measurements only, with no interpolating path or invalid numbers", () => {
  const chart=scoreChart([{score:.5,retained_tokens:0},{score:.4,retained_tokens:10}]);
  assert.equal((chart.match(/<circle/g)||[]).length,2);
  assert.ok(!chart.includes("<path"));assert.ok(!/NaN|Infinity/.test(chart));
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
