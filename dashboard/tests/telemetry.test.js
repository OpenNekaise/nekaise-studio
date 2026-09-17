import test from "node:test";
import assert from "node:assert/strict";
import { elapsedLabel, phaseFor, tokenChart, usageCharts } from "../dist/telemetry.js";

test("phase labels follow real stages without suggesting progress during a wait", () => {
  assert.equal(phaseFor("running", "revise").name, "Practice");
  assert.equal(phaseFor("running", "train").tone, "learning");
  assert.equal(phaseFor("running", "grade").tone, "assessment");
  for (const status of ["paused", "waiting", "stopped", "complete"]) assert.equal(phaseFor(status, "train").moving, false);
  assert.equal(phaseFor("recovering", "train").name, "Orchestrator review");
});
test("elapsed time carries days without rounding into another minute", () => {
  assert.equal(elapsedLabel(59.9), "00:00:59");
  assert.equal(elapsedLabel(90061), "1d 01:01:01");
  assert.equal(elapsedLabel(null), "—");
});
test("cumulative token charts show measured steps and escape labels", () => {
  const points = [{ at: "2026-09-15T12:00:00Z", tokens: 0 }, { at: "2026-09-15T12:01:00Z", tokens: 120 }];
  const chart = tokenChart(points, "<img>Teacher");
  assert.match(chart, /Latest 120/);
  assert.ok(!chart.includes("<img>"));
  assert.ok(!/NaN|Infinity/.test(chart));
  assert.match(chart, /H[\d.]+V[\d.]+/);
  assert.match(tokenChart([], "Missing"), /Waiting for reported/);
  assert.ok(!/NaN|Infinity/.test(tokenChart([points[0]], "One point")));
});
test("session layout separates provider usage from actual training exposure", () => {
  const t = { elapsed_seconds: 100, completed_rounds: 2, teacher: { total: 120, input: 100, output: 20, cached: 80, pending_calls: 1, missing_calls: 2, series: [] }, training: { total: 30, updates: 2, series: [] } };
  const html = usageCharts(t, { status: "running" }, { number: 3, stage: "train" });
  assert.match(html, /Teacher tokens/);
  assert.match(html, /Tokens trained/);
  assert.match(html, /cache included once/);
  assert.match(html, /2 calls without usage/);
  assert.match(html, /1 call in progress/);
  assert.ok(!/Happening in the studio|teacher calls|All activity/.test(html));
});
