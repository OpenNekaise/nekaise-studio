import test from "node:test";
import assert from "node:assert/strict";
import { efficiencyCard, efficiencyChart, ratioLabel } from "../dist/efficiency.js";

test("efficiency uses unbounded ratios, preserves zero, and fits observed points", () => {
  const points = [0, 1, 4].map((ratio, i) => ({ ratio, iteration: i+1, campaign_id: '<script>', number: i+1, trained_tokens: ratio*100, teacher_tokens: 100 }));
  const html = efficiencyChart(points);
  assert.match(html, /4×/);
  assert.match(html, /score-trend/);
  assert.match(html, /Smoothed efficiency/);
  assert.ok(!html.includes('<script>'));
  assert.ok(!/NaN|Infinity/.test(html));
  assert.equal(ratioLabel(0), '0×');
  assert.equal(ratioLabel(null), '—');
  assert.match(efficiencyChart([{ ...points[0], ratio: null }]), /Waiting for a completed iteration/);
});

test("headline uses pooled backend ratio and labels incomplete provider usage", () => {
  const html = efficiencyCard({ efficiency: { ratio: null, reported_ratio: .0123, status: 'missing', trained_tokens: 123, teacher_tokens: 10000, teacher_usage: { missing_calls: 2, pending_calls: 1 }, missing_observations: 2, series: [] } });
  assert.match(html, /0.0123×/);
  assert.match(html, /Partial reported usage/);
  assert.match(html, /3 calls have pending or missing usage/);
  assert.match(html, /Author usage is separate/);
  assert.ok(!/NaN|Infinity|undefined/.test(efficiencyCard(null)));
});
