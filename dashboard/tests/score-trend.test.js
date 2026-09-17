import test from "node:test";
import assert from "node:assert/strict";
import { smoothScores, scoreTrend } from "../dist/score-trend.js";

test("a fit needs three distinct valid observations; missing scores never become zero", () => {
  for (const values of [[], [{x: 1, y: .5}], [{x: 1, y: .5}, {x: 2, y: .4}],
    [{x: 1, y: .5}, {x: 1, y: .4}, {x: 1, y: .9}],
    [null, {x: 0, y: 0}, {x: 1, y: null}, {x: 2, y: NaN}, {x: 3, y: 2}]]) {
    assert.deepEqual(smoothScores(values), []);
    assert.equal(scoreTrend(values, x => x, y => y), "");
  }
});

test("constant scores including zero remain constant without extending the observed range", () => {
  for (const score of [0, .37, 1]) {
    const fitted = smoothScores([100, 110, 900, 1000].map(x => ({x, y: score})));
    assert.equal(fitted[0].x, 100); assert.equal(fitted.at(-1).x, 1000);
    for (const p of fitted) assert.ok(Math.abs(p.y-score) < 1e-12);
  }
});

test("smoothing reduces alternating noise and respects token distance rather than array position", () => {
  const noisy = Array.from({length: 81}, (_, x) => ({x, y: x % 2}));
  const fitted = smoothScores(noisy);
  for (const p of fitted.slice(10, -10)) assert.ok(p.y > .49 && p.y < .51);
  const uneven = [0, 1, 100, 101, 102].map((x, i) => ({x, y: i < 2 ? 0 : 1}));
  const byTokens = smoothScores(uneven);
  assert.ok(byTokens[0].y < .01); assert.ok(byTokens.at(-1).y > .99);
  const scaled = smoothScores(uneven.map(p => ({x: 5000 + p.x * 12000, y: p.y})));
  scaled.forEach((p, i) => assert.ok(Math.abs(p.y - byTokens[i].y) < 1e-12));
});

test("duplicate positions and sparse gaps stay finite, bounded and leave input observations unchanged", () => {
  const points = [0, 0, 1, 2, 3, 1000000000].map((x, i) => ({x, y: i % 2}));
  const before = structuredClone(points);
  const fitted = smoothScores(points);
  assert.deepEqual(points, before);
  assert.ok(fitted.length > 3);
  fitted.forEach(p => assert.ok(Number.isFinite(p.y) && p.y >= 0 && p.y <= 1));
  const path = scoreTrend(points, x => x / 2000000, y => (1-y)*150);
  assert.ok(!/NaN|Infinity|undefined/.test(path));
  assert.match(path, /Smoothed trend of displayed observations/);
});
