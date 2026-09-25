import test from "node:test";
import assert from "node:assert/strict";
import { gpqaCard, SFT_BASELINE_ID } from "../dist/gpqa.js";

const result = { status: "complete", model_label: "Kai 0.1", round_number: 14, model_id: "a".repeat(64),
  score: 0, correct: 0, ci95: [0, .03], invalid: 198, budget_exhausted: 0, max_new_tokens: 1024,
  max_input_tokens: 4096, protocol: "gpqa-diamond-gen-1", protocol_id: "b".repeat(64), dataset_sha256: "c".repeat(64), updated_at: "2026-09-25T12:00:00Z" };

test("empty, unavailable, running, and real zero remain distinct", () => {
  assert.match(gpqaCard(null), /Not evaluated/);
  assert.match(gpqaCard({ status: "unavailable" }), /Unavailable/);
  assert.doesNotMatch(gpqaCard(null), />0%/);
  const full = gpqaCard({ status: "ok", runs: [result] });
  assert.match(full, /0\.0%/);
  assert.match(full, /Kai 0.1 · iteration 14/);
  assert.match(full, /0 \/ 198 correct/);
  assert.match(full, /independent of the selected training run/);
  const progress = gpqaCard({ status: "ok", runs: [{ ...result, status: "running", completed: 12, score: null }, result] });
  assert.match(progress, /12 \/ 198/);
  assert.match(progress, /Last completed result/);
  assert.match(gpqaCard({ status: "ok", stale: true, runs: [{ ...result, status: "running" }] }), /No recent progress/);
});

test("untrusted labels are escaped and failure never invents a score", () => {
  const html = gpqaCard({ status: "ok", runs: [{ ...result, model_label: "<script>alert(1)</script>" }] });
  assert.doesNotMatch(html, /<script>/);
  assert.match(html, /&lt;script&gt;/);
  assert.match(gpqaCard({ status: "ok", runs: [{ ...result, status: "failed" }] }), /Latest attempt failed/);
  assert.doesNotMatch(gpqaCard({ status: "ok", runs: [{ ...result, status: "failed" }] }), /0 \/ 198 correct/);
});

const baseline = { ...result, model_id: SFT_BASELINE_ID, model_label: "Local checkpoint", round_number: null,
  score: .5, correct: 99, ci95: [.43, .57], invalid: 4, updated_at: "2026-09-25T13:00:00Z" };

test("measured baseline draws a horizontal line without replacing the model headline", () => {
  const html = gpqaCard({ status: "ok", runs: [baseline, result] });
  assert.match(html, /class="usage-total">0\.0%/);
  assert.match(html, /class="gpqa-baseline-line"[^>]*y1="112" y2="112"/);
  assert.match(html, /SFT starting point · 50\.0%/);
  assert.match(html, /Difference from starting point: -50\.0 percentage points/);
  assert.match(html, /MiniCPM5-1B-SFT · starting point/);
  assert.equal((html.match(/class="gpqa-score-point"/g) || []).length, 1);
});

test("zero baseline is real, but pending or failed baselines cannot draw a line", () => {
  const zero = gpqaCard({ status: "ok", runs: [{ ...baseline, score: 0, correct: 0, ci95: [0, .02] }, result] });
  assert.match(zero, /SFT starting point · 0\.0%/);
  assert.match(zero, /class="gpqa-baseline-line"/);
  for (const status of ["running", "failed", "interrupted"]) {
    const html = gpqaCard({ status: "ok", runs: [{ ...baseline, status, score: null, completed: 10 }, result] });
    assert.doesNotMatch(html, /class="gpqa-baseline-line"/);
    assert.match(html, /awaiting a completed, verified evaluation/);
    assert.match(html, /class="usage-total">0\.0%/);
  }
});

test("protocol mismatch is explicit and excludes incompatible chart points", () => {
  const other = { ...result, protocol_id: "d".repeat(64), updated_at: "2026-09-25T14:00:00Z" };
  const html = gpqaCard({ status: "ok", runs: [other, baseline, result] });
  assert.doesNotMatch(html, /class="gpqa-baseline-line"/);
  assert.match(html, /Baseline uses a different protocol/);
  assert.match(html, /1 result\(s\) under other protocols/);
  assert.equal((html.match(/class="gpqa-score-point"/g) || []).length, 1);
  for (const change of [{ dataset_sha256: "e".repeat(64) }, { max_new_tokens: 256 }, { max_input_tokens: 2048 }]) {
    assert.doesNotMatch(gpqaCard({ status: "ok", runs: [{ ...baseline, ...change }, result] }), /class="gpqa-baseline-line"/);
  }
});

test("baseline-only state has a reference line but no invented student score", () => {
  const html = gpqaCard({ status: "ok", runs: [baseline] });
  assert.match(html, /class="usage-total">—/);
  assert.match(html, /class="gpqa-baseline-line"/);
  assert.doesNotMatch(html, /class="gpqa-score-point"/);
  assert.match(html, /No completed model evaluation beyond the starting point yet/);
});

test("a baseline name cannot impersonate the pinned model identity", () => {
  const html = gpqaCard({ status: "ok", runs: [{ ...result, model_label: "MiniCPM5-1B-SFT · starting point" }] });
  assert.doesNotMatch(html, /class="gpqa-baseline-line"/);
});
