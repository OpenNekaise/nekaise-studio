import test from "node:test";
import assert from "node:assert/strict";
import { gpqaCard } from "../dist/gpqa.js";

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
