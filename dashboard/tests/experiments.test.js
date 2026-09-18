import test from "node:test";
import assert from "node:assert/strict";
import { createExperimentBrowser } from "../dist/experiment-browser.js";
import { experimentCard, experimentsView } from "../dist/experiments.js";

const card = () => ({
  id: "r1", round_id: "r1", campaign_id: "c1", round_number: 1,
  planned_at: "2026-09-18T12:00:00Z", execution: { status: "failed" },
  strategy_version: "a".repeat(64), strategy: { name: "Fixture strategy", approach: "Compare units", parent_version: "" },
  starting_point: { checkpoint: "/fixture/input" }, artifacts: { select: { hash: "b".repeat(64) } },
  plan: { title: "<script>Fixture hypothesis</script>", hypothesis: "A & B", intervention: "Use contrasts",
    observation_plan: "Inspect transfer", budget_basis: "Unmatched", reconsider_if: "No transfer" },
  review: null, related_rounds: [], assessment: { status: "not_recorded", requested_pairs: null, paired: [] },
});

test("experiment card distinguishes pre-training intent, missing conclusions and unavailable usage", () => {
  const html = experimentCard(card());
  assert.match(html, /Hypothesis · recorded before training/);
  assert.match(html, /No Teacher conclusion recorded/);
  assert.match(html, /execution failed/);
  assert.match(html, /Completed assessment evidence is not available/);
  assert.match(html, /&lt;script&gt;/);
  assert.ok(!html.includes("<script>"));
  assert.match(html, /data-experiment-round="r1"/);
  assert.ok(!html.includes("Independent eval"));
});

test("conclusions and paired evidence do not infer success or aggregate learning", () => {
  const value = card();
  value.review = { status: "abandoned", findings: "No observed improvement", limitations: "One round", next_action: "Try another approach", evaluation_ids: ["q1"] };
  value.assessment = { status: "complete", requested_pairs: 1, paired: [{ id: "q1", question: "Which unit?", before_score: .5, after_score: .5, score_delta: null, weights_changed: false, comparable: false }] };
  value.learning_work = { measured_work_all_attempts: { tokens: 300 }, teacher_efficiency: { ratio: null, status: "missing", teacher_tokens: null } };
  const html = experimentCard(value);
  assert.match(html, /Investigation abandoned/);
  assert.match(html, /No observed improvement/);
  assert.match(html, /Weights unchanged/);
  assert.match(html, /Usage incomplete or unavailable/);
  assert.ok(!html.includes("+0.0 pp"));
  assert.match(experimentsView({ page: { items: [], total: 0 }, before: null }), /No teaching experiments recorded/);
});

test("entirely unreported author usage is unknown rather than zero cost", () => {
  const value = card();
  value.learning_work = { measured_work_all_attempts: { tokens: 0 },
    material_author_work: { calls: 2, calls_without_usage: 2, reported_input_tokens: 0, reported_output_tokens: 0 } };
  const html = experimentCard(value);
  assert.match(html, /Material Author tokens<\/dt><dd>—<\/dd>/);
  assert.match(html, /partial usage/);
});

test("cursor paging and strategy filtering remain read-only", async () => {
  const requests = [];
  const browser = createExperimentBrowser(async path => {
    requests.push(path);
    if (path.startsWith("/experiments?")) return { items: [{ round_id: path.includes("before") ? "r0" : "r1" }], total: 30, next_before: 12 };
    return { round_id: path.split("/").at(-1) };
  });
  await browser.refresh();
  assert.equal(browser.state.detail.round_id, "r1");
  await browser.load(12);
  assert.equal(browser.state.detail.round_id, "r0");
  await browser.refresh();
  assert.equal(browser.state.before, 12);
  await browser.filter("a".repeat(64));
  assert.equal(browser.state.before, null);
  assert.ok(requests.at(-2).includes("strategy_version="));
  assert.ok(requests.every(path => path.startsWith("/experiments")));
});

test("late pages and details cannot overwrite newer selections", async () => {
  let finishPage, finishDetail;
  const browser = createExperimentBrowser(async path => {
    if (path === "/experiments?limit=20") return new Promise(resolve => { finishPage = resolve; });
    if (path.startsWith("/experiments?")) return { items: [{ round_id: "new" }], total: 1 };
    if (path === "/experiments/old") return new Promise(resolve => { finishDetail = resolve; });
    return { round_id: "new" };
  });
  const stalePage = browser.load();
  await browser.filter("f".repeat(64));
  finishPage({ items: [{ round_id: "old" }], total: 1 });
  await stalePage;
  assert.equal(browser.state.selectedId, "new");
  const staleDetail = browser.select("old");
  await browser.select("new");
  finishDetail({ round_id: "old" });
  await staleDetail;
  assert.equal(browser.state.detail.round_id, "new");
});

test("failed reads expose errors and recover on the next read", async () => {
  let fail = true;
  const browser = createExperimentBrowser(async () => {
    if (fail) throw new Error("Fixture unavailable");
    return { items: [], total: 0, next_before: null };
  });
  await browser.refresh();
  assert.match(browser.state.error, /Fixture unavailable/);
  assert.equal(browser.state.loading, false);
  fail = false;
  await browser.refresh();
  assert.equal(browser.state.error, "");
});

test("an absent legacy plan is an explicit message rather than a blank pane", async () => {
  const browser = createExperimentBrowser(async () => null);
  await browser.select("legacy");
  assert.match(browser.state.error, /No experiment plan was recorded/);
});
