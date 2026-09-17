import test from "node:test";
import assert from "node:assert/strict";
import { createBenchmarkBrowser } from "../dist/benchmark-browser.js";

const snapshot = (campaign = "a", id = "1", count = 241) => ({
  campaign_id: campaign, points: [{ model_id: "baseline" }],
  history: { id: id.repeat(64), cohort_id: "a".repeat(64), count, pages: Math.ceil(count/100), page_size: 100 },
});
const page = (s, n) => ({ schema_version: 2, campaign_id: s.campaign_id, history_id: s.history.id,
  cohort_id: s.history.cohort_id, count: s.history.count, pages: s.history.pages, page_size: 100,
  page: n, points: Array.from({ length: Math.min(100, s.history.count - n*100) }, (_, i) => ({ model_id: `point-${n*100+i}` })) });

test("browse all pages while newer observations preserve the opened immutable snapshot", async () => {
  const s = snapshot(), newer = snapshot("a", "2", 242), calls = [];
  const browser = createBenchmarkBrowser(async path => {
    calls.push(path);
    const url = new URL(path, "https://fixture.invalid");
    const current = url.searchParams.get("history") === s.history.id ? s : newer;
    return page(current, Number(url.searchParams.get("page")));
  });
  await browser.open(s);
  assert.equal(browser.state.data.page, 2);
  assert.equal(browser.state.data.points.length, 41);
  browser.observe(null);
  assert.equal(browser.state.open, true);
  assert.equal(browser.state.snapshot, s);
  browser.observe(newer);
  assert.equal(browser.state.snapshot, s);
  assert.equal(browser.state.newerAvailable, true);
  await browser.load(0);
  assert.equal(browser.state.data.history_id, s.history.id);
  browser.metric("choice");
  assert.equal(browser.state.metric, "choice");
  browser.metric("toString");
  assert.equal(browser.state.metric, "choice");
  browser.select("point-1");
  assert.equal(browser.state.selected_model_id, "point-1");
  browser.select("not-in-page");
  assert.equal(browser.state.selected_model_id, "point-1");
  browser.overview();
  assert.equal(browser.state.page, null);
  browser.select("baseline");
  assert.equal(browser.state.selected_model_id, "baseline");
  await browser.refresh();
  assert.equal(browser.state.data.history_id, newer.history.id);
  assert.equal(browser.state.newerAvailable, false);
  assert.equal(calls.length, 3);
});

test("campaign changes, newer page requests and close invalidate late responses", async () => {
  const pending = [], s = snapshot();
  const browser = createBenchmarkBrowser(path => new Promise(resolve => pending.push({path, resolve})));
  const first = browser.open(s);
  const second = browser.load(0);
  pending[1].resolve(page(s, 0)); await second;
  pending[0].resolve(page(s, 2)); await first;
  assert.equal(browser.state.data.page, 0);
  const third = browser.load(1);
  browser.observe(snapshot("b"));
  pending[2].resolve(page(s, 1)); await third;
  assert.equal(browser.state.open, false);
  assert.equal(browser.state.data, null);
  const fourth = browser.open(s);
  browser.close();
  pending[3].resolve(page(s, 2)); await fourth;
  assert.equal(browser.state.open, false);
  assert.equal(browser.state.data, null);
});

test("missing or mixed snapshot pages are explicit errors and the same page can retry", async () => {
  const s = snapshot(); let broken = true, calls = 0;
  const browser = createBenchmarkBrowser(async () => { calls++; return broken ? page(snapshot("other"), 2) : page(s, 2); });
  await browser.open(s);
  assert.equal(browser.state.data, null);
  assert.match(browser.state.error, /does not match/);
  assert.equal(browser.state.loading, false);
  broken = false;
  await browser.load(2);
  assert.equal(browser.state.error, null);
  assert.equal(browser.state.data.page, 2);
  await browser.load(3); await browser.load(-1); await browser.load(NaN);
  assert.equal(calls, 2);
});

test("v1 or unavailable projections never trigger a history request", async () => {
  const browser = createBenchmarkBrowser(async () => assert.fail("Unexpected history read"));
  await browser.open({ campaign_id: "a", schema_version: 1 });
  assert.equal(browser.state.open, false);
  await browser.open({ campaign_id: "a", status: "unavailable" });
  assert.equal(browser.state.open, false);
});
