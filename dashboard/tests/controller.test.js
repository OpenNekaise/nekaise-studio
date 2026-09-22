import test from "node:test";
import assert from "node:assert/strict";

// Exercise navigation against read-only API fixtures without a browser or model process.
test("iteration navigation, previous runs and stale requests keep the correct run selected", async () => {
  const original = new Map();
  const replace = (name, value) => { original.set(name, Object.getOwnPropertyDescriptor(globalThis, name)); Object.defineProperty(globalThis, name, { configurable: true, writable: true, value }); };
  const listeners = new Map(), registered = new Map(), nodes = new Map(), requests = [];
  class Element {
    constructor(id) { this.id = id; this.innerHTML = ""; this.textContent = ""; this.hidden = false; this.dataset = {}; this.classList = { toggle() {}, contains() { return false; } }; }
    querySelectorAll() { return []; }
    querySelector(selector) { return get(selector === "[type=submit]" ? "chat-send" : selector.replace("#", "")); }
    removeEventListener(type) { listeners.delete(`${this.id}:${type}`); }
    addEventListener(type, handler) { listeners.set(`${this.id}:${type}`, handler); }
    setAttribute() {}
    removeAttribute() {}
    focus() {}
  }
  const get = id => { if (!nodes.has(id)) nodes.set(id, new Element(id)); return nodes.get(id); };
  const campaign = id => ({ id, name: `Run ${id}`, status: "complete", completed_rounds: 1, updated_at: "same", created_at: "2026-09-14T12:00:00Z", config: { focus: "Heat transfer", student_model: "Student", teacher_model: "Teacher" } });
  const round = (id, campaignId, number) => ({ id, campaign_id: campaignId, number, status: "complete", updated_at: "same", metrics: [{ step: 1, loss: 2 }], lessons: [{ id: "lesson", concept: `Topic ${id}`, kind: "sft", prompt: "Question", student: `Attempt ${id}`, teacher: `Revision ${id}`, errors: [], evidence: [], document: { id: "" } }], evaluations: [{ id: "eval", question: `Assessment ${id}`, reference: "Reference", rubric: [], student: `Answer ${id}`, grade: { score: .5, verdict: "partial", feedback: `Feedback ${id}` } }], score: .5 });
  const a1 = round("a1", "a", 1), a2 = round("a2", "a", 2), b1 = round("b1", "b", 1);
  const snapshot = (id, selected) => ({ campaign: campaign(id), rounds: id === "a" ? [a2, a1] : [b1], round: selected || (id === "a" ? a2 : b1), events: [], stages: [], teacher_usage: { calls: 1 }, timestamp: "2026-09-14T12:00:00Z" });
  let deferOld = false, failUsage = false, finishOld;
  const response = data => ({ ok: true, json: async () => structuredClone(data) });
  replace("document", { getElementById: get, querySelectorAll: () => [], addEventListener: (type, handler) => listeners.set(type, handler), activeElement: null, hidden: false, modelContext: { registerTool: tool => registered.set(tool.name, tool) } });
  replace("window", { getSelection: () => ({ toString: () => "" }), addEventListener() {} });
  replace("localStorage", { getItem: () => "b", setItem() {} });
  replace("setInterval", () => 0);
  replace("fetch", async (path, options) => {
    requests.push({ path, method: options?.method || "GET" });
    if (path === "/api/model") return response({ available: true, model: { id: "snapshot123", run_name: "Current student", round_number: 2 } });
    if (path === "/api/campaigns") return response([campaign("a"), campaign("b")]);
    if (path === "/api/experiments?limit=20") return response({ items: [], total: 0, next_before: null });
    if (failUsage && /\/(telemetry|benchmark)$/.test(path)) throw new Error("Measurement service unavailable");
    if (/^\/api\/campaigns\/[ab]\/telemetry$/.test(path)) return response({ campaign_id: path.split("/")[3], elapsed_seconds: 90, clock_running: false, completed_rounds: 1, teacher: { total: 120, input: 100, output: 20, cached: 80, series: [] }, training: { total: 50, updates: 1, series: [] } });
    if (/^\/api\/campaigns\/[ab]\/benchmark$/.test(path)) return response({ campaign_id: path.split("/")[3], status: "not_ready" });
    if (path === "/api/reports") return response({ current: { summary: "Fixture current status", next_action: "Automatic review scheduled", campaign: campaign("a") }, items: [{ id: 1, kind: "failure", reason: "Fixture review", status: "waiting" }], next_before: 1 });
    if (path === "/api/reports?before=1") return response({ current: { summary: "Fixture current status", next_action: "Automatic review scheduled", campaign: campaign("a") }, items: [], next_before: null });
    if (path === "/api/reports/1") return response({ recovery: { id: 1, campaign_id: "a", status: "waiting", decision: { action: "pause", report: "Fixture agent narrative" } }, turns: [], events: [] });
    if (path === "/api/campaigns/a") return response(snapshot("a"));
    if (path === "/api/campaigns/b") return response(snapshot("b"));
    if (path === "/api/rounds/a1") return response(a1);
    if (path === "/api/campaigns/a?round_id=a1") {
      if (deferOld) return new Promise(resolve => { finishOld = () => resolve(response(snapshot("a", a1))); });
      return response(snapshot("a", a1));
    }
    throw new Error(`Unexpected read: ${path}`);
  });
  const click = dataset => listeners.get("click")({ target: { closest: () => ({ id: "", dataset, classList: { contains: () => false }, hasAttribute: name => name === "data-expand-activity" && "expandActivity" in dataset }) } });
  try {
    await import(`../dist/app.js?controller-test=${Date.now()}`);
    assert.match(get("content").innerHTML, /Teacher assessment/);
    assert.match(get("content").innerHTML, /data-round="a1"/);
    assert.equal(get("run-toolbar").hidden, false);
    assert.ok(!requests.some(r => r.path.startsWith("/api/reports")));
    const beforeModel = requests.length;
    await click({ view: "model" });
    assert.match(get("content").innerHTML, /Message the model/);
    assert.equal(get("run-toolbar").hidden, true);
    assert.deepEqual(requests.slice(beforeModel).map(r => r.path), ["/api/campaigns", "/api/model"]);
    await click({ view: "overview" });
    assert.match(get("content").innerHTML, /Teacher assessment/);
    const homeRequests = requests.length;
    await click({ view: "reports" });
    assert.match(get("content").innerHTML, /Current execution/);
    assert.ok(!/Teacher tokens|Tokens trained|usage-total/.test(get("content").innerHTML));
    assert.ok(requests.slice(homeRequests).every(r => !/telemetry|benchmark/.test(r.path)));
    assert.match(get("content").innerHTML, /Fixture agent narrative/);
    assert.equal(get("run-toolbar").hidden, true);
    await click({ reportPage: "1" });
    assert.match(get("content").innerHTML, /Latest reports/);
    await click({ reportPage: "latest" });
    assert.match(get("content").innerHTML, /Fixture agent narrative/);
    await click({ view: "overview" });
    assert.match(get("content").innerHTML, /data-round="a1"/);
    for (const title of ["Teacher tokens", "Tokens trained", "Teacher assessment", "Independent eval"]) assert.ok(get("content").innerHTML.includes(title), title);
    assert.ok(requests.some(r => /telemetry/.test(r.path)));
    assert.ok(requests.some(r => /benchmark/.test(r.path)));
    failUsage = true;
    await click({ studioSection: "overview" });
    assert.match(get("content").innerHTML, /Usage unavailable/);
    assert.match(get("content").innerHTML, /Evaluation unavailable/);
    assert.match(get("content").innerHTML, /data-round="a1"/);
    assert.equal((get("content").innerHTML.match(/class="panel usage-card /g) || []).length, 5);
    failUsage = false;
    const beforeExperiments = requests.length;
    await click({ studioSection: "experiments" });
    assert.match(get("content").innerHTML, /No teaching experiments recorded/);
    assert.ok(requests.slice(beforeExperiments).every(r => !/telemetry|benchmark/.test(r.path)));
    await click({ studioSection: "teaching" });
    await click({ round: "a1" });
    assert.match(get("content").innerHTML, /Attempt a1/);
    assert.match(get("content").innerHTML, /Revision a1/);
    assert.match(get("content").innerHTML, /Feedback a1/);
    assert.equal(registered.get("get_loop_status").execute().round, 1);
    const beforeHistory = requests.length;
    await click({ view: "history" });
    assert.deepEqual(requests.slice(beforeHistory).map(r => r.path), ["/api/campaigns"]);
    assert.match(get("content").innerHTML, /Previous runs/);
    assert.match(get("content").innerHTML, /data-campaign="b"/);
    assert.equal(get("run-toolbar").hidden, true);
    await click({ campaign: "a" });
    deferOld = true;
    const stale = click({ round: "a1" });
    await click({ campaign: "b" });
    finishOld();
    await stale;
    assert.equal(registered.get("get_loop_status").execute().campaign, "Run b");
    assert.match(get("content").innerHTML, /data-round="b1"/);
    assert.ok(!get("content").innerHTML.includes("Attempt a1"));
    assert.equal(get("run-toolbar").hidden, false);
    assert.ok(requests.every(request => request.method === "GET"));
  } finally {
    for (const [name, descriptor] of original) { if (descriptor) Object.defineProperty(globalThis, name, descriptor); else delete globalThis[name]; }
  }
});
