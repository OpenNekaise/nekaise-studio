import { createModelChat } from "./model-chat.js?v=ebb687a0fdd7";
import { escapeHTML as e, time, modelLabel, readAPIResponse } from "./lib.js?v=ebb687a0fdd7";
import { overview, iterationView, historyView, reportsView, emptyStudio, badge, recoveryNotice, studioScope, studioNavigation } from "./views.js?v=ebb687a0fdd7";
import { elapsedLabel } from "./telemetry.js?v=ebb687a0fdd7";

import { createAssessmentHistory } from "./teacher-assessment.js?v=ebb687a0fdd7";
import { createBenchmarkBrowser } from "./benchmark-browser.js?v=ebb687a0fdd7";
import { createExperimentBrowser } from "./experiment-browser.js?v=ebb687a0fdd7";

const $ = id => document.getElementById(id);
const state = {
  campaignId: "",
  selectedRound: "", view: "overview", lessonId: "", diff: false,
  data: null, campaigns: [], iterations: {}, snapshots: {},
  signature: "", headerSignature: "", contentScope: "", busy: false, refreshing: false,
  expandedActivity: false, selectionVersion: 0,
  reports: null, reportDetail: null, reportId: null, reportBefore: null,
  benchmark: null, gpqa: null, teacherAssessment: null, telemetry: null, telemetryReceived: 0, online: false,
  studioSection: "overview", historyQuery: "", historyFilter: "all",
};
let timer, clockTimer, toastTimer;
const modelChat = createModelChat(api);
const loadAssessmentHistory = createAssessmentHistory(api);
const benchmarkBrowser = createBenchmarkBrowser(api, () => render(true));
const experimentBrowser = createExperimentBrowser(api, () => render());

async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    ...options, headers: { "Content-Type": "application/json", ...options.headers },
    signal: AbortSignal.timeout(15000),
  });
  return readAPIResponse(response);
}
function toast(message) {
  $("toast").textContent = message;
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 5000);
}
function showError(message) {
  $("error-banner").textContent = message;
  $("error-banner").hidden = !message;
}
function connected(ok) {
  state.online = ok;
  $("connection-dot").classList.toggle("offline", !ok);
  $("connection-label").textContent = ok ? "Connected" : "Offline";
}
function renderHeader() {
  const c = state.data?.campaign;
  const signature = JSON.stringify([state.campaigns.map(c => [c.id, c.display_name || c.name, c.retention]), c?.id, c?.status, state.busy, state.view]);
  if (signature === state.headerSignature) return;
  state.headerSignature = signature;
  document.querySelectorAll(".nav-item").forEach(button => {
    const active = button.dataset.view === (state.view === "iteration" ? "overview" : state.view);
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
  });
  $("run-toolbar").hidden = ["history", "reports", "model"].includes(state.view) || !state.campaigns.length;
  $("run-selection").innerHTML = state.campaigns.length ? `<label for="campaign-picker">Run</label><select id="campaign-picker" aria-label="Select run">${state.campaigns.filter(row => row.retention !== "archive" || row.id === state.campaignId).map(row => `<option value="${e(row.id)}" ${row.id === state.campaignId ? "selected" : ""}>${e(row.display_name || row.name)}</option>`).join("")}</select>` : "";
  let actions = c ? badge(c.status) : "";
  if (c?.status === "ready") actions += '<button class="button primary" data-action="start">Start run</button>';
  else if (["running", "queued"].includes(c?.status)) actions += '<button class="button secondary" data-action="pause">Pause</button><button class="button secondary" data-action="stop">Stop</button>';
  else if (c?.status === "waiting") actions += '<button class="button secondary" data-action="pause">Pause recovery</button><button class="button secondary" data-action="stop">Stop</button>';
  else if (c?.status === "recovering") actions += '<button class="button secondary" data-action="stop">Stop recovery</button>';
  else if (c?.status === "complete") actions += '<button class="button primary" data-action="resume">Continue run</button>';
  else if (["paused", "failed", "interrupted", "stopped"].includes(c?.status)) actions += '<button class="button primary" data-action="resume">Resume run</button>';
  else if (c?.status === "pausing") actions += '<button class="button secondary" data-action="stop">Stop now</button>';
  $("campaign-actions").innerHTML = actions;
  document.querySelectorAll("[data-action]").forEach(button => { button.disabled = state.busy; });
}
function currentCampaign() {
  return state.campaigns.find(c => ["running", "queued", "pausing", "stopping", "waiting", "recovering"].includes(c.status)) || state.campaigns[0];
}
function render(force = false) {
  renderHeader();
  if (state.view === "model") {
    modelChat.mount($("content"));
    $("updated-label").textContent = "Latest completed student · Interactive chat";
    state.contentScope = "model";
    state.signature = "";
    return;
  }
  modelChat.unmount();
  const s = state.data;
  const signature = JSON.stringify([s, state.iterations, state.snapshots, state.campaigns, state.view, state.studioSection, state.historyQuery, state.historyFilter, state.selectedRound, state.lessonId, state.diff, state.expandedActivity, state.reports, state.reportDetail, state.reportId, state.reportBefore, state.telemetry, state.benchmark, state.gpqa, state.teacherAssessment, benchmarkBrowser.state, experimentBrowser.state]);
  if (!force && (signature === state.signature || window.getSelection()?.toString())) return;
  state.signature = signature;
  const content = $("content"), detailState = new Map();
  const scope = state.view === "reports" ? `reports:${state.reportId}` : `${state.campaignId}:${state.view}:${state.studioSection}:${state.selectedRound}`;
  const sameScope = scope === state.contentScope;
  state.contentScope = scope;
  if (sameScope) for (const detail of content.querySelectorAll("details[data-detail]")) detailState.set(detail.dataset.detail, detail.open);
  const scrolls = new Map(sameScope ? [...content.querySelectorAll("details[data-detail] .prose")].map((el, index) => [index, el.scrollTop]) : []);
  const listScrolls = new Map([...content.querySelectorAll("[data-scroll]")].map(el => [el.dataset.scroll, el.scrollTop]));
  const focused = document.activeElement;
  const focusKey = focused?.id || focused?.dataset?.benchmarkPoint || focused?.dataset?.benchmarkMetric || focused?.dataset?.experimentRound || focused?.dataset?.experimentSelect || focused?.dataset?.experimentPage || focused?.dataset?.experimentStrategy || focused?.dataset?.scoreRound || focused?.dataset?.report || focused?.dataset?.round || focused?.dataset?.campaign || focused?.dataset?.studioSection || focused?.id;
  const selection = typeof focused?.selectionStart === "number" ? [focused.selectionStart, focused.selectionEnd] : null;
  const data = s ? { ...s, gpqa: state.gpqa, experiments: experimentBrowser.state, currentCampaign: currentCampaign(), iterations: state.iterations, teacherAssessment: state.teacherAssessment?.campaign_id === s.campaign.id ? state.teacherAssessment : null, expandedActivity: state.expandedActivity, benchmark: state.benchmark?.campaign_id === s.campaign.id ? state.benchmark : null, benchmarkBrowser: benchmarkBrowser.state.snapshot?.campaign_id === s.campaign.id ? benchmarkBrowser.state : {}, telemetry: state.telemetry?.campaign_id === s.campaign.id ? state.telemetry : null } : null;
  if (state.view === "reports") content.innerHTML = reportsView(state.reports, state.reportDetail, state.reportId, state.reportBefore);
  else if (state.view === "history") content.innerHTML = historyView(state.campaigns, { query: state.historyQuery, filter: state.historyFilter });
  else if (!s) content.innerHTML = state.campaigns.length ? '<div class="empty-inline" role="status">Loading run…</div>' : emptyStudio();
  else content.innerHTML = (state.view === "iteration" ? recoveryNotice(s) + studioScope(data) + studioNavigation("teaching") + iterationView(data, state) : overview(data, state.studioSection));
  for (const detail of content.querySelectorAll("details[data-detail]")) if (detailState.has(detail.dataset.detail)) detail.open = detailState.get(detail.dataset.detail);
  [...content.querySelectorAll("details[data-detail] .prose")].forEach((el, index) => { if (scrolls.has(index)) el.scrollTop = scrolls.get(index); });
  for (const el of content.querySelectorAll("[data-scroll]")) if (listScrolls.has(el.dataset.scroll)) el.scrollTop = listScrolls.get(el.dataset.scroll);
  if (focusKey) {
    const restored = [...content.querySelectorAll("button,select,input,a[data-score-round],a[data-benchmark-point]")].find(el => (el.id || el.dataset.benchmarkPoint || el.dataset.benchmarkMetric || el.dataset.experimentRound || el.dataset.experimentSelect || el.dataset.experimentPage || el.dataset.experimentStrategy || el.dataset.scoreRound || el.dataset.report || el.dataset.round || el.dataset.campaign || el.dataset.studioSection || el.id) === focusKey);
    restored?.focus({ preventScroll: true });
    if (selection) restored?.setSelectionRange?.(...selection);
  }
  $("updated-label").textContent = state.view === "reports" ? `Observed ${time(state.reports?.current?.observed_at)} · CoAPT Mid-training` : state.view === "history" ? `${state.campaigns.length} recorded runs · Teaching history remains available` : s ? `Updated ${time(s.timestamp)} · ${modelLabel(s.campaign.config.student_model)}` : "";
  updateDuration();
}
function updateDuration() {
  const t = state.telemetry;
  if (!t) return;
  const extra = t.clock_running && state.online ? Math.max(0, (Date.now() - state.telemetryReceived) / 1000) : 0;
  document.querySelectorAll("[data-elapsed]").forEach(el => { if (el.dataset.elapsed === t.campaign_id) el.textContent = elapsedLabel(t.elapsed_seconds + extra); });
}
async function fillTelemetry(id, version) {
  let telemetry;
  try { telemetry = await api(`/campaigns/${encodeURIComponent(id)}/telemetry`); }
  catch { telemetry = { campaign_id: id, unavailable: true }; }
  if (version !== state.selectionVersion) return;
  state.telemetry = telemetry;
  state.telemetryReceived = Date.now();
}
async function fillGPQA(version) {
  let value;
  try { value = await api("/benchmarks/gpqa-diamond"); }
  catch { value = { status: "unavailable", runs: [] }; }
  if (version === state.selectionVersion) state.gpqa = value;
}
async function fillBenchmark(id, version) {
  let evaluation;
  try { evaluation = await api(`/campaigns/${encodeURIComponent(id)}/benchmark`); }
  catch { evaluation = { campaign_id: id, status: "unavailable" }; }
  if (version === state.selectionVersion) { state.benchmark = evaluation; benchmarkBrowser.observe(evaluation); }
}
function mergeIteration(prior, next) {
  if (prior?.updated_at === next.updated_at && prior?.material_count === next.material_count && (prior?.materials?.length || 0) > (next.materials?.length || 0)) {
    return { ...prior, ...next, materials: prior.materials, material_next_offset: prior.material_next_offset };
  }
  return { ...prior, ...next };
}
function remember(snapshot) {
  state.snapshots[snapshot.campaign.id] = snapshot;
  if (snapshot.round) state.iterations[snapshot.round.id] = mergeIteration(state.iterations[snapshot.round.id], snapshot.round);
}
async function fillIterations(snapshot, version) {
  const missing = snapshot.rounds.filter(round => !state.iterations[round.id] || state.iterations[round.id].updated_at !== round.updated_at);
  for (let start = 0; start < missing.length; start += 4) {
    const results = await Promise.allSettled(missing.slice(start, start + 4).map(round => api(`/rounds/${encodeURIComponent(round.id)}`)));
    if (version !== state.selectionVersion) return;
    for (const result of results) if (result.status === "fulfilled") state.iterations[result.value.id] = mergeIteration(state.iterations[result.value.id], result.value);
    render();
  }
}
async function refresh(force = false) {
  if (state.refreshing && !force) return;
  const version = state.selectionVersion;
  state.refreshing = true;
  try {
    const campaigns = await api("/campaigns");
    if (version !== state.selectionVersion) return;
    const selected = state.data?.campaign;
    if (selected?.id === state.campaignId && !campaigns.some(c => c.id === selected.id)) campaigns.push(selected);
    state.campaigns = campaigns;
    if (!campaigns.some(c => c.id === state.campaignId)) state.campaignId = currentCampaign()?.id || "";
    if (state.view === "model") {
      render(); await modelChat.refresh(); connected(true); showError(""); return;
    }
    if (state.view === "history") {
      connected(true); showError(""); render(force);
      return;
    }
    if (state.view === "reports") {
      await fillReports(version);
      if (version === state.selectionVersion) { connected(true); showError(""); }
      return;
    }
    if (state.campaignId) {
      const id = state.campaignId;
      const [snapshot] = await Promise.all([api(`/campaigns/${encodeURIComponent(id)}`), ...(state.studioSection === "overview" && state.view !== "iteration" ? [fillTelemetry(id, version), fillBenchmark(id, version), fillGPQA(version)] : [])]);
      if (version !== state.selectionVersion || id !== state.campaignId) return;
      state.data = snapshot;
      remember(snapshot);
      localStorage.setItem("nekaise.campaign", id);
      if (state.view === "iteration" && !state.selectedRound) state.selectedRound = snapshot.round?.id || "";
      render(force);
      if (state.studioSection === "experiments" && state.view === "overview") await experimentBrowser.refresh();
      if (["teaching", "overview"].includes(state.studioSection)) await Promise.all([
        fillIterations(snapshot, version),
        state.studioSection === "overview" && state.view === "overview" ? loadAssessmentHistory({ ...snapshot, rounds: snapshot.rounds.map(r => state.iterations[r.id]?.updated_at === r.updated_at ? state.iterations[r.id] : r) }, campaigns,
          () => version === state.selectionVersion,
          data => { state.teacherAssessment = data; render(); }) : Promise.resolve(),
      ]);
      if (version !== state.selectionVersion) return;
      showError(state.view === "history" || ["waiting", "recovering"].includes(snapshot.campaign.status) ? "" : snapshot.campaign.error || "");
    } else { state.data = null; render(force); }
    connected(true);
  } catch (error) {
    if (version === state.selectionVersion) { connected(false); showError(`Unable to refresh: ${error.message}`); }
  } finally { if (version === state.selectionVersion) state.refreshing = false; }
}
async function changeCampaign(id) {
  if (!state.campaigns.some(c => c.id === id)) return;
  state.selectionVersion += 1;
  state.refreshing = false;
  state.campaignId = id;
  state.selectedRound = "";
  state.lessonId = "";
  state.view = "overview";
  state.studioSection = "overview";
  state.data = state.snapshots[id] || null;
  state.expandedActivity = false;
  state.signature = "";
  showError(""); render(true); window.scrollTo?.({ top: 0, behavior: "instant" });
  await refresh(true);
}
async function navigate(view) {
  if (!["overview", "history", "reports", "model"].includes(view)) return;
  if (view === "overview" && state.view === "reports" && state.reports?.current?.campaign?.id) {
    state.campaignId = state.reports.current.campaign.id;
    state.data = state.snapshots[state.campaignId] || null;
  }
  state.selectionVersion += 1;
  state.refreshing = false;
  state.view = view;
  if (view === "overview") state.studioSection = "overview";
  state.selectedRound = "";
  state.lessonId = "";
  state.signature = "";
  showError(""); render(true); window.scrollTo?.({ top: 0, behavior: "instant" });
  await refresh(true);
}
async function fillReports(version) {
  const before = state.reportBefore;
  const reports = await api(`/reports${before ? `?before=${before}` : ""}`);
  if (version !== state.selectionVersion || state.view !== "reports" || before !== state.reportBefore) return;
  state.reports = reports;
  if (version !== state.selectionVersion || state.view !== "reports") return;
  if (!reports.items.some(r => r.id === state.reportId)) { state.reportId = reports.items[0]?.id || null; state.reportDetail = null; }
  render();
  const id = state.reportId;
  if (id) {
    const detail = await api(`/reports/${id}`);
    if (version !== state.selectionVersion || state.view !== "reports" || id !== state.reportId) return;
    state.reportDetail = detail;
    render();
  }
}
async function openReport(id, before) {
  state.selectionVersion += 1;
  state.refreshing = false;
  state.reportId = id;
  if (before !== undefined) state.reportBefore = before;
  state.reportDetail = null;
  render(true); window.scrollTo?.({ top: 0, behavior: "instant" });
  await refresh(true);
}
async function openIteration(id, lessonId = "", assessment = false) {
  if (!state.data) return;
  const version = state.selectionVersion, campaignId = state.campaignId;
  state.selectedRound = id;
  state.lessonId = lessonId;
  state.view = "iteration";
  state.studioSection = "teaching";
  render(true); window.scrollTo?.({ top: 0, behavior: "instant" });
  try {
    const snapshot = await api(`/campaigns/${encodeURIComponent(campaignId)}?round_id=${encodeURIComponent(id)}`);
    if (version !== state.selectionVersion || state.selectedRound !== id || state.view !== "iteration") return;
    state.iterations[id] = snapshot.round;
    if (!state.data.rounds.some(round => round.id === id)) state.data.rounds.push(snapshot.round);
    render(true);
    if (assessment) document.querySelector?.(".iteration-assessment")?.scrollIntoView({ block: "start" });
  } catch (error) { if (version === state.selectionVersion && state.selectedRound === id) showError(`Unable to open iteration: ${error.message}`); }
}
async function openRecordedIteration(campaignId, roundId, assessment = false) {
  // Records can belong to campaigns older than the run picker's current page.
  if (!state.campaigns.some(c => c.id === campaignId)) {
    const version = state.selectionVersion;
    try {
      const snapshot = await api(`/campaigns/${encodeURIComponent(campaignId)}`);
      if (version !== state.selectionVersion) return;
      state.campaigns.push(snapshot.campaign); remember(snapshot);
    } catch (error) { return showError(`Unable to open iteration: ${error.message}`); }
  }
  const version = state.selectionVersion + (campaignId !== state.campaignId ? 1 : 0);
  if (campaignId !== state.campaignId) await changeCampaign(campaignId);
  if (state.selectionVersion !== version) return;
  if (state.campaignId === campaignId) return openIteration(roundId, "", assessment);
}
async function changeStudioSection(section) {
  if (!["overview", "teaching", "experiments", "activity"].includes(section)) return;
  state.selectionVersion += 1;
  state.refreshing = false;
  state.studioSection = section;
  state.view = "overview";
  state.selectedRound = "";
  state.lessonId = "";
  showError(""); render(true); window.scrollTo?.({ top: 0, behavior: "instant" });
  await refresh(true);
}
function openCreate() {
  $("form-error").hidden = true;
  const config = state.data?.campaign.config;
  if (config) {
    $("campaign-form").elements.teacher_provider.value = config.teacher_provider;
    $("campaign-form").elements.teacher_model.value = config.teacher_model;
  }
  $("campaign-dialog").showModal();
}
async function action(verb) {
  if (state.busy || !state.campaignId) return;
  state.busy = true;
  renderHeader();
  try {
    const result = await api(`/campaigns/${encodeURIComponent(state.campaignId)}/actions`, { method: "POST", body: JSON.stringify({ action: verb }) });
    if (result.campaign_id && result.campaign_id !== state.campaignId) {
      state.campaigns = await api("/campaigns");
      await changeCampaign(result.campaign_id);
    }
    toast(verb === "pause" ? "Pause requested. The current stage will finish first." : verb === "stop" ? "Stop requested. Completed work is saved." : `${verb === "resume" ? "Resume" : "Start"} requested.`);
    await refresh(true);
  } catch (error) { showError(error.message); }
  finally { state.busy = false; renderHeader(); }
}
async function showSystem() {
  $("system-dialog").showModal();
  $("system-content").textContent = "Checking…";
  try {
    const data = await api("/system");
    const labels = { model_python: "Training environment", claude: "Claude Code", codex: "Codex CLI", student_cache: "Student model cache", corpus: "Corpus", worker: "Loop worker", storage: "Storage" };
    $("system-content").innerHTML = `<p class="quiet">${e(data.training || "CoAPT Mid-training")}</p>` + Object.entries(labels).map(([key, label]) => `<div class="system-row"><div>${label}${data[key].path ? `<small>${e(data[key].path)}</small>` : ""}</div>${badge(data[key].available ? "ready" : key === "worker" ? "idle" : "missing", true)}</div>`).join("");
  } catch (error) { $("system-content").textContent = error.message; }
}

document.addEventListener("click", async event => {
  const benchmarkPoint = event.target.closest("[data-benchmark-point]");
  if (benchmarkPoint?.dataset?.benchmarkPoint && state.view === "overview" && state.studioSection === "overview") {
    event.preventDefault();
    benchmarkBrowser.select(benchmarkPoint.dataset.benchmarkPoint);
    return;
  }
  const score = event.target.closest("[data-score-round]");
  if (score?.dataset?.scoreRound) {
    event.preventDefault();
    return openRecordedIteration(score.dataset.scoreCampaign, score.dataset.scoreRound, true);
  }
  const target = event.target.closest("button");
  if (!target) return;
  if (state.view === "overview" && state.studioSection === "overview") {
    if ("benchmarkOpen" in target.dataset) return benchmarkBrowser.open(state.benchmark);
    if ("benchmarkClose" in target.dataset) return benchmarkBrowser.close();
    if ("benchmarkRefresh" in target.dataset) return benchmarkBrowser.refresh();
    if ("benchmarkOverview" in target.dataset) return benchmarkBrowser.overview();
    if ("benchmarkPage" in target.dataset) return benchmarkBrowser.load(Number(target.dataset.benchmarkPage));
    if ("benchmarkMetric" in target.dataset) return benchmarkBrowser.metric(target.dataset.benchmarkMetric);
  }
  if (target.dataset.experimentRound) return openRecordedIteration(target.dataset.experimentCampaign, target.dataset.experimentRound);
  if ("experimentStrategy" in target.dataset) {
    const reading = experimentBrowser.filter(target.dataset.experimentStrategy);
    if (state.studioSection !== "experiments" || state.view !== "overview") await changeStudioSection("experiments");
    return reading;
  }
  if (state.view === "overview" && state.studioSection === "experiments") {
    if (target.dataset.experimentSelect) return experimentBrowser.select(target.dataset.experimentSelect);
    if (target.dataset.experimentPage) return experimentBrowser.load(target.dataset.experimentPage === "latest" ? null : Number(target.dataset.experimentPage));
    if ("experimentRefresh" in target.dataset) return experimentBrowser.refresh();
  }
  if (target.dataset.materialNext) {
    const id = target.dataset.materialNext, before = state.iterations[id], version = state.selectionVersion;
    if (!before || before.material_next_offset == null) return;
    target.disabled = true;
    try {
      const page = await api(`/rounds/${encodeURIComponent(id)}/materials?offset=${before.material_next_offset}`);
      if (version !== state.selectionVersion || state.iterations[id]?.updated_at !== before.updated_at) return;
      state.iterations[id] = { ...state.iterations[id], materials: [...new Map([...(before.materials || []), ...page.rows].map(row => [row.id, row])).values()], material_count: page.total, material_next_offset: page.next_offset };
      render(true);
    } catch (error) { showError(error.message); }
    finally { target.disabled = false; }
    return;
  }
  if (["new-campaign", "empty-create"].includes(target.id)) return openCreate();
  if (target.classList.contains("close-modal")) return target.closest("dialog").close();
  if (target.dataset.view) return navigate(target.dataset.view);
  if (target.dataset.studioSection) return changeStudioSection(target.dataset.studioSection);
  if (target.dataset.report) return openReport(Number(target.dataset.report));
  if (target.dataset.reportPage) return openReport(null, target.dataset.reportPage === "latest" ? null : Number(target.dataset.reportPage));
  if (target.dataset.campaign) return changeCampaign(target.dataset.campaign);
  if (target.dataset.round) return openIteration(target.dataset.round);
  if (target.dataset.action) return action(target.dataset.action);
  if (target.dataset.diff) { state.diff = target.dataset.diff === "true"; return render(true); }
  if (target.hasAttribute("data-expand-activity")) { state.expandedActivity = !state.expandedActivity; return render(true); }
  if (target.id === "system-button") return showSystem();
});
document.addEventListener("change", event => {
  if (event.target.id === "history-filter") { state.historyFilter = event.target.value; return render(true); }
  if (event.target.id === "campaign-picker") return changeCampaign(event.target.value);
  if (event.target.id === "round-picker") return openIteration(event.target.value);
  if (event.target.name === "teacher_provider") $("campaign-form").elements.teacher_model.value = event.target.value === "claude" ? "claude-fable-5-1" : "gpt-5.6-terra";
  if (event.target.name === "orchestrator_provider") $("campaign-form").elements.orchestrator_model.value = event.target.value === "claude" ? "claude-fable-5-1" : "gpt-6-astra";
});
document.addEventListener("input", event => {
  if (event.target.id === "history-search") { state.historyQuery = event.target.value; render(true); }
});
$("campaign-form").addEventListener("submit", async event => {
  event.preventDefault();
  const form = event.currentTarget, button = form.querySelector("[type=submit]");
  button.disabled = true; $("form-error").hidden = true;
  try {
    const values = Object.fromEntries(new FormData(form)), name = values.name;
    delete values.name;
    for (const key of ["rounds", "lessons_per_round", "train_steps", "learning_rate", "max_seq_len", "train_epochs", "tokens_per_update"]) values[key] = Number(values[key]);
    values.token_mix = Object.fromEntries(["teacher", "corpus", "replay"].map(key => [key, Number(values[`mix_${key}`])]));
    for (const key of ["teacher", "corpus", "replay"]) delete values[`mix_${key}`];
    const campaign = await api("/campaigns", { method: "POST", body: JSON.stringify({ name, config: values }) });
    $("campaign-dialog").close();
    state.campaigns = await api("/campaigns");
    await changeCampaign(campaign.id);
    toast("Run created.");
  } catch (error) { $("form-error").textContent = error.message; $("form-error").hidden = false; }
  finally { button.disabled = false; }
});
document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });
await refresh();
timer = setInterval(() => { if (!document.hidden) refresh(); }, 2000);
clockTimer = setInterval(() => { if (!document.hidden) updateDuration(); }, 1000);
window.addEventListener("pagehide", () => { clearInterval(timer); clearInterval(clockTimer); modelChat.unmount(); });

if (document.modelContext?.registerTool) {
  const lifecycle = new AbortController();
  const register = tool => Promise.resolve(document.modelContext.registerTool(tool, { signal: lifecycle.signal })).catch(() => {});
  register({
    name: "get_loop_status", description: "Read the displayed run and iteration.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true, untrustedContentHint: true },
    execute: () => {
      const shown = state.view === "reports" ? state.reports?.current : state.data;
      const round = state.view === "reports" ? shown?.round : state.iterations[state.selectedRound] || shown?.round;
      return { campaign: shown?.campaign?.name, status: shown?.campaign?.status, round: round?.number, stage: round?.stage };
    },
  });
  register({
    name: "inspect_lesson", description: "Open a recorded lesson alongside its iteration's assessment.",
    inputSchema: { type: "object", properties: { lesson_id: { type: "string" } }, required: ["lesson_id"], additionalProperties: false },
    annotations: { readOnlyHint: true, untrustedContentHint: true },
    execute: async input => {
      const round = state.iterations[state.selectedRound] || state.data?.round;
      const lesson = round?.lessons.find(row => row.id === input.lesson_id);
      if (!lesson) throw new Error("Lesson not found in the selected iteration");
      await openIteration(round.id, lesson.id);
      return { id: lesson.id, concept: lesson.concept, gate: lesson.gate };
    },
  });
  window.addEventListener("pagehide", () => lifecycle.abort());
}
