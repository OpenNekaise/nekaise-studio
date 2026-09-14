import { escapeHTML as e, time, modelLabel } from "./lib.js";
import { overview, iterationView, historyView, emptyStudio, badge, recoveryNotice } from "./views.js";

const $ = id => document.getElementById(id);
const state = {
  campaignId: localStorage.getItem("nekaise.campaign") || "",
  selectedRound: "", view: "overview", lessonId: "", diff: false,
  data: null, campaigns: [], iterations: {}, snapshots: {},
  signature: "", headerSignature: "", contentScope: "", busy: false, refreshing: false,
  expandedActivity: false, selectionVersion: 0,
};
let timer, toastTimer;

async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    ...options, headers: { "Content-Type": "application/json", ...options.headers },
    signal: AbortSignal.timeout(15000),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data));
  return data;
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
  $("connection-dot").classList.toggle("offline", !ok);
  $("connection-label").textContent = ok ? "Connected" : "Offline";
}
function renderHeader() {
  const c = state.data?.campaign;
  const signature = JSON.stringify([state.campaigns.map(c => [c.id, c.name]), c?.id, c?.status, state.busy, state.view]);
  if (signature === state.headerSignature) return;
  state.headerSignature = signature;
  document.querySelectorAll(".nav-item").forEach(button => {
    const active = button.dataset.view === "history" ? state.view === "history" : state.view !== "history";
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
  });
  $("run-toolbar").hidden = state.view === "history" || !state.campaigns.length;
  $("run-selection").innerHTML = state.campaigns.length ? `<label for="campaign-picker">Run</label><select id="campaign-picker" aria-label="Select run">${state.campaigns.map(row => `<option value="${e(row.id)}" ${row.id === state.campaignId ? "selected" : ""}>${e(row.name)}</option>`).join("")}</select>` : "";
  let actions = c ? badge(c.status) : "";
  if (c?.status === "ready") actions += '<button class="button primary" data-action="start">Start run</button>';
  else if (["running", "queued"].includes(c?.status)) actions += '<button class="button secondary" data-action="pause">Pause</button><button class="button secondary" data-action="stop">Stop</button>';
  else if (c?.status === "waiting") actions += '<button class="button primary" data-action="resume">Resume now</button><button class="button secondary" data-action="pause">Pause recovery</button><button class="button secondary" data-action="stop">Stop</button>';
  else if (c?.status === "recovering") actions += '<button class="button secondary" data-action="stop">Stop recovery</button>';
  else if (c?.status === "complete") actions += '<button class="button primary" data-action="resume">Continue run</button>';
  else if (["paused", "failed", "interrupted", "stopped"].includes(c?.status)) actions += '<button class="button primary" data-action="resume">Resume run</button>';
  else if (c?.status === "pausing") actions += '<button class="button secondary" data-action="stop">Stop now</button>';
  $("campaign-actions").innerHTML = actions;
  document.querySelectorAll("[data-action]").forEach(button => { button.disabled = state.busy; });
}
function render(force = false) {
  renderHeader();
  const s = state.data;
  const signature = JSON.stringify([s, state.iterations, state.snapshots, state.campaigns, state.view, state.selectedRound, state.lessonId, state.diff, state.expandedActivity]);
  if (!force && (signature === state.signature || window.getSelection()?.toString())) return;
  state.signature = signature;
  const content = $("content"), detailState = new Map();
  const scope = `${state.campaignId}:${state.view}:${state.selectedRound}`;
  const sameScope = scope === state.contentScope;
  state.contentScope = scope;
  if (sameScope) for (const detail of content.querySelectorAll("details[data-detail]")) detailState.set(detail.dataset.detail, detail.open);
  const scrolls = new Map(sameScope ? [...content.querySelectorAll("details[data-detail] .prose")].map((el, index) => [index, el.scrollTop]) : []);
  const focused = document.activeElement;
  const focusKey = focused?.dataset?.round || focused?.dataset?.campaign || focused?.id;
  const data = s ? { ...s, iterations: state.iterations, expandedActivity: state.expandedActivity } : null;
  if (state.view === "history") content.innerHTML = historyView(state.campaigns, state.snapshots);
  else if (!s) content.innerHTML = state.campaigns.length ? '<div class="empty-inline" role="status">Loading run…</div>' : emptyStudio();
  else content.innerHTML = recoveryNotice(s) + (state.view === "iteration" ? iterationView(data, state) : overview(data));
  for (const detail of content.querySelectorAll("details[data-detail]")) if (detailState.has(detail.dataset.detail)) detail.open = detailState.get(detail.dataset.detail);
  [...content.querySelectorAll("details[data-detail] .prose")].forEach((el, index) => { if (scrolls.has(index)) el.scrollTop = scrolls.get(index); });
  if (focusKey) [...content.querySelectorAll("button,select")].find(el => (el.dataset.round || el.dataset.campaign || el.id) === focusKey)?.focus({ preventScroll: true });
  $("updated-label").textContent = s ? `Updated ${time(s.timestamp)} · ${modelLabel(s.campaign.config.student_model)}` : "";
}
function remember(snapshot) {
  state.snapshots[snapshot.campaign.id] = snapshot;
  if (snapshot.round) state.iterations[snapshot.round.id] = { ...state.iterations[snapshot.round.id], ...snapshot.round };
}
async function fillIterations(snapshot, version) {
  const missing = snapshot.rounds.filter(round => !state.iterations[round.id] || state.iterations[round.id].updated_at !== round.updated_at);
  for (let start = 0; start < missing.length; start += 4) {
    const results = await Promise.allSettled(missing.slice(start, start + 4).map(round => api(`/rounds/${encodeURIComponent(round.id)}`)));
    if (version !== state.selectionVersion) return;
    for (const result of results) if (result.status === "fulfilled") state.iterations[result.value.id] = { ...state.iterations[result.value.id], ...result.value };
    render();
  }
}
async function fillHistory(version) {
  const missing = state.campaigns.filter(c => !state.snapshots[c.id] || state.snapshots[c.id].campaign.updated_at !== c.updated_at || ["running", "queued", "pausing", "stopping"].includes(c.status));
  for (let start = 0; start < missing.length; start += 4) {
    const results = await Promise.allSettled(missing.slice(start, start + 4).map(c => api(`/campaigns/${encodeURIComponent(c.id)}`)));
    if (version !== state.selectionVersion || state.view !== "history") return;
    for (const result of results) if (result.status === "fulfilled") remember(result.value);
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
    state.campaigns = campaigns;
    if (!campaigns.some(c => c.id === state.campaignId)) state.campaignId = campaigns[0]?.id || "";
    if (state.campaignId) {
      const id = state.campaignId;
      const snapshot = await api(`/campaigns/${encodeURIComponent(id)}`);
      if (version !== state.selectionVersion || id !== state.campaignId) return;
      state.data = snapshot;
      remember(snapshot);
      localStorage.setItem("nekaise.campaign", id);
      if (state.view === "iteration" && !state.selectedRound) state.selectedRound = snapshot.round?.id || "";
      render(force);
      await fillIterations(snapshot, version);
      if (state.view === "history") await fillHistory(version);
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
  state.data = state.snapshots[id] || null;
  state.expandedActivity = false;
  state.signature = "";
  showError(""); render(true);
  await refresh(true);
}
async function navigate(view) {
  if (!["overview", "history"].includes(view)) return;
  state.view = view;
  state.selectedRound = "";
  state.lessonId = "";
  state.signature = "";
  showError(""); render(true);
  if (view === "history") await fillHistory(state.selectionVersion);
}
async function openIteration(id, lessonId = "") {
  if (!state.data) return;
  const version = state.selectionVersion, campaignId = state.campaignId;
  state.selectedRound = id;
  state.lessonId = lessonId;
  state.view = "iteration";
  render(true);
  try {
    const snapshot = await api(`/campaigns/${encodeURIComponent(campaignId)}?round_id=${encodeURIComponent(id)}`);
    if (version !== state.selectionVersion || state.selectedRound !== id || state.view !== "iteration") return;
    state.iterations[id] = snapshot.round;
    if (!state.data.rounds.some(round => round.id === id)) state.data.rounds.push(snapshot.round);
    render(true);
  } catch (error) { if (version === state.selectionVersion && state.selectedRound === id) showError(`Unable to open iteration: ${error.message}`); }
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
    $("system-content").innerHTML = Object.entries(labels).map(([key, label]) => `<div class="system-row"><div>${label}${data[key].path ? `<small>${e(data[key].path)}</small>` : ""}</div>${badge(data[key].available ? "ready" : key === "worker" ? "idle" : "missing", true)}</div>`).join("");
  } catch (error) { $("system-content").textContent = error.message; }
}

document.addEventListener("click", async event => {
  const target = event.target.closest("button");
  if (!target) return;
  if (["new-campaign", "empty-create"].includes(target.id)) return openCreate();
  if (target.classList.contains("close-modal")) return target.closest("dialog").close();
  if (target.dataset.view) return navigate(target.dataset.view);
  if (target.dataset.campaign) return changeCampaign(target.dataset.campaign);
  if (target.dataset.round) return openIteration(target.dataset.round);
  if (target.dataset.action) return action(target.dataset.action);
  if (target.dataset.diff) { state.diff = target.dataset.diff === "true"; return render(true); }
  if (target.hasAttribute("data-expand-activity")) { state.expandedActivity = !state.expandedActivity; return render(true); }
  if (target.id === "system-button") return showSystem();
});
document.addEventListener("change", event => {
  if (event.target.id === "campaign-picker") return changeCampaign(event.target.value);
  if (event.target.id === "round-picker") return openIteration(event.target.value);
  if (event.target.name === "teacher_provider") $("campaign-form").elements.teacher_model.value = event.target.value === "claude" ? "claude-fable-5-1" : "gpt-5.6-terra";
  if (event.target.name === "orchestrator_provider") $("campaign-form").elements.orchestrator_model.value = event.target.value === "claude" ? "claude-fable-5-1" : "gpt-6-astra";
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
window.addEventListener("pagehide", () => clearInterval(timer));

if (document.modelContext?.registerTool) {
  const lifecycle = new AbortController();
  const register = tool => Promise.resolve(document.modelContext.registerTool(tool, { signal: lifecycle.signal })).catch(() => {});
  register({
    name: "get_loop_status", description: "Read the displayed run and iteration.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true, untrustedContentHint: true },
    execute: () => {
      const round = state.iterations[state.selectedRound] || state.data?.round;
      return { campaign: state.data?.campaign.name, status: state.data?.campaign.status, round: round?.number, stage: round?.stage };
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
