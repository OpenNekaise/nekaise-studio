import { escapeHTML as e, number, time } from "./lib.js";
import {
  overview,
  lessonsView,
  evaluationView,
  activityView,
  badge,
  recoveryNotice,
} from "./views.js";

const $ = (id) => document.getElementById(id);
const state = {
  campaignId: localStorage.getItem("nekaise.campaign") || "",
  selectedRound: "",
  view: "overview",
  lessonId: "",
  diff: false,
  data: null,
  campaigns: [],
  signature: "",
  busy: false,
};
const headings = {
  overview: [
    "THE LEARNING LOOP",
    "A little better, every round.",
    "Follow the lessons, the revisions, and what your student learns next.",
  ],
  lessons: [
    "INSIDE THE TEACHING ROOM",
    "From attempt to understanding.",
    "Read what the student wrote, what the teacher changed, and the evidence behind it.",
  ],
  evaluation: [
    "ONLINE EVALUATION",
    "Find the next thing to learn.",
    "Fresh questions reveal gaps and shape the next round’s curriculum.",
  ],
  activity: [
    "THE ROUND JOURNAL",
    "A clear record of the work.",
    "Every stage, every checkpoint, and the decisions that connect them.",
  ],
};
let timer, toastTimer;

async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
    signal: AbortSignal.timeout(15000),
  });
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail || data),
    );
  return data;
}
function toast(message) {
  $("toast").textContent = message;
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ($("toast").hidden = true), 5000);
}
function showError(message) {
  $("error-banner").textContent = message;
  $("error-banner").hidden = !message;
}
function connected(ok) {
  $("connection-dot").classList.toggle("offline", !ok);
  $("connection-label").textContent = ok
    ? "Connected to studio"
    : "Connection interrupted";
}

function renderSidebar() {
  $("campaign-list").innerHTML = state.campaigns.length
    ? state.campaigns
        .map(
          (c) =>
            `<button class="campaign-link ${c.id === state.campaignId ? "selected" : ""}" data-campaign="${e(c.id)}"><span class="dot ${c.status === "running" ? "running-pulse" : ""}"></span>${e(c.name)}</button>`,
        )
        .join("")
    : '<p class="rail-muted">No campaigns yet</p>';
  const c = state.data?.campaign;
  $("campaign-breadcrumb").innerHTML = state.campaigns.length
    ? `<select id="campaign-picker" class="topbar-campaign" aria-label="Select campaign">${state.campaigns.map((row) => `<option value="${e(row.id)}" ${row.id === state.campaignId ? "selected" : ""}>${e(row.name)}</option>`).join("")}</select>`
    : "Your workspace";
  const [eyebrow, title, description] = headings[state.view];
  $("page-eyebrow").textContent = eyebrow;
  $("page-title").textContent = title;
  $("page-description").textContent = description;
  document.querySelectorAll(".nav-item").forEach((b) => {
    b.classList.toggle("active", b.dataset.view === state.view);
    b.setAttribute(
      "aria-current",
      b.dataset.view === state.view ? "page" : "false",
    );
  });
  let actions = "";
  if (c) {
    actions += badge(c.status);
    if (c.status === "ready")
      actions +=
        '<button class="button primary" data-action="start">▷ Start loop</button>';
    else if (["running", "queued"].includes(c.status))
      actions +=
        '<button class="button secondary" data-action="pause" title="Finish the current stage, then pause">Ⅱ Pause</button><button class="button secondary" data-action="stop" title="Stop the current stage; resume will retry it">■ Stop</button>';
    else if (c.status === "waiting")
      actions += '<button class="button primary" data-action="resume">▷ Resume now</button><button class="button secondary" data-action="pause">Ⅱ Pause recovery</button><button class="button secondary" data-action="stop">■ Stop</button>';
    else if (c.status === "recovering")
      actions += '<button class="button secondary" data-action="stop">■ Stop recovery</button>';
    else if (c.status === "complete")
      actions += '<button class="button primary" data-action="resume">▷ Continue learning</button>';
    else if (["paused", "failed", "interrupted", "stopped"].includes(c.status))
      actions +=
        '<button class="button primary" data-action="resume">▷ Resume loop</button>';
    else if (c.status === "pausing")
      actions +=
        '<button class="button secondary" data-action="stop">■ Stop now</button>';
  }
  actions +=
    '<button class="button secondary" id="new-campaign" aria-label="New campaign">＋ New</button>';
  $("campaign-actions").innerHTML = actions;
  if (state.busy)
    document
      .querySelectorAll("[data-action]")
      .forEach((b) => (b.disabled = true));
}

function render(force = false) {
  renderSidebar();
  const s = state.data;
  if (!s) return;
  const signature = JSON.stringify([
    s.campaign.status,
    s.round,
    s.events,
    s.teacher_usage,
    s.recovery,
    state.view,
    state.lessonId,
    state.diff,
    state.selectedRound,
  ]);
  if (!force && signature === state.signature) return;
  // Do not destroy a user's text selection while they inspect a generation.
  if (!force && window.getSelection()?.toString()) return;
  state.signature = signature;
  s.selectedRound = state.selectedRound;
  const content = $("content");
  const openDetails = new Set(
    [...content.querySelectorAll("details[open][data-detail]")].map(
      (x) => x.dataset.detail,
    ),
  );
  const closedDetails = new Set(
    [...content.querySelectorAll("details:not([open])[data-detail]")].map(
      (x) => x.dataset.detail,
    ),
  );
  const active = document.activeElement?.id;
  const scrolls = [...content.querySelectorAll(".prose")].map(
    (x) => x.scrollTop,
  );
  content.innerHTML = recoveryNotice(s) + (
    state.view === "overview"
      ? overview(s)
      : state.view === "lessons"
        ? lessonsView(s, state)
        : state.view === "evaluation"
          ? evaluationView(s)
          : activityView(s));
  for (const d of content.querySelectorAll("details[data-detail]")) {
    if (openDetails.has(d.dataset.detail)) d.open = true;
    if (closedDetails.has(d.dataset.detail)) d.open = false;
  }
  [...content.querySelectorAll(".prose")].forEach((el, i) => {
    if (scrolls[i]) el.scrollTop = scrolls[i];
  });
  if (active === "round-picker") $(active)?.focus({ preventScroll: true });
  showError(["waiting", "recovering"].includes(s.campaign.status) ? "" : s.campaign.error || s.round?.error || "");
  $("updated-label").textContent =
    `Updated ${time(s.timestamp)} · ${s.campaign.config.student_model}`;
}

async function refresh(force = false) {
  try {
    const campaigns = await api("/campaigns");
    state.campaigns = campaigns;
    if (!campaigns.some((c) => c.id === state.campaignId))
      state.campaignId = campaigns[0]?.id || "";
    if (state.campaignId) {
      const id = state.campaignId,
        round = state.selectedRound;
      const snapshot = await api(
        `/campaigns/${encodeURIComponent(id)}${round ? `?round_id=${encodeURIComponent(round)}` : ""}`,
      );
      if (id !== state.campaignId || round !== state.selectedRound) return;
      state.data = snapshot;
      localStorage.setItem("nekaise.campaign", id);
      render(force);
    } else renderSidebar();
    connected(true);
  } catch (error) {
    connected(false);
    showError(`Unable to refresh the studio: ${error.message}`);
  }
}

async function changeCampaign(id) {
  state.campaignId = id;
  state.selectedRound = "";
  state.lessonId = "";
  state.signature = "";
  await refresh(true);
}
function navigate(view) {
  if (!headings[view]) return;
  state.view = view;
  state.signature = "";
  render(true);
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
  renderSidebar();
  try {
    const result = await api(`/campaigns/${state.campaignId}/actions`, {
      method: "POST",
      body: JSON.stringify({ action: verb }),
    });
    if (result.campaign_id && result.campaign_id !== state.campaignId)
      await changeCampaign(result.campaign_id);
    toast(
      verb === "pause"
        ? "Pause requested. The current stage will finish first."
        : verb === "stop"
          ? "Stop requested. Completed stages remain saved."
          : `${verb === "resume" ? "Resume" : "Start"} requested.`,
    );
    await refresh(true);
  } catch (error) {
    showError(error.message);
  } finally {
    state.busy = false;
    renderSidebar();
  }
}

document.addEventListener("click", async (event) => {
  const target = event.target.closest("button,[data-lesson],[data-round]");
  if (!target) return;
  if (
    target.id === "new-campaign" ||
    target.id === "new-campaign-small" ||
    target.id === "empty-create"
  )
    return openCreate();
  if (target.classList.contains("close-modal"))
    return target.closest("dialog").close();
  if (target.dataset.view) return navigate(target.dataset.view);
  if (target.dataset.campaign) return changeCampaign(target.dataset.campaign);
  if (target.dataset.action) return action(target.dataset.action);
  if (target.dataset.lesson) {
    state.lessonId = target.dataset.lesson;
    return navigate("lessons");
  }
  if (target.dataset.selectLesson) {
    state.lessonId = target.dataset.selectLesson;
    return render(true);
  }
  if (target.dataset.diff) {
    state.diff = target.dataset.diff === "true";
    return render(true);
  }
  if (target.dataset.round) {
    state.selectedRound = target.dataset.round;
    state.view = "lessons";
    state.lessonId = "";
    return refresh(true);
  }
  if (target.id === "system-button") {
    $("system-dialog").showModal();
    $("system-content").textContent = "Checking local capabilities…";
    try {
      const data = await api("/system");
      const labels = {
        model_python: "Model execution environment",
        claude: "Claude Code",
        codex: "Codex CLI",
        student_cache: "MiniCPM5 1B model cache",
        corpus: "Nekaise corpus",
        worker: "Loop worker",
        storage: "Round storage",
      };
      $("system-content").innerHTML =
        Object.entries(labels)
          .map(
            ([key, label]) =>
              `<div class="system-row"><div>${label}${data[key].path ? `<small>${e(data[key].path)}</small>` : ""}</div>${badge(data[key].available ? "ready" : key === "worker" ? "idle" : "missing", true)}</div>`,
          )
          .join("") +
        `<p class="quiet" style="margin-top:20px">${e(data.training)}. Agentic SFT and OPD are planned extensions.</p>`;
    } catch (error) {
      $("system-content").textContent = error.message;
    }
  }
});

document.addEventListener("change", (event) => {
  if (event.target.id === "campaign-picker")
    return changeCampaign(event.target.value);
  if (event.target.id === "round-picker") {
    state.selectedRound = event.target.value;
    state.lessonId = "";
    refresh(true);
  }
  if (event.target.name === "teacher_provider")
    $("campaign-form").elements.teacher_model.value =
      event.target.value === "claude" ? "claude-fable-5-1" : "gpt-5.6-terra";
  if (event.target.name === "orchestrator_provider")
    $("campaign-form").elements.orchestrator_model.value =
      event.target.value === "claude" ? "claude-fable-5-1" : "gpt-6-astra";
});

$("campaign-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector("[type=submit]");
  button.disabled = true;
  $("form-error").hidden = true;
  try {
    const values = Object.fromEntries(new FormData(form)),
      name = values.name;
    delete values.name;
    for (const key of [
      "rounds",
      "lessons_per_round",
      "train_steps",
      "learning_rate",
      "max_seq_len",
      "train_epochs",
      "tokens_per_update",
    ])
      values[key] = Number(values[key]);
    values.token_mix = Object.fromEntries(["teacher", "corpus", "replay"].map((key) => [key, Number(values[`mix_${key}`])]));
    for (const key of ["teacher", "corpus", "replay"]) delete values[`mix_${key}`];
    const campaign = await api("/campaigns", {
      method: "POST",
      body: JSON.stringify({ name, config: values }),
    });
    $("campaign-dialog").close();
    state.view = "overview";
    await changeCampaign(campaign.id);
    toast("Campaign created. Start the loop when you’re ready.");
  } catch (error) {
    $("form-error").textContent = error.message;
    $("form-error").hidden = false;
  } finally {
    button.disabled = false;
  }
});

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) refresh();
});
await refresh();
timer = setInterval(() => {
  if (!document.hidden) refresh();
}, 2000);
window.addEventListener("pagehide", () => clearInterval(timer));

// Optional browser-agent interface, using exactly the same state as the UI.
if (document.modelContext?.registerTool) {
  const lifecycle = new AbortController();
  const register = (tool) =>
    Promise.resolve(
      document.modelContext.registerTool(tool, { signal: lifecycle.signal }),
    ).catch(() => {});
  register({
    name: "get_loop_status",
    description: "Read the currently displayed campaign and round status.",
    inputSchema: {
      type: "object",
      properties: {},
      additionalProperties: false,
    },
    annotations: { readOnlyHint: true, untrustedContentHint: true },
    execute: () => ({
      campaign: state.data?.campaign.name,
      status: state.data?.campaign.status,
      round: state.data?.round?.number,
      stage: state.data?.round?.stage,
    }),
  });
  register({
    name: "inspect_lesson",
    description:
      "Open a lesson in the teaching room and return its evidence gate result.",
    inputSchema: {
      type: "object",
      properties: { lesson_id: { type: "string" } },
      required: ["lesson_id"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: true, untrustedContentHint: true },
    execute: (input) => {
      const row = state.data?.round?.lessons.find(
        (r) => r.id === input.lesson_id,
      );
      if (!row) throw new Error("Lesson not found in the selected round");
      state.lessonId = row.id;
      navigate("lessons");
      return { id: row.id, concept: row.concept, gate: row.gate };
    },
  });
  window.addEventListener("pagehide", () => lifecycle.abort());
}
