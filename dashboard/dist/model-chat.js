import { escapeHTML as e, time } from "./lib.js?v=3ca885116e1d";

export async function readChatStream(response, receive) {
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const detail = Array.isArray(data.detail) ? data.detail.map(x => x.msg).join("; ") : data.detail;
    throw new Error(detail || `Chat request failed (${response.status})`);
  }
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let pending = "", done = false;
  try {
    while (true) {
      const chunk = await reader.read();
      pending += decoder.decode(chunk.value, { stream: !chunk.done });
      const lines = pending.split("\n"); pending = lines.pop();
      for (const line of lines) if (line.trim()) {
        const event = JSON.parse(line);
        if (event.type === "error") throw new Error(event.message);
        if (event.type === "done") done = true;
        receive(event);
      }
      if (chunk.done) break;
    }
    if (pending.trim() || !done) throw new Error("Connection ended before the reply finished. Please retry.");
  } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
}

function modelInfo(status) {
  const model = status?.model;
  return model ? `<span>${e(model.run_name)} · Iteration ${e(model.round_number)}</span><span class="quiet">Completed ${time(model.completed_at)}</span>` : `<span>${e(status?.message || "Checking model availability…")}</span>`;
}

function modelName(model) { return model?.identity?.display_name || "Model"; }
function modelDescription(model) {
  return model?.identity ? `${model.identity.origin === "Sweden" ? "A Swedish AI" : model.identity.name} from ${model.identity.developer}. Talk to the latest completed checkpoint.`
    : "Talk to the latest completed model. Each message uses the newest available checkpoint.";
}

export function chatView(state) {
  return `<section class="model-page" aria-labelledby="model-title">
    <div class="section-heading"><div><div class="eyebrow">Latest student</div><h1 id="model-title">${e(modelName(state.status?.model))}</h1>
    <p class="quiet" id="model-description">${e(modelDescription(state.status?.model))}</p></div>
    <button class="button secondary" id="model-new" ${state.busy ? "disabled" : ""}>New chat</button></div>
    <div class="model-info" id="model-info">${modelInfo(state.status)}</div>
    <p class="model-note quiet">CPU chat keeps GPU capacity available for training. Conversations stay in this page and are not used for training.</p>
    <div class="model-messages" id="model-messages" aria-live="polite" aria-label="Conversation">${state.messages.length ? state.messages.map(m => `<article class="model-message ${m.role}"><div class="model-message-heading">${m.role === "user" ? "You" : e(modelName(m.model))}${m.model ? ` <span class="quiet">· Iteration ${e(m.model.round_number)} · ${e(m.model.id.slice(0, 8))}</span>` : ""}</div><div class="model-message-text">${e(m.content)}</div>${m.partial && m.role === "assistant" ? '<small class="quiet">Incomplete reply · excluded from conversation context</small>' : ""}</article>`).join("") : '<div class="model-empty">Ask a question, explore an idea, or try a follow-up.</div>'}</div>
    <div id="model-progress" class="quiet" role="status">${e(state.progress)}</div>
    <div id="model-error" class="error-banner" role="alert" ${state.error ? "" : "hidden"}>${e(state.error)}</div>
    <form id="model-form" class="model-composer"><label for="model-input">Message</label>
      <textarea id="model-input" rows="3" maxlength="4000" placeholder="Message the model…" ${state.busy ? "disabled" : ""}>${e(state.draft)}</textarea>
      <div class="model-composer-actions"><span class="quiet">Enter to send · Shift + Enter for a new line</span><button type="button" class="button secondary" id="model-stop" ${state.busy ? "" : "hidden"}>Stop reply</button><button type="submit" class="button primary" ${state.busy || !state.status?.available ? "disabled" : ""}>Send</button></div>
    </form></section>`;
}

export function createModelChat(api, fetcher = fetch) {
  const state = { status: null, messages: [], draft: "", busy: false, progress: "", error: "" };
  let container = null, controller = null;
  function draw() {
    if (!container) return;
    const input = container.querySelector("#model-input"), focused = input === document.activeElement;
    const selection = focused ? [input.selectionStart, input.selectionEnd] : null;
    const log = container.querySelector("#model-messages");
    const nearBottom = !log || log.scrollHeight - log.scrollTop - log.clientHeight < 80;
    const scroll = log?.scrollTop || 0;
    container.innerHTML = chatView(state);
    const nextLog = container.querySelector("#model-messages");
    nextLog.scrollTop = nearBottom ? nextLog.scrollHeight : scroll;
    if (focused) { const next = container.querySelector("#model-input"); next.focus(); next.setSelectionRange(...selection); }
  }
  async function refresh() {
    try { state.status = await api("/model"); }
    catch { state.status = { available: false, message: "Unable to check model availability. Reconnecting…" }; }
    // Preserve the composer DOM and streamed answer while the dashboard polls.
    if (container) {
      container.querySelector("#model-title").textContent = modelName(state.status?.model);
      container.querySelector("#model-description").textContent = modelDescription(state.status?.model);
      container.querySelector("#model-info").innerHTML = modelInfo(state.status);
      container.querySelector("[type=submit]").disabled = state.busy || !state.status?.available;
    }
  }
  async function submit(event) {
    event.preventDefault();
    if (state.busy || !state.draft.trim() || !state.status?.available) return;
    const text = state.draft.trim();
    const history = state.messages.filter(m => !m.partial).map(({ role, content }) => ({ role, content }));
    if (history.length >= 30) { state.error = "Start a new chat to continue; this conversation has reached its limit."; draw(); return; }
    state.busy = true; state.error = ""; state.progress = "Loading the latest model…"; state.draft = "";
    const user = { role: "user", content: text }, answer = { role: "assistant", content: "" };
    state.messages.push(user, answer); controller = new AbortController(); draw();
    try {
      const response = await fetcher("/api/model/chat", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: [...history, { role: "user", content: text }] }), signal: controller.signal });
      await readChatStream(response, event => {
        if (event.type === "model") answer.model = event.model;
        if (event.type === "ready") state.progress = "Replying…";
        if (event.type === "delta") answer.content += event.text;
        if (event.type === "done") { answer.content = event.text; state.progress = `${event.tokens} tokens · ${event.seconds.toFixed(1)}s${event.stop_reason === "length" ? " · Reply length limit reached" : ""}`; }
        draw();
      });
    } catch (error) {
      user.partial = true; answer.partial = true;
      state.error = error.name === "AbortError" ? "Reply stopped. You can send the message again." : error.message;
      state.draft = text; state.progress = "";
    } finally {
      state.busy = false; controller = null; draw();
      container?.querySelector("#model-input")?.focus();
    }
  }
  function onInput(event) { if (event.target.id === "model-input") state.draft = event.target.value; }
  function onKey(event) {
    if (event.target.id === "model-input" && event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); submit(event); }
  }
  function onClick(event) {
    if (event.target.closest("#model-stop")) controller?.abort();
    if (event.target.closest("#model-new") && !state.busy) { state.messages = []; state.error = ""; state.progress = ""; state.draft = ""; draw(); container.querySelector("#model-input").focus(); }
  }
  return { state, refresh,
    mount(root) {
      if (container === root) return;
      container = root;
      root.addEventListener("submit", submit); root.addEventListener("input", onInput);
      root.addEventListener("keydown", onKey); root.addEventListener("click", onClick); draw();
    },
    unmount() {
      if (!container) return;
      container.removeEventListener("submit", submit); container.removeEventListener("input", onInput);
      container.removeEventListener("keydown", onKey); container.removeEventListener("click", onClick);
      container = null; controller?.abort();
    },
  };
}
