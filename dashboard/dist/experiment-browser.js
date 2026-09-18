// Read-only, cursor-paged teaching history. New selections supersede pending reads.
export function createExperimentBrowser(api, changed = () => {}) {
  const state = { page: null, before: null, strategyVersion: "", selectedId: "", detail: null,
    loading: false, detailLoading: false, error: "" };
  let pageRequest = 0, detailRequest = 0;

  async function select(id) {
    const request = ++detailRequest;
    if (id !== state.selectedId) state.detail = null;
    state.selectedId = id;
    state.detailLoading = true;
    state.error = "";
    changed();
    try {
      const detail = await api(`/experiments/${encodeURIComponent(id)}`);
      if (request !== detailRequest) return;
      state.detail = detail;
      if (!detail) state.error = "No experiment plan was recorded for this iteration.";
    } catch (error) {
      if (request !== detailRequest) return;
      state.detail = null;
      state.error = `Unable to read experiment: ${error.message}`;
    } finally {
      if (request === detailRequest) { state.detailLoading = false; changed(); }
    }
  }

  async function load(before = null, strategyVersion = state.strategyVersion) {
    const request = ++pageRequest;
    const newPage = before !== state.before || strategyVersion !== state.strategyVersion;
    if (newPage) {
      ++detailRequest;
      Object.assign(state, { page: null, detail: null, selectedId: "", detailLoading: false });
    }
    Object.assign(state, { before, strategyVersion, loading: true, error: "" });
    changed();
    const query = new URLSearchParams({ limit: "20" });
    if (before !== null) query.set("before", String(before));
    if (strategyVersion) query.set("strategy_version", strategyVersion);
    try {
      const page = await api(`/experiments?${query}`);
      if (request !== pageRequest) return;
      state.page = page;
      if (!page.items.some(row => row.round_id === state.selectedId)) {
        ++detailRequest;
        state.selectedId = page.items[0]?.round_id || "";
        state.detail = null;
      }
      changed();
      if (state.selectedId) await select(state.selectedId);
    } catch (error) {
      if (request === pageRequest) state.error = `Unable to read experiments: ${error.message}`;
    } finally {
      if (request === pageRequest) { state.loading = false; changed(); }
    }
  }

  return { state, select, load,
    refresh: () => state.loading ? Promise.resolve() : load(state.before),
    filter: version => load(null, version) };
}
