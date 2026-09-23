import { METRICS } from "./benchmark.js?v=96804ad8b77d";

// A browsing session pins an immutable aggregate snapshot while live polling continues.
export function createBenchmarkBrowser(read, changed = () => {}) {
  let serial = 0, latest = null;
  const empty = () => ({ open: false, metric: "score", snapshot: null, history_id: null,
    page: null, data: null, loading: false, error: null, selected_model_id: null, newerAvailable: false });
  let state = empty();
  const notify = () => changed(state);
  function reset() { serial += 1; state = empty(); latest = null; }
  function observe(projection) {
    if (typeof projection?.campaign_id !== "string") return;
    if (latest?.campaign_id !== projection?.campaign_id) reset();
    latest = projection;
    state.newerAvailable = !!(state.open && projection?.history && state.history_id !== projection.history.id);
  }
  async function load(page) {
    const snapshot = state.snapshot, history = snapshot?.history;
    if (!state.open || !history || !Number.isInteger(page) || page < 0 || page >= history.pages) return;
    const ticket = ++serial;
    Object.assign(state, { page, loading: true, error: null, data: null, selected_model_id: null });
    notify();
    try {
      const data = await read(`/campaigns/${encodeURIComponent(snapshot.campaign_id)}/benchmark/history?history=${encodeURIComponent(history.id)}&page=${page}`);
      if (ticket !== serial || !state.open) return;
      if (data?.schema_version !== 2 || data.campaign_id !== snapshot.campaign_id ||
          data.history_id !== history.id || data.cohort_id !== history.cohort_id ||
          data.page !== page || data.pages !== history.pages || data.count !== history.count ||
          data.page_size !== 100 || !Array.isArray(data.points) ||
          data.points.length !== Math.min(100, history.count - page * 100)) {
        throw new Error("Observation page is unavailable or does not match this history.");
      }
      state.data = data;
      if (!data.points.some(p => p.model_id === state.selected_model_id)) state.selected_model_id = null;
    } catch (error) {
      if (ticket !== serial || !state.open) return;
      state.error = `${error.message} Select the page again to retry.`;
    } finally {
      if (ticket === serial && state.open) { state.loading = false; notify(); }
    }
  }
  async function open(projection = latest) {
    observe(projection);
    if (!projection?.history) return;
    if (state.open) { notify(); return; }
    state = { ...empty(), open: true, snapshot: projection, history_id: projection.history.id };
    await load(projection.history.pages - 1);
  }
  return {
    get state() { return state; }, observe, open, load, reset,
    close() { serial += 1; state.open = false; state.loading = false; notify(); },
    async refresh() { const projection = latest; reset(); await open(projection); },
    overview() { serial += 1; Object.assign(state, { page: null, data: null, loading: false, error: null, selected_model_id: null }); notify(); },
    metric(name) { if (Object.hasOwn(METRICS, name)) { state.metric = name; notify(); } },
    select(id) {
      // The view shows the pinned overview until a page arrives, including failed reads.
      const points = state.page == null || !state.data ? state.snapshot?.points : state.data.points;
      if (points?.some(p => p.model_id === id)) { state.selected_model_id = id; notify(); }
    },
  };
}
