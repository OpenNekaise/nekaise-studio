import { escapeHTML as e, number, dateLabel } from "./lib.js?v=28aacea7ef76";
import { scoreTrend, trendLegend } from "./score-trend.js?v=28aacea7ef76";

const validScore = value => Number.isFinite(value) && value >= 0 && value <= 1;
const pct = value => validScore(value) ? `${number(value * 100, 1)}%` : "—";

export function assessmentPoint(round) {
  if (!round?.checkpoint || round.checkpoint === round.model_before) return null;
  const grade = round.stages?.filter(s => s.stage === "grade").sort((a, b) => b.attempt - a.attempt || b.id - a.id)[0];
  const finished = Array.isArray(round.stages) ? grade?.status === "complete" : round.status === "complete" || round.stage === "adapt";
  if (!finished || !validScore(round.score)) return null;
  if (Array.isArray(round.evaluations) && (!round.evaluations.length || round.evaluations.some(q => !validScore(q.grade?.score)))) return null;
  return { round_id: round.id, campaign_id: round.campaign_id, number: round.number, score: round.score,
    at: grade?.finished_at || round.updated_at, created_at: round.created_at,
    reflection_incomplete: round.status === "failed" && round.stage === "adapt", questions: round.evaluations?.length ?? null };
}

// Retain chart projections, not prompts or the full ancestor snapshots.
export function createAssessmentHistory(read) {
  const cache = new Map();
  const project = snapshot => {
    const rounds = new Map(snapshot.rounds.map(r => [r.id, r]));
    if (snapshot.round) rounds.set(snapshot.round.id, snapshot.round);
    return { campaign: snapshot.campaign, rows: [...rounds.values()].map(r => ({ id: r.id, number: r.number, created_at: r.created_at, point: assessmentPoint(r) })) };
  };
  return async function load(snapshot, catalog, active = () => true, progress = () => {}) {
    const available = new Map(catalog.map(c => [c.id, c])), seen = new Set(), lineage = [], errors = [];
    let id = snapshot.campaign.id, boundary = Infinity;
    const publish = loading => {
      const points = lineage.flatMap(run => run.rows.map(r => r.point).filter(p => p && Date.parse(p.at) <= run.boundary))
        .sort((a, b) => Date.parse(a.created_at) - Date.parse(b.created_at) || a.number - b.number);
      const unique = [...new Map(points.map(p => [p.round_id, p])).values()];
      const result = { campaign_id: snapshot.campaign.id, points: unique, loading, partial: errors.length > 0, runs: lineage.length };
      if (active()) progress(result);
      return result;
    };
    while (id && !seen.has(id) && active()) {
      seen.add(id);
      let run;
      try {
        const known = available.get(id), cached = cache.get(id);
        if (id === snapshot.campaign.id) {
          run = project(snapshot);
          if (cached?.campaign.updated_at === run.campaign.updated_at) {
            run.rows = [...new Map([...cached.rows, ...run.rows].map(r => [r.id, r])).values()];
            Object.assign(run, { olderIds: cached.olderIds, olderRows: cached.olderRows });
          }
        } else if (cached && known && cached.campaign.updated_at === known.updated_at) run = cached;
        else run = project(await read(`/campaigns/${encodeURIComponent(id)}`));
        if (!active()) break;
        run.boundary = boundary;
        lineage.push(run);
        cache.set(id, run);
        publish(true);
        const earliest = run.rows.reduce((a, r) => !a || r.number < a.number ? r : a, null);
        if (earliest?.number > 1 || run.olderIds?.some(rid => !run.olderRows?.has(rid))) {
          if (!run.olderIds) {
            const ids = new Set(), knownIds = new Set(run.rows.map(r => r.id));
            let after = 0;
            while (active()) {
              const events = await read(`/events?campaign_id=${encodeURIComponent(id)}&after=${after}&limit=200`);
              for (const event of events) if (event.round_id && !knownIds.has(event.round_id) && Date.parse(event.created_at) < Date.parse(earliest.created_at)) ids.add(event.round_id);
              if (!events.length || events.length < 200 || events.some(event => Date.parse(event.created_at) >= Date.parse(earliest.created_at))) break;
              const next = Math.max(...events.map(event => event.id));
              if (next <= after) throw Error("History cursor did not advance");
              after = next;
            }
            if (!active()) break;
            run.olderIds = [...ids];
          }
          run.olderRows ||= new Map();
          const missing = run.olderIds.filter(rid => !run.olderRows.has(rid));
          for (let start = 0; start < missing.length && active(); start += 4) {
            const batch = missing.slice(start, start + 4);
            const results = await Promise.allSettled(batch.map(rid => read(`/rounds/${encodeURIComponent(rid)}`)));
            for (let i = 0; i < results.length; i++) {
              const result = results[i];
              if (result.status === "fulfilled") {
                const r = result.value;
                if (r.campaign_id !== id) { errors.push(id); continue; }
                run.olderRows.set(r.id, { id: r.id, number: r.number, created_at: r.created_at, point: assessmentPoint(r) });
              } else errors.push(id);
            }
          }
          run.rows = [...new Map([...run.rows, ...run.olderRows.values()].map(r => [r.id, r])).values()];
        }
        boundary = Math.min(boundary, Date.parse(run.campaign.created_at));
        id = run.campaign.parent_campaign_id;
      } catch {
        errors.push(id);
        // The catalog may still give us a parent even when one snapshot is unavailable.
        const campaign = run?.campaign || available.get(id);
        boundary = Math.min(boundary, Date.parse(campaign?.created_at));
        id = campaign?.parent_campaign_id;
      }
    }
    if (id && seen.has(id)) errors.push(id);
    return publish(false);
  };
}

export function assessmentChart(points) {
  if (!points.length) return '<div class="token-chart-empty">A score appears after training and teacher grading finish.</div>';
  const w = 560, h = 200, left = 48, right = 22, top = 16, bottom = 38;
  const x = i => points.length === 1 ? (left + w - right) / 2 : left + i / (points.length - 1) * (w - left - right);
  const y = score => top + (1 - score) * (h - top - bottom);
  const grid = [0, .5, 1].map(score => `<line x1="${left}" x2="${w - right}" y1="${y(score)}" y2="${y(score)}"/><text x="${left - 8}" y="${y(score) + 4}" text-anchor="end">${score * 100}%</text>`).join("");
  const trend = scoreTrend(points.map((p, i) => ({ x: i, y: p.score })), x, y);
  const dots = points.map((p, i) => {
    const label = `${p.reflection_incomplete ? "Reflection incomplete · " : ""}Assessment ${i + 1} · ${pct(p.score)} · Run ${p.campaign_id.replace(/^campaign_/, "")} · Iteration ${p.number}${p.questions !== null ? ` · ${p.questions} questions` : ""} · ${dateLabel(p.at)}`;
    return `<a href="#teacher-assessment" data-score-round="${e(p.round_id)}" data-score-campaign="${e(p.campaign_id)}" aria-label="${e(label)}"><circle class="assessment-hit" cx="${x(i)}" cy="${y(p.score)}" r="9"/><circle class="assessment-point ${i === points.length - 1 ? "latest" : ""}" cx="${x(i)}" cy="${y(p.score)}" r="${i === points.length - 1 ? 5 : 3.5}"/><title>${e(label)}</title></a>`;
  }).join("");
  return `<svg class="${trend ? "has-score-trend" : ""}" viewBox="0 0 ${w} ${h}" role="group" aria-label="Teacher assessment scores and smoothed trend by completed assessment; questions vary each iteration">${grid}${trend}${dots}<text x="${left}" y="${h - 10}">1</text><text x="${w - right}" y="${h - 10}" text-anchor="end">${points.length} completed assessments</text></svg>${trend ? trendLegend : ""}`;
}

export function teacherAssessmentCard(data, currentRound) {
  const points = data?.points || [], latest = points.at(-1);
  const pending = currentRound && latest?.round_id !== currentRound.id;
  const state = !data || data.loading ? "Loading history" : data.partial ? "Partial history" : latest ? "Teacher graded" : "Awaiting assessment";
  return `<section class="panel usage-card teacher-assessment" id="teacher-assessment"><div class="usage-heading"><h2>Teacher assessment</h2><span class="usage-live">${state}</span></div><strong class="usage-total">${pct(latest?.score)}</strong><p class="usage-description">After training · mean question score</p><div class="token-chart">${assessmentChart(points)}</div><div class="usage-footer"><span>${latest ? `Iteration ${number(latest.number)} · Run ${e(latest.campaign_id.replace(/^campaign_/, ""))}${latest.questions !== null ? ` · ${number(latest.questions)} questions` : ""}` : "No completed post-training assessment recorded"}</span><span>${pending && latest ? "Current iteration has no completed post-training score yet. " : ""}Questions vary by iteration; scores are not a fixed benchmark.</span>${data?.partial ? "<span>Some history is unavailable; the next refresh will retry.</span>" : ""}</div><details class="eval-details" data-detail="teacher-assessment-details"><summary>Assessment details</summary><div><p>Mean of the teacher's scores for the student's current answers. Input-checkpoint comparison grades and rounds without a weight update are excluded.</p><p>Assessment feedback informs the teacher's next curriculum. Each point opens its recorded answers and grading.</p>${latest ? `<button class="text-button" data-score-round="${e(latest.round_id)}" data-score-campaign="${e(latest.campaign_id)}">View graded answers ↗</button>` : ""}</div></details></section>`;
}
