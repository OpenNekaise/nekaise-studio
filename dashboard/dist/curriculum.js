import { escapeHTML as e, number, percent } from "./lib.js?v=30d66fdc235c";

export function curriculumCard(progress, cycle = null, throughput = null) {
  if (!progress) return "";
  const state = progress.state, receipt = progress.last_receipt;
  const tracks = receipt?.targets?.by_track || {};
  const total = receipt?.targets?.total || 0;
  const authors = Object.keys(receipt?.targets?.forward_authors || {});
  const inventory = state.inventory;
  return `<section class="panel detail-panel"><div class="detail-body"><h2>Curriculum progression</h2>
    <p class="quiet">Verified training exposure · learning scores do not hold either loop</p>
    ${cycle ? `<p>Teaching cycle ${number(cycle.number)} · ${number(cycle.blocks.filter(b => b.status === "complete").length)} / ${number(cycle.blocks.length)} blocks complete · ${cycle.status === "complete" ? "review recorded" : "review pending"}</p>
    <dl class="training-details">${cycle.blocks.map(b => `<div><dt>Block ${number(b.number)} · ${e(b.status)}</dt><dd>${b.prepared_targets == null ? "Preparing targets" : `${number(b.prepared_targets)} prepared targets · ${number(b.raw_targets)} raw corpus · ${number(b.web_targets || 0)} GPC web prose`}${b.preparation_only ? " · training parent pending" : ""}</dd></div>`).join("")}</dl><p class="quiet">Preparation can overlap training. Prepared tokens count as coverage only after a verified save; online assessment follows the final block.</p>` : ""}
    <dl class="training-details"><div><dt>Built environment</dt><dd>Pass ${number(state.corpus_cycle + 1)} · ${number(state.documents_completed)} document completions · ${number(state.chars_trained)} source characters trained</dd></div>
    <div><dt>Current inventory</dt><dd>${inventory ? `${number(inventory.documents)} eligible documents · ${number(inventory.excluded_documents)} outside the published training view` : "Inventory opens on the next training round"}</dd></div>
    <div><dt>General curriculum</dt><dd>${number(state.gpc_completed)} unit exposures · ${number(progress.units)} units across ${number(progress.domains)} domains · pass ${number(progress.gpc_cycle + 1)}</dd></div>
    <div><dt>GPC webpage coverage</dt><dd>${number(state.web_chars_trained || 0)} verified source characters trained</dd></div>
    <div><dt>Next unit</dt><dd>${e(progress.next_unit.title)}</dd></div>
    <div><dt>Adaptive follow-up</dt><dd>Teacher chooses the mix · maximum ${percent(progress.policy.remediation_cap)}</dd></div></dl>
    ${receipt ? `<p>Last completed recipe: ${percent(tracks.corpus / total)} corpus · ${percent(tracks.gpc / total)} general · ${percent((tracks.remediation || 0) / total)} follow-up.</p><p class="quiet">Forward Authors: ${authors.map(e).join(" · ")}. Coverage starts with this policy; earlier sampled passages are not counted retroactively.</p>` : '<p class="quiet">No completed training under this progression policy yet.</p>'}
    ${throughput ? `<h3>Continuous training</h3><dl class="training-details">
    <div><dt>Measured training time</dt><dd>${percent(throughput.training_time_share)} of all elapsed time</dd></div>
    <div><dt>Saved target throughput</dt><dd>${number(throughput.durable_targets_per_second, 1)} tokens/s across elapsed time</dd></div>
    <div><dt>Prepared supply</dt><dd>${number(throughput.prepared_windows)} windows · ${number(throughput.prepared_targets)} targets</dd></div>
    <div><dt>Saved general chat share</dt><dd>${throughput.durable_targets ? percent(throughput.durable_scope_targets.general_chat / throughput.durable_targets) : "Awaiting verified training"}</dd></div>
    </dl><p class="quiet">Measured since ${e(throughput.since)}. Startup, data preparation, saves, reviews, recovery and pauses remain in elapsed time. GPU memory allocation is not utilization; more training time is not evidence of better answers.</p>` : ""}
    </div></section>`;
}
