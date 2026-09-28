import { escapeHTML as e, number, percent } from "./lib.js?v=9d7841b9939d";

export function curriculumCard(progress, cycle = null) {
  if (!progress) return "";
  const state = progress.state, receipt = progress.last_receipt;
  const tracks = receipt?.targets?.by_track || {};
  const total = receipt?.targets?.total || 0;
  const authors = Object.keys(receipt?.targets?.forward_authors || {});
  const inventory = state.inventory;
  return `<section class="panel detail-panel"><div class="detail-body"><h2>Curriculum progression</h2>
    <p class="quiet">Verified training exposure · learning scores do not hold either loop</p>
    ${cycle ? `<p>Teaching cycle ${number(cycle.number)} · ${number(cycle.blocks.filter(b => b.status === "complete").length)} / ${number(cycle.blocks.length)} blocks complete · ${cycle.status === "complete" ? "review recorded" : "review pending"}</p>
    <dl class="training-details">${cycle.blocks.map(b => `<div><dt>Block ${number(b.number)} · ${e(b.status)}</dt><dd>${b.prepared_targets == null ? "Preparing targets" : `${number(b.prepared_targets)} prepared targets · ${number(b.raw_targets)} raw corpus`}${b.preparation_only ? " · training parent pending" : ""}</dd></div>`).join("")}</dl><p class="quiet">Preparation can overlap training. Prepared tokens count as coverage only after a verified save; online assessment follows the final block.</p>` : ""}
    <dl class="training-details"><div><dt>Built environment</dt><dd>Pass ${number(state.corpus_cycle + 1)} · ${number(state.documents_completed)} document completions · ${number(state.chars_trained)} source characters trained</dd></div>
    <div><dt>Current inventory</dt><dd>${inventory ? `${number(inventory.documents)} eligible documents · ${number(inventory.excluded_documents)} outside the published training view` : "Inventory opens on the next training round"}</dd></div>
    <div><dt>General curriculum</dt><dd>${number(state.gpc_completed)} unit exposures · ${number(progress.units)} units across ${number(progress.domains)} domains · pass ${number(progress.gpc_cycle + 1)}</dd></div>
    <div><dt>Next unit</dt><dd>${e(progress.next_unit.title)}</dd></div>
    <div><dt>Adaptive follow-up</dt><dd>Teacher chooses the mix · maximum ${percent(progress.policy.remediation_cap)}</dd></div></dl>
    ${receipt ? `<p>Last completed recipe: ${percent(tracks.corpus / total)} corpus · ${percent(tracks.gpc / total)} general · ${percent((tracks.remediation || 0) / total)} follow-up.</p><p class="quiet">Forward Authors: ${authors.map(e).join(" · ")}. Coverage starts with this policy; earlier sampled passages are not counted retroactively.</p>` : '<p class="quiet">No completed training under this progression policy yet.</p>'}
    </div></section>`;
}
