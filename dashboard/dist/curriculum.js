import { escapeHTML as e, number, percent } from "./lib.js?v=8966e20f4ec9";

export function curriculumCard(progress) {
  if (!progress) return "";
  const state = progress.state, receipt = progress.last_receipt;
  const tracks = receipt?.targets?.by_track || {};
  const total = receipt?.targets?.total || 0;
  const authors = Object.keys(receipt?.targets?.forward_authors || {});
  const inventory = state.inventory;
  return `<section class="panel detail-panel"><div class="detail-body"><h2>Curriculum progression</h2>
    <p class="quiet">Verified training exposure · learning scores do not hold either loop</p>
    <dl class="training-details"><div><dt>Built environment</dt><dd>Pass ${number(state.corpus_cycle + 1)} · ${number(state.documents_completed)} document completions · ${number(state.chars_trained)} source characters trained</dd></div>
    <div><dt>Current inventory</dt><dd>${inventory ? `${number(inventory.documents)} eligible documents · ${number(inventory.excluded_documents)} outside the published training view` : "Inventory opens on the next training round"}</dd></div>
    <div><dt>General curriculum</dt><dd>${number(state.gpc_completed)} unit exposures · ${number(progress.units)} units across ${number(progress.domains)} domains · pass ${number(progress.gpc_cycle + 1)}</dd></div>
    <div><dt>Next unit</dt><dd>${e(progress.next_unit.title)}</dd></div>
    <div><dt>Adaptive follow-up</dt><dd>Teacher chooses the mix · maximum ${percent(progress.policy.remediation_cap)}</dd></div></dl>
    ${receipt ? `<p>Last completed recipe: ${percent(tracks.corpus / total)} corpus · ${percent(tracks.gpc / total)} general · ${percent((tracks.remediation || 0) / total)} follow-up.</p><p class="quiet">Forward Authors: ${authors.map(e).join(" · ")}. Coverage starts with this policy; earlier sampled passages are not counted retroactively.</p>` : '<p class="quiet">No completed training under this progression policy yet.</p>'}
    </div></section>`;
}
