import test from "node:test";
import assert from "node:assert/strict";
import { curriculumCard } from "../dist/curriculum.js";

test("curriculum displays pending or verified exposure, never assumed mastery", () => {
  assert.equal(curriculumCard(null), "");
  const progress = { state: { corpus_cycle: 0, documents_completed: 0, chars_trained: 0, gpc_completed: 0 },
    units: 265, domains: 20, gpc_cycle: 0, next_unit: {title:"<unsafe>"}, policy: {remediation_cap:.2} };
  const pending = curriculumCard(progress);
  assert.match(pending, /No completed training/);
  assert.match(pending, /&lt;unsafe&gt;/);
  progress.last_receipt = { targets: {total:100, by_track:{corpus:45,gpc:40,remediation:15},forward_authors:{luna:10,deepseek:20,kimi:10}} };
  assert.match(curriculumCard(progress), /luna · deepseek · kimi/);
  assert.match(curriculumCard(progress), /15% follow-up/);
  const buffered = curriculumCard(progress, {number:1,status:"training",blocks:[{number:1,status:"complete",prepared_targets:128000,raw_targets:98000},{number:2,status:"prepared",prepared_targets:125000,raw_targets:98000,preparation_only:true}]});
  assert.match(buffered, /1 \/ 2 blocks complete/);
  assert.match(buffered, /review pending/);
  assert.match(buffered, /training parent pending/);
  assert.match(buffered, /only after a verified save/);
});

test("continuous card distinguishes wall time, prepared supply and chat share", () => {
  const progress = {state:{corpus_cycle:0,documents_completed:2,chars_trained:900,gpc_completed:1,web_chars_trained:300},
    units:265,domains:20,gpc_cycle:0,next_unit:{title:"Science"},policy:{remediation_cap:.2}};
  const view=curriculumCard(progress,null,{since:"<date>",training_time_share:.2,durable_targets:1000,
    durable_targets_per_second:10,prepared_windows:2,prepared_targets:2000,durable_scope_targets:{general_chat:100}});
  assert.match(view,/20% of all elapsed time/);
  assert.match(view,/300 verified source characters/);
  assert.match(view,/10%/);
  assert.match(view,/&lt;date&gt;/);
  assert.match(view,/recovery and pauses remain/);
});
