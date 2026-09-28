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
});
