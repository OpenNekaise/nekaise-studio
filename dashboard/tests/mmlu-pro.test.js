import test from 'node:test';
import assert from 'node:assert/strict';
import { mmluCard } from '../dist/mmlu-pro.js';
import { SFT_BASELINE_ID } from '../dist/gpqa.js';
const row = overrides => ({status:'complete',model_id:'a'.repeat(64),model_label:'Kai 0.1',
  round_number:12, protocol_id:'b'.repeat(64), dataset_sha256:'c'.repeat(64), max_input_tokens:8192,
  protocol:'mmlu-pro-choice-1',n:12032,completed:12032,correct:1203,score:.1,ci95:[.08,.12],chance_score:.112,
  started_at:'2026-09-29T10:00:00Z',updated_at:'2026-09-29T12:00:00Z',subjects:[{category:'biology',n:717,correct:70,score:70/717}],...overrides});
const card = runs => mmluCard({status:'ok',runs});
test('empty and pending runs have no invented score',()=>{
  assert.match(mmluCard(null),/MMLU-Pro/);
  assert.match(mmluCard(null),/usage-total">—/);
  const html=card([row({status:'running',score:null,completed:70,subjects:null})]);
  assert.match(html,/70 \/ 12,032/);assert.match(html,/usage-total">—/);
  assert.match(html,/after all 12,032 answers are verified/);
});
test('real zero is shown, subject/model text escaped, protocol explained',()=>{
  const html=card([row({score:0,correct:0,model_label:'<script>BAD</script>'})]);
  assert.match(html,/usage-total">0\.0%/);assert.doesNotMatch(html,/<script>/);
  assert.match(html,/biology/);assert.match(html,/different from the published five-shot/);
  assert.match(html,/Random choice · 11.2%/);assert.match(html,/weekly/);
});
test('baseline line only uses matching full evaluation',()=>{
  const baseline=row({model_id:SFT_BASELINE_ID,score:.12,started_at:'2026-09-28T10:00:00Z'});
  let html=card([row({}),baseline]);assert.match(html,/<line class="gpqa-baseline-line"/);
  assert.match(html,/-2\.0 percentage points/);
  html=card([row({protocol_id:'d'.repeat(64)}),baseline]);
  assert.doesNotMatch(html,/<line class="gpqa-baseline-line"/);assert.match(html,/different protocol/);
});
test('failed/latest evaluation keeps old completed score and warns stale progress',()=>{
  let html=card([row({status:'failed',score:null,started_at:'2026-09-30T10:00:00Z'}),row({})]);
  assert.match(html,/Latest attempt failed/);assert.match(html,/Last completed result/);
  html=mmluCard({status:'ok',stale:true,runs:[row({status:'running',score:null})]});
  assert.match(html,/No recent progress/);
});
