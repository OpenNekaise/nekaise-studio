import test from "node:test";
import assert from "node:assert/strict";
import { escapeHTML, diffWords, lossChart, safeURL } from "../dist/lib.js";
import { recoveryNotice, overview, teachingPlan, lessonsView } from "../dist/views.js";

test("untrusted student and teacher text cannot become HTML", () => {
  const result = diffWords(
    "<script>alert(1)</script> watts",
    "<img src=x onerror=alert(1)> kelvin per watt",
  );
  assert.ok(!result.before.includes("<script>"));
  assert.ok(!result.after.includes("<img"));
  assert.ok(result.after.includes("<mark>"));
  assert.equal(safeURL("javascript:alert(1)"), "#");
});
test("diff preserves unchanged passages", () => {
  assert.deepEqual(diffWords("The heat flows.", "The heat flows."), {
    before: "The heat flows.",
    after: "The heat flows.",
  });
});
test("chart uses measured steps and handles empty and single-step runs", () => {
  assert.match(lossChart([]), /when training starts/);
  const chart = lossChart([{ step: 1, loss: 2.7 }]);
  assert.ok(!chart.includes("NaN"));
  assert.match(chart, /Step 1: loss 2.7000/);
  assert.match(lossChart([{ step: 1, loss: NaN }]), /when training starts/);
});

test("waiting is normal, escaped and shows the next retry", () => {
  const view = recoveryNotice({campaign:{status:"waiting"},recovery:{error:"<script>quota</script>",retry_at:"2026-09-15T12:00:00Z"}});
  assert.match(view, /Next automatic attempt/);
  assert.ok(!view.includes("<script>"));
  assert.equal(recoveryNotice({campaign:{status:"complete"}}), "");
});

test("token ledger separates prepared and measured training tokens", () => {
  const view = overview({campaign:{status:"waiting",config:{rounds:-1,lessons_per_round:2,learning_rate:.00001}},rounds:[],round:{number:1,status:"waiting",lessons:[],evaluations:[],metrics:[],token_ledger:{total_tokens:80,streams:{teacher:{prepared_tokens:60},corpus:{prepared_tokens:20},replay:{prepared_tokens:0}},missing_streams:["replay"]}},stages:[],events:[],teacher_usage:{calls:0}});
  assert.match(view, /75%/);
  assert.match(view, /25%/);
  assert.match(view, /TRAINED/);
  assert.ok(!view.includes("NaN"));
  assert.ok(!view.includes("-1 rounds"));
});

test("teacher curriculum and durable notes are shown without interpreting HTML", () => {
  const plan = {lessons:[{}],readings:[],replay:[],train_epochs:2,notes:"<script>plan</script>",evaluation_instructions:"Choose a diagnostic"};
  const html=teachingPlan({round:{curriculum:plan,teaching_strategy:{student_notes:"<img src=x>",next_round_instructions:"Review older lessons",action:"continue",reason:"Teacher decision"}}});
  assert.match(html,/1 tasks/);
  assert.match(html,/Review older lessons/);
  assert.ok(!html.includes("<script>"));
  assert.ok(!html.includes("<img"));
});

test("teacher-authored lessons expose their exact training text without a source", () => {
  const row={id:"l1",kind:"sft",concept:"Custom exercise",prompt:"Task",student_prompt:"Exact task context",student:"Attempt",teacher:"Correction",training_text:"Teacher-chosen <sequence>",errors:[],evidence:[],sources:[],document:{id:"",text:"",selection_reason:"Teacher choice",replay:false},gate:{passed:true,mode:"trusted_teacher",reason:"Teach"}};
  const html=lessonsView({round:{number:1,lessons:[row]},rounds:[]},{diff:false});
  assert.match(html,/no corpus source declared/);
  assert.match(html,/EXACT TRAINING TEXT/);
  assert.match(html,/Teacher-chosen &lt;sequence&gt;/);
});
