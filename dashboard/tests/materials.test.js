import test from "node:test";
import assert from "node:assert/strict";
import { materialList, learningWork } from "../dist/views.js";
import { phaseFor } from "../dist/telemetry.js";

test("synthetic material is labelled without inventing a student observation", () => {
  const html = materialList({id:"round", material_count:30, material_next_offset:20, materials:[{
    id:"candidate", concept:"<script>topic</script>", student_prompt:"Question", teacher:"<img src=x>",
    use_for_training:true, reason:"Teacher accepted", sources:[], document:{id:""},
    material_origin:{author_id:"API author", model:"model", teacher_edited:true, review_scope:"Selected exact batch"}
  }]});
  assert.match(html, /no student attempt was requested/);
  assert.match(html, /Edited by teacher/);
  assert.match(html, /data-material-next="round"/);
  assert.match(html, /30 candidates/);
  assert.ok(!html.includes("<script>") && !html.includes("<img src=x>"));
  assert.ok(!html.includes("Student attempt</h3>") && !html.includes("Teacher revision</h3>"));
});

test("author accounting separates provider reservations from student work", () => {
  const html = learningWork({id:"r", number:1, measured_work_all_attempts:{tokens:2048,updates:1,elapsed_seconds:1},stage_seconds:{expand:10},finished_stage_seconds:11,teacher_stage_seconds:0,
    material_author_work:{jobs:{complete:2,failed:1},calls:3,reserved_output_tokens_all_attempts:4096,reported_input_tokens:300,reported_output_tokens:200,calls_without_usage:1,limitations:"Unknown usage is not zero cost."},
    material_sources:{targets_by_origin:{"<author>":900,primary_teacher:1148}}
  });
  assert.match(html, /Reported API tokens/);
  assert.match(html, /output tokens reserved/);
  assert.match(html, /Prepared targets by material origin/);
  assert.match(html, /&lt;author&gt;/);
  assert.ok(!html.includes("<author>"));
  assert.equal(phaseFor("running", "expand").moving, true);
  assert.equal(phaseFor("running", "material_select").name, "Practice");
});
