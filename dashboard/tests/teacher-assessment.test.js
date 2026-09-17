import test from "node:test";
import assert from "node:assert/strict";
import { assessmentPoint, assessmentChart, teacherAssessmentCard, createAssessmentHistory } from "../dist/teacher-assessment.js";

const at = day => `2026-09-${String(day).padStart(2, "0")}T12:00:00Z`;
const run = (id, day, parent = null) => ({ id, created_at: at(day), updated_at: at(day), parent_campaign_id: parent });
const round = (id, campaign, number, day, score = .5) => ({ id, campaign_id: campaign, number, created_at: at(day), updated_at: at(day), status: "complete", stage: "adapt", model_before: "before", checkpoint: `after-${id}`, score });
const snapshot = (campaign, rounds) => ({ campaign, rounds, round: rounds[0] });

test("teacher chart requires completed grading after a weight update, including a failed reflection", () => {
  const r = { ...round("r", "a", 1, 2, 0), stages: [{ id: 1, stage: "grade", attempt: 1, status: "complete", finished_at: at(2) }], evaluations: [{ student: "", grade: { score: 0 }, comparison: { grade: { score: 1 } } }] };
  assert.equal(assessmentPoint(r).score, 0);
  assert.equal(assessmentPoint(r).questions, 1);
  assert.equal(assessmentPoint({ ...r, checkpoint: "before" }), null);
  assert.equal(assessmentPoint({ ...r, stages: [{ stage: "grade", attempt: 1, status: "running" }] }), null);
  assert.equal(assessmentPoint({ ...r, evaluations: [...r.evaluations, { grade: null }] }), null);
  assert.equal(assessmentPoint({ ...r, stages: [...r.stages, { id: 2, stage: "grade", attempt: 2, status: "failed" }] }), null);
  assert.equal(assessmentPoint({ ...r, score: 1.1 }), null);
  assert.equal(assessmentPoint({ ...r, status: "failed" }).reflection_incomplete, true);
  assert.equal(assessmentPoint({ ...round("pending", "a", 2, 3), status: "running", stage: "grade" }), null);
});

test("teacher score dots preserve zero and links, escape labels and identify the fitted trend", () => {
  const point = assessmentPoint(round('<img>', 'campaign_<script>', 1, 2, 0));
  const svg = assessmentChart([point]);
  assert.match(svg, /0\.0%/);assert.ok(!svg.includes('<path'));assert.ok(!svg.includes('<script>'));assert.ok(!svg.includes('<img>'));assert.ok(!/NaN|Infinity/.test(svg));
  assert.match(teacherAssessmentCard(null), /usage-total">—/);
  assert.match(teacherAssessmentCard({ points: [point], partial: true }), /Partial history/);
  assert.match(teacherAssessmentCard({ points: [point] }, { id: 'later' }), /Current iteration has no completed/);
  const fitted = assessmentChart([point, { ...point, score: .9 }, { ...point, score: .2 }]);
  assert.match(fitted, /class="score-trend"/);
  assert.match(fitted, /Smoothed trend/);
  assert.match(fitted, /questions vary each iteration/);
  assert.equal((fitted.match(/data-score-round=/g) || []).length, 3);
  assert.equal((fitted.match(/class="assessment-point /g) || []).length, 3);
});

test("assessment history follows ancestors, excludes siblings and observations after a branch", async () => {
  const parent = run('parent', 1), child = run('child', 4, 'parent'), sibling = run('sibling', 3, 'parent');
  const reads=[];
  const load=createAssessmentHistory(async path=>{reads.push(path);assert.equal(path,'/campaigns/parent');return snapshot(parent,[round('late','parent',3,5),round('p','parent',1,2)]);});
  const result=await load(snapshot(child,[round('c','child',1,6)]),[child,parent,sibling]);
  assert.deepEqual(result.points.map(p=>p.round_id),['p','c']);assert.equal(result.partial,false);
  await load(snapshot(child,[round('c','child',1,6)]),[child,parent,sibling]);assert.equal(reads.length,1);
});

test("history pages recover rounds omitted by the snapshot and retry a missing old round", async () => {
  const campaign=run('a',1), selected=snapshot(campaign,[round('r3','a',3,5)]), requests=[];
  let fail=true;
  const load=createAssessmentHistory(async path=>{
    requests.push(path);
    if(path==='/events?campaign_id=a&after=0&limit=200')return Array.from({length:200},(_,i)=>({id:i+1,round_id:'r1',created_at:at(2)}));
    if(path==='/events?campaign_id=a&after=200&limit=200')return [{id:201,round_id:'r2',created_at:at(3)},{id:202,round_id:'r3',created_at:at(5)}];
    if(path==='/rounds/r1')return round('r1','a',1,2);
    if(path==='/rounds/r2'){if(fail)throw Error('temporary');return round('r2','a',2,3);}
    throw Error(path);
  });
  const first=await load(selected,[campaign]);assert.equal(first.partial,true);assert.deepEqual(first.points.map(p=>p.round_id),['r1','r3']);
  fail=false;const before=requests.length;const second=await load(selected,[campaign]);
  assert.equal(second.partial,false);assert.deepEqual(second.points.map(p=>p.round_id),['r1','r2','r3']);assert.deepEqual(requests.slice(before),['/rounds/r2']);
  const again=requests.length;await load(selected,[campaign]);assert.equal(requests.length,again);
  const newer=snapshot(campaign,[round('r4','a',4,6),round('r3','a',3,5)]);
  assert.deepEqual((await load(newer,[campaign])).points.map(p=>p.round_id),['r1','r2','r3','r4']);assert.equal(requests.length,again);
});

test("unavailable or cancelled history cannot become a zero or replace another selection", async () => {
  const child=run('c',3,'p'),parent=run('p',1);
  const load=createAssessmentHistory(async()=>{throw Error('unavailable');});
  const result=await load(snapshot(child,[]),[child,parent]);assert.equal(result.partial,true);assert.equal(result.points.length,0);
  let active=true,finish;const updates=[];
  const delayed=createAssessmentHistory(()=>new Promise(resolve=>{finish=resolve;}));
  const work=delayed(snapshot(child,[]),[child,parent],()=>active,data=>updates.push(data));
  active=false;const count=updates.length;finish(snapshot(parent,[round('r','p',1,2)]));await work;assert.equal(updates.length,count);
});
