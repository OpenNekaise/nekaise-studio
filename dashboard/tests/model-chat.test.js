import test from 'node:test';
import assert from 'node:assert/strict';
import { readChatStream, chatView } from '../dist/model-chat.js';

function response(text) {
  const bytes = new TextEncoder().encode(text);
  return { ok: true, body: new ReadableStream({ start(c) { for (const byte of bytes) c.enqueue(new Uint8Array([byte])); c.close(); } }) };
}
test('CJK and emoji survive arbitrarily split stream chunks', async () => {
  const events=[];
  await readChatStream(response('{"type":"delta","text":"热阻🙂\\n"}\n{"type":"done","text":"热阻🙂"}\n'),e=>events.push(e));
  assert.equal(events[0].text,'热阻🙂\n');
  assert.equal(events[1].type,'done');
});
test('incomplete or failed streams are not successful answers', async () => {
  await assert.rejects(readChatStream(response('{"type":"delta","text":"partial"}\n'),()=>{}),/before the reply finished/);
  await assert.rejects(readChatStream(response('{"type":"error","message":"busy"}\n'),()=>{}),/busy/);
});
test('model text is escaped and version, reset and stop controls remain visible', () => {
  const html=chatView({status:{available:true,model:{id:'abcdef0123',run_name:'Run',round_number:2}},messages:[{role:'assistant',content:'<script>x</script>',model:{id:'abcdef0123',round_number:1}}],busy:true,draft:'</textarea>',progress:'Replying…',error:''});
  assert.ok(!html.includes('<script>'));
  assert.match(html,/&lt;script&gt;/);
  assert.match(html,/Iteration 1/);
  assert.match(html,/Iteration 2/);
  assert.match(html,/Stop reply/);
  assert.match(html,/New chat/);
});

test('named release belongs to each checkpoint and never changes reply text', () => {
  const html = chatView({status:{available:true,model:{id:'new',round_number:4,identity:{name:'Kai',display_name:'Kai 0.1',origin:'Sweden',developer:'Nekaise'}}},messages:[
    {role:'assistant',content:'I am named MiniCPM.',model:{id:'old',round_number:3,identity:{display_name:'Kai 0.0'}}},
    {role:'assistant',content:'legacy reply',model:{id:'legacy',round_number:1}},
  ],busy:false,draft:'',progress:'',error:''});
  assert.match(html, /Kai 0.1/);
  assert.match(html, /Kai from Nekaise/);
  assert.ok(!html.includes('Swedish AI'));
  assert.match(html.replace(/<[^>]*>/g, ''), /Kai 0.0 · Iteration 3/);
  assert.match(html.replace(/<[^>]*>/g, ''), /Model · Iteration 1/);
  assert.match(html, /I am named MiniCPM\./);
});

test('identity metadata is escaped', () => {
  const html = chatView({status:{available:true,model:{identity:{display_name:'<Kai>',origin:'Sweden',developer:'<Nekaise>'}}},messages:[],busy:false,draft:'',progress:'',error:''});
  assert.ok(!html.includes('<Kai>') && !html.includes('<Nekaise>'));
  assert.match(html, /&lt;Kai&gt;/);
  assert.match(html, /&lt;Nekaise&gt;/);
});
