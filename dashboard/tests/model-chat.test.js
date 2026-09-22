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
