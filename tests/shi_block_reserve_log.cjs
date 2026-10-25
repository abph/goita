const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../frontend/index.html'), 'utf8');
const start = html.indexOf('const GAME_LOG_AI_DETAIL_LABELS');
const end = html.indexOf('function localizeGameLogAiDecision(', start);
assert.ok(start >= 0 && end > start);
const context = vm.createContext({});
vm.runInContext(html.slice(start, end), context);
for (const language of ['ja', 'en', 'zh']) {
  const detail = 'block_shi_keep_receivers_reserve_3_needed_2';
  const text = context.localizeGameLogAiDetail(detail, language);
  assert.notEqual(text, detail);
  assert.match(text, /3/);
  assert.match(text, /2/);
}
const japanese = context.localizeGameLogAiDetail('block_shi_keep_receivers_reserve_2_needed_2', 'ja');
assert.match(japanese, /他の受け駒/);
assert.match(japanese, /しで受ける分は別に必要/);
assert.match(japanese, /確定上がりではない/);
console.log('Shi attack reserve log descriptions passed');
