const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../frontend/index.html'), 'utf8');
const labels = html.indexOf('const GAME_LOG_AI_DETAIL_LABELS');
const start = html.indexOf('function localizeGameLogAiDetail(');
const end = html.indexOf('function localizeGameLogAiDecision(', start);
assert.ok(labels >= 0 && start > labels && end > start);
const context = vm.createContext({});
vm.runInContext(html.slice(labels, end), context);
for (const language of ['ja', 'en', 'zh']) {
  for (const phase of ['receive', 'attack']) {
    const detail = `${phase}_first_three_shi_approval`;
    const text = context.localizeGameLogAiDetail(detail, language);
    assert.ok(text && text !== detail);
  }
}
assert.match(context.localizeGameLogAiDetail('receive_first_three_shi_approval', 'ja'), /1枚残し.*初回/);
assert.match(context.localizeGameLogAiDetail('attack_first_three_shi_approval', 'ja'), /初回.*しを返す/);
console.log('First three-shi approval log descriptions passed');
