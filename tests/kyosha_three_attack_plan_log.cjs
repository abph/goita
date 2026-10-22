const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../frontend/index.html'), 'utf8');
const start = html.indexOf('function localizeGameLogAiDetail(');
const end = html.indexOf('function localizeGameLogAiDecision(', start);
const pieces = { '4': '銀', '5': '金' };
const context = vm.createContext({GAME_LOG_AI_DETAIL_LABELS: {ja:{},en:{},zh:{}},
  gameLogCandidatePieceLabel: p => pieces[p] || p});
vm.runInContext(html.slice(start, end), context);
for (const language of ['ja', 'en', 'zh']) {
  const text = context.localizeGameLogAiDetail('pass_kyosha_three_attack_plan_554_penalty_160', language);
  assert.ok(text.includes('金→金→銀') && text.includes('160'));
  assert.ok(!text.includes('pass_kyosha'));
}
assert.match(context.localizeGameLogAiDetail('pass_kyosha_three_attack_plan_554_penalty_160', 'ja'), /立て直し.*パス/);
console.log('Three attack lance plan log descriptions passed');
