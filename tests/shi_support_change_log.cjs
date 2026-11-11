const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../frontend/index.html'), 'utf8');
const start = html.indexOf('const GAME_LOG_AI_DETAIL_LABELS');
const end = html.indexOf('function localizeGameLogAiDecision(', start);
const context = vm.createContext({gameLogCandidatePieceLabel: piece => ({'2':'香','6':'角'})[piece]});
vm.runInContext(html.slice(start, end), context);
for (const language of ['ja','en','zh']) {
  for (const piece of ['2','6']) {
    for (const ownShi of [0,1]) {
      const detail = `receive_limited_shi_support_${ownShi}_followup_${piece}`;
      assert.notEqual(context.localizeGameLogAiDetail(detail, language), detail);
    }
    const detail = `attack_limited_shi_support_${piece}`;
    assert.notEqual(context.localizeGameLogAiDetail(detail, language), detail);
  }
}
assert.match(context.localizeGameLogAiDetail('receive_limited_shi_support_1_followup_2','ja'), /しは1枚.*香.*協力が難しい/);
assert.match(context.localizeGameLogAiDetail('attack_limited_shi_support_6','ja'), /角.*0枚という断定ではない/);
console.log('Limited shi support receive/followup explanations passed');
