const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../frontend/index.html'), 'utf8');
const start = html.indexOf('const GAME_LOG_AI_DETAIL_LABELS');
const end = html.indexOf('function localizeGameLogAiDecision(', start);
const pieceStart = html.indexOf('function gameLogCandidatePieceLabel(');
const pieceEnd = html.indexOf('function formatGameLogCandidateGap(', pieceStart);
assert.ok(start >= 0 && end > start);
const context = vm.createContext({});
vm.runInContext(html.slice(start, end) + html.slice(pieceStart, pieceEnd), context);
for (const lang of ['ja', 'en', 'zh']) {
  const detail = 'safe_30_target_50_chance_95_risk_1_public_response_4';
  const text = context.localizeGameLogAiDetail(detail, lang);
  assert.notEqual(text, detail);
  for (const number of ['30', '50', '95%', '1%']) assert.ok(text.includes(number));
  assert.match(context.localizeGameLogAiDetail('safe_30_target_50_chance_80_risk_5', lang), /80%/);
}
assert.match(context.localizeGameLogAiDetail('safe_30_target_50_chance_95_risk_1_public_response_4', 'ja'), /銀/);
assert.match(context.localizeGameLogAiDetail('safe_30_target_50_chance_95_risk_1_public_response_4', 'ja'), /確定上がりではない/);
console.log('Public response upside log descriptions passed');
