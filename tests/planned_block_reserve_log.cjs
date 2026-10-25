const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../frontend/index.html'), 'utf8');
const labels = html.indexOf('const GAME_LOG_AI_DETAIL_LABELS');
const end = html.indexOf('function localizeGameLogAiDecision(', labels);
const pieceStart = html.indexOf('function gameLogCandidatePieceLabel(');
const pieceEnd = html.indexOf('function formatGameLogCandidateGap(', pieceStart);
assert.ok(labels >= 0 && end > labels && pieceStart > end && pieceEnd > pieceStart);
const context = vm.createContext({});
vm.runInContext(html.slice(labels, end) + html.slice(pieceStart, pieceEnd), context);
const detail = 'plan_block_surplus_3_reserve_1_needed_1_width_4';
for (const language of ['ja', 'en', 'zh']) {
  for (const piece of ['2', '3', '4', '5', '6', '7']) {
    const key = `plan_block_surplus_${piece}_reserve_1_needed_1_width_4`;
    const text = context.localizeGameLogAiDetail(key, language);
    assert.ok(text && text !== key);
    assert.ok(text.includes(context.gameLogCandidatePieceLabel(piece, language)));
    assert.ok(text.includes('1') && text.includes('4'));
  }
}
assert.match(context.localizeGameLogAiDetail(detail, 'ja'), /馬を伏せて馬.*残り1回.*1枚.*4種類.*一巡.*条件付き.*確定上がりではない/);
assert.match(context.localizeGameLogAiDetail(detail, 'en'), /projected.*assuming.*conditional.*not a forced-win proof/);
assert.match(context.localizeGameLogAiDetail(detail, 'zh'), /假设.*有条件.*并非必胜证明/);
console.log('Planned block reserve log descriptions passed');
