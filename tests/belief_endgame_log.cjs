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
  const detail = 'belief_endgame_worlds_48_win_56_danger_29';
  const text = context.localizeGameLogAiDetail(detail, language);
  assert.notEqual(text, detail);
  assert.match(text, /48/);
  assert.match(text, /56%/);
  assert.match(text, /29%/);
}
assert.match(context.localizeGameLogAiDetail('belief_endgame_worlds_48_win_56_danger_29', 'ja'), /確定上がりではない/);
console.log('Belief endgame estimate log descriptions passed');
