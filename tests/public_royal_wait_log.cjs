const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../frontend/index.html'), 'utf8');
const start = html.indexOf('function localizeGameLogAiDetail(');
const end = html.indexOf('function localizeGameLogAiDecision(', start);
assert.ok(start >= 0 && end > start);
const context = vm.createContext({GAME_LOG_AI_DETAIL_LABELS:{ja:{},en:{},zh:{}}});
vm.runInContext(html.slice(start, end), context);
const detail = 'pass_full_receive_cover_public_route_safe_30_target_50';
for (const language of ['ja','en','zh']) {
  const text = context.localizeGameLogAiDetail(detail, language);
  assert.ok(text.includes('30') && text.includes('50'));
  assert.ok(!text.includes(detail));
}
assert.match(context.localizeGameLogAiDetail(detail,'ja'), /30点の上がりを確保/);
assert.match(context.localizeGameLogAiDetail(detail,'ja'), /50点を狙う/);
console.log('Public royal wait log descriptions passed');
