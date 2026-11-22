const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync('frontend/index.html', 'utf8');
function extract(name) {
  const start = html.indexOf('function ' + name + '(');
  assert.ok(start >= 0);
  const end = html.indexOf('\nfunction ', start + 1);
  return html.slice(start, end < 0 ? undefined : end);
}
const context = vm.createContext({});
vm.runInContext(['localizeGameLogAttackCandidates', 'gameLogCandidatePieceLabel'].map(extract).join('\n'), context);
for (const language of ['ja', 'en', 'zh']) {
  const message = context.localizeGameLogAttackCandidates(encodeURIComponent(JSON.stringify({
    public_receive_proof: {source: 'public_information', proven: true, receive: '3',
      hand_size_before: 8, minimum_score: 40, followups: [{action: ['attack', null, '9'], minimum_score: 40}]}
  })), language, 'receive');
  assert.match(message, /40/);
  assert.match(message, language === 'ja' ? /確定上がり/ : language === 'en' ? /proves a finish/ : /必胜路线/);
  assert.match(message, language === 'ja' ? /非公開手駒や標本の勝率は使っていません/ : language === 'en' ? /No hidden opponent pieces or sampled win rates/ : /不使用对手隐藏牌或抽样胜率/);
  assert.ok(!message.includes('undefined') && !message.includes('NaN'));
}
console.log('Public eight-card receive proof log: Japanese, English and Chinese passed');
