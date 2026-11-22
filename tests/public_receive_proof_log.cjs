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
  const comparison = context.localizeGameLogAttackCandidates(encodeURIComponent(JSON.stringify({
    guaranteed_receive_comparison: {
      basis: 'minimum_score',
      selected: [['pass', null, null]],
      candidates: [
        {actions: [['pass', null, null]], minimum_score: 30, maximum_score: 50, expected_score: 47.2},
        {actions: [['receive', '2', null], ['attack', null, '4']], minimum_score: 10, maximum_score: 50, expected_score: null},
      ],
    },
  })), language, 'receive');
  assert.match(comparison, /30.*47\.2.*10/);
  assert.match(comparison, language === 'ja' ? /最低保証点が高い.*期待値未算出/ :
    language === 'en' ? /higher guaranteed score.*estimate unavailable/ : /保底分数更高.*未获得估算/);
}
console.log('Public receive proof and guaranteed-route comparison logs passed in all three languages');
