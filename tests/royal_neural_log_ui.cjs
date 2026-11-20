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
const context = vm.createContext({GAME_LOG_AI_DETAIL_LABELS: {ja: {}, en: {}, zh: {}}});
vm.runInContext(['localizeGameLogAiDetail', 'localizeGameLogAttackCandidates']
    .map(extract).join('\n'), context);
for (const language of ['ja', 'en', 'zh']) {
    for (const suffix of ['receive_use_or_reserve', 'attack_or_block_use_or_reserve']) {
        const message = context.localizeGameLogAiDetail('royal_neural_priority_' + suffix, language);
        assert.ok(message && !message.includes('royal_neural_priority_'));
        assert.match(message, language === 'ja' ? /僅差条件を外して/ : language === 'en' ? /without the close-score restriction/ : /不限于/);
    }
    const message = context.localizeGameLogAttackCandidates(encodeURIComponent(JSON.stringify({
        neural_royal_decision: {selected_action: ['pass', null, null], rule_action: ['receive', '8', null]},
        late_receive_comparison: {status: 'incomplete'}
    })), language, 'receive');
    assert.match(message, /NEURAL-ROYAL/);
    assert.match(message, language === 'ja' ? /証明ではありません/ : language === 'en' ? /not a forced-win proof/ : /不是必胜证明/);
}
console.log('Royal neural log: Japanese, English and Chinese passed');
