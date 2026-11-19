const fs=require('node:fs');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const html=fs.readFileSync('frontend/index.html','utf8');
function extract(name) {
  const start=html.indexOf('function '+name+'(');
  assert.ok(start>=0);
  const end=html.indexOf('\nfunction ',start+1);
  return html.slice(start,end<0?undefined:end);
}
const context=vm.createContext({});
vm.runInContext(['localizeGameLogAiDetail','localizeGameLogAttackCandidates','gameLogCandidatePieceLabel']
  .map(extract).join('\n'),context);
for(const language of ['ja','en','zh']) {
  const reserve=context.localizeGameLogAiDetail('late_receive_routes_depth_10_samples_48_pass_reserve_uncertain',language);
  const plain=context.localizeGameLogAiDetail('late_receive_routes_depth_10_samples_48_receive_attack_4',language);
  const followup=context.localizeGameLogAiDetail('late_receive_followup_4',language);
  assert.ok(reserve && plain && followup);
  assert.ok(!reserve.includes('late_receive_'));
  assert.match(reserve,language==='ja'?/標本のばらつき/:language==='en'?/sample dispersion/:/样本波动/);
  assert.match(plain,language==='ja'?/確定上がりではない/:language==='en'?/not a guaranteed win/:/并非必胜证明/);
  const payload=encodeURIComponent(JSON.stringify({late_receive_comparison:{pass_value:40000,receive_value:45000,forced_round_return_mass:0.25}}));
  const summary=context.localizeGameLogAttackCandidates(payload,language,'receive');
  assert.ok(summary.includes('40000.0') && summary.includes('45000.0') && summary.includes('25%'));
  assert.match(summary,language==='ja'?/下限/:language==='en'?/lower bound/:/下限/);
}
console.log('Late receive logs: Japanese/English/Chinese selection reasons, uncertainty, actual route values and full-lap lower bound passed');
