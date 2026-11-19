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
  for(const piece of ['2','3','4','5']) {
    const pair=context.localizeGameLogAiDetail(`receive_enemy_first_shared_pair_${piece}`,language);
    const pass=context.localizeGameLogAiDetail(`pass_enemy_first_shared_pair_${piece}`,language);
    assert.ok(!pair.includes('shared_pair') && !pass.includes('shared_pair'));
    assert.match(pair,language==='ja'?/推定/:language==='en'?/inferred/:/推测/);
  }
  const pairValues=context.localizeGameLogAttackCandidates(encodeURIComponent(JSON.stringify({
    opening_shared_pair:{scores:[{action:['pass',null,null],score:-26},{action:['receive','4',null],score:67}]}
  })),language,'receive');
  assert.ok(pairValues.includes('-26.0') && pairValues.includes('67.0'));
  assert.match(pairValues,language==='ja'?/勝率ではない/:language==='en'?/not win probabilities/:/并非胜率/);
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
  const limited=context.localizeGameLogAiDetail('late_receive_routes_depth_6_samples_48_pass_budget_limited',language);
  assert.ok(limited.includes('6') && !limited.includes('late_receive_'));
  assert.match(limited,language==='ja'?/上限/:language==='en'?/limit/:/限制/);
  const incomplete=context.localizeGameLogAttackCandidates(encodeURIComponent(JSON.stringify({
    late_receive_comparison:{status:'incomplete',stop_reason:'time_limit',applied:false}
  })),language,'receive');
  assert.match(incomplete,language==='ja'?/元の判断を維持/:language==='en'?/Kept the original/:/保留原判断/);
  assert.ok(!incomplete.includes('NaN') && !incomplete.includes('undefined'));
  const notAdopted=context.localizeGameLogAttackCandidates(encodeURIComponent(JSON.stringify({
    late_receive_comparison:{status:'budget_limited',depth:6,pass_value:40000,receive_value:45000,applied:false}
  })),language,'receive');
  assert.ok(notAdopted.includes('6'));
  assert.match(notAdopted,language==='ja'?/採用せず/:language==='en'?/not adopted/:/未采用/);
}
console.log('Late receive logs: Japanese/English/Chinese selection reasons, uncertainty, actual route values and full-lap lower bound passed');
