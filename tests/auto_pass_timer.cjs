const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../frontend/index.html'), 'utf8');
const start = html.indexOf('function automaticPassContextKey(');
const end = html.indexOf('function localFollowupAttackActions(', start);
assert.ok(start > 0 && end > start);

const pass = {action_type:'pass', block:null, attack:null};
function harness() {
  const timers = new Map(), sent = [];
  let now = 0, nextTimer = 0;
  const context = vm.createContext({
    personalSettings:{autoPassWhenUnable:true}, gid:'main', currentWsGid:'main', mySeat:'A',
    latestState:{action_token:'turn-1', is_started:true, finished:false, phase:'receive',
      turn:'A', attacker:'D', current_attack:'1', round_count:1, log:['D: attack 1'],
      hands:{A:['2','3','4','5']}, owned_human_seats:['A'], ai_seats:[]},
    latestLegal:[pass], autoPlayEnabled:false, isProcessingCpu:false, pendingSubmission:false,
    automaticPassTimer:null, automaticPassTimerKey:'', automaticPassAttemptKey:'', confirmedLegalActionsKey:'',
    setTimeout(callback, delay) { const id = ++nextTimer; timers.set(id, {callback, at:now+delay}); return id; },
    clearTimeout(id) { timers.delete(id); },
    submitAction:async action => { sent.push(action); return true; },
  });
  context.isSpectator = () => context.mySeat === 'WATCH';
  context.actionIsPending = () => context.pendingSubmission;
  context.seatIsAi = (state, seat) => state.ai_seats.includes(seat);
  vm.runInContext(html.slice(start, end), context);
  const confirm = () => { context.confirmedLegalActionsKey = context.automaticPassContextKey(context.latestState); };
  confirm();
  return {context, timers, sent, confirm, async advance(ms) {
    const until = now + ms;
    for (;;) {
      const next = [...timers].filter(([, timer]) => timer.at <= until).sort((a,b) => a[1].at-b[1].at)[0];
      if (!next) break;
      now = next[1].at; timers.delete(next[0]); await next[1].callback();
    }
    now = until;
  }};
}

(async () => {
  const h = harness();
  h.context.syncAutomaticPass();
  await h.advance(600);
  h.context.syncAutomaticPass(); // A chat refresh must not extend the delay.
  await h.advance(399);
  assert.equal(h.sent.length, 0);
  await h.advance(1);
  assert.equal(h.sent.length, 1);
  assert.equal(h.sent[0], pass);
  h.context.syncAutomaticPass();
  await h.advance(5000);
  assert.equal(h.sent.length, 1, 'one attempt per confirmed turn');
  h.context.latestState.action_token = 'turn-2';
  h.confirm(); h.context.syncAutomaticPass();
  await h.advance(1000);
  assert.equal(h.sent.length, 2, 'a new turn may pass again');

  const ineligible = [
    c => { c.personalSettings.autoPassWhenUnable = false; },
    c => { delete c.personalSettings.autoPassWhenUnable; },
    c => { c.personalSettings.autoPassWhenUnable = 'true'; },
    c => { c.currentWsGid = 'lobby'; },
    c => { c.mySeat = 'WATCH'; },
    c => { c.latestState.is_started = false; },
    c => { c.latestState.finished = true; },
    c => { c.latestState.tutorial_mode = true; },
    c => { c.latestState.turn = 'B'; },
    c => { c.latestState.phase = 'attack'; },
    c => { c.latestState.current_attack = null; },
    c => { c.autoPlayEnabled = true; },
    c => { c.isProcessingCpu = true; },
    c => { c.pendingSubmission = true; },
    c => { c.latestState.owned_human_seats = []; },
    c => { c.latestState.ai_seats = ['A']; },
    c => { c.latestState.action_token = ''; },
    c => { c.confirmedLegalActionsKey = ''; },
    c => { c.latestLegal = []; },
    c => { c.latestLegal = [pass, {action_type:'receive',block:'1',attack:null}]; },
    c => { c.latestLegal = [pass, {action_type:'receive',block:'8',attack:null}]; },
    c => { c.latestLegal = [pass, {action_type:'receive',block:'9',attack:null}]; },
  ];
  for (const change of ineligible) {
    const test = harness(); change(test.context); test.context.syncAutomaticPass();
    await test.advance(1100);
    assert.equal(test.sent.length, 0);
    assert.equal(test.timers.size, 0);
  }
  // Recheck every condition at expiry, including transitions without a refresh.
  for (const change of [...ineligible,
    c => { c.gid = c.currentWsGid = 'another-room'; },
    c => { c.mySeat = 'B'; },
    c => { c.latestState.action_token = 'reset-with-identical-cards'; },
    c => { c.latestState.hands.A.push('1'); },
    c => { c.latestState.log.push('updated'); },
  ]) {
    const test = harness(); test.context.syncAutomaticPass();
    await test.advance(500); change(test.context);
    await test.advance(500);
    assert.equal(test.sent.length, 0, 'a stale timer cannot act on a different context');
  }
  const failed = harness();
  failed.context.syncAutomaticPass();
  await failed.advance(500);
  failed.context.cancelAutomaticPass(true); // Failed legal-action fetch / socket disconnect.
  failed.context.syncAutomaticPass();
  await failed.advance(2000);
  assert.equal(failed.sent.length, 0, 'old pass-only data is not a new server confirmation');
  failed.confirm(); failed.context.syncAutomaticPass();
  await failed.advance(1000);
  assert.equal(failed.sent.length, 1, 'a fresh successful response enables the timer again');
  console.log('Automatic pass timer: delay, duplicate suppression, receives/royals, tutorial, ownership, stale turns, resets, disconnects and failed confirmations passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
