const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const script = fs.readFileSync(path.join(__dirname, '../frontend/member.js'), 'utf8');
const tick = () => new Promise(resolve => setImmediate(resolve));
(async () => {
  const status = {textContent: ''};
  const notice = {hidden: true, textContent: ''};
  const autoSave = {checked: false, disabled: false};
  const listeners = {};
  const root = {innerHTML: '', querySelector: () => status, querySelectorAll: () => [],
    addEventListener: (name, fn) => {listeners[name] = fn;}};
  let member = {member_id: 'qa-auto', paid_active: true, must_change_password: false, auto_save_kifu: false};
  let invalidations = 0, fail = false, lastRequest;
  const context = vm.createContext({
    document: {querySelectorAll: s => s === '[data-member-panel]' ? [root] : [],
      getElementById: id => id === 'researchKifuAutoSave' ? autoSave : id === 'memberAutoKifuNotice' ? notice : {appendChild() {}}},
    fetch: async (url, options) => {
      lastRequest = {url, options};
      if (url.endsWith('kifu-auto-save')) {
        if (fail) return {ok: false, status: 500, json: async () => ({detail: 'failed'})};
        member = {...member, auto_save_kifu: JSON.parse(options.body).enabled};
      }
      return {ok: true, json: async () => ({member})};
    },
    resetMemberKifuLibrary() {}, invalidateResearchKifuStatistics() {invalidations++;},
  });
  context.window = context;
  vm.runInContext(script, context);
  await tick();
  assert.equal(autoSave.checked, false);
  assert.equal(autoSave.disabled, false);
  const input = {checked: true, disabled: false, matches: () => true};
  listeners.change({target: input});
  await tick();
  assert.equal(lastRequest.url, '/api/member/kifu-auto-save');
  assert.deepEqual(JSON.parse(lastRequest.options.body), {enabled: true});
  assert.equal(autoSave.checked, true);
  fail = true;
  input.checked = false;
  listeners.change({target: input});
  await tick();
  assert.equal(autoSave.checked, true); // Failed request must not pretend consent changed.
  context.goitaMembers.automaticKifuResult({member_id: 'other', status: 'saved'});
  assert.equal(invalidations, 0);
  context.goitaMembers.automaticKifuResult({member_id: member.member_id, status: 'saved'});
  assert.equal(invalidations, 1);
  context.goitaMembers.automaticKifuResult({member_id: member.member_id, status: 'limit'});
  assert.equal(autoSave.checked, false);
  assert.equal(notice.hidden, false);
  assert.match(notice.textContent, /1000/);
  member = null;
  await context.goitaMembers.refresh();
  assert.equal(notice.hidden, true);
  context.goitaMembers.automaticKifuResult({member_id: 'qa-auto', status: 'error'});
  assert.equal(notice.hidden, true);
  console.log('Automatic kifu UI: opt-in, request failure, owner isolation, capacity and logout passed');
})().catch(error => {console.error(error); process.exitCode = 1;});
