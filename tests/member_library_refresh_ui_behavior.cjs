const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const tick = () => new Promise(resolve => setImmediate(resolve));
(async () => {
  const listeners = {};
  const account = {}, library = {}, slot = {}, status = {};
  const root = {innerHTML: '',
    querySelector: s => ({'[data-member-account]': account, '[data-member-library]': library, '[data-member-library-slot]': slot}[s] || status),
    querySelectorAll: () => [], addEventListener: (name, fn) => {listeners[name] = fn;},
  };
  const nodes = {};
  let member = {member_id: 'refresh-test', paid_active: true, must_change_password: false};
  let loads = 0, resets = 0;
  const context = vm.createContext({
    document: {querySelectorAll: s => s === '[data-member-panel]' ? [root] : [],
      getElementById: id => nodes[id] ||= {appendChild() {}}},
    fetch: async () => ({ok: true, json: async () => ({member})}),
    resetMemberKifuLibrary: () => resets++, loadResearchKifuList: () => loads++,
    resetMemberKifuDisclosures() {}, mountMemberKifuLibrary() {}, stopResearchKifuReplay() {},
  });
  context.window = context;
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../frontend/member.js'), 'utf8'), context);
  await tick();
  const click = action => listeners.click({target: {closest: selector => selector === `[data-action="${action}"]`}});
  click('library');
  assert.equal(loads, 1);
  const oldResets = resets;
  await context.goitaMembers.refresh({reloadLibrary: true, root});
  assert.equal(loads, 2);
  assert.equal(resets, oldResets + 1); // detail/replay/cache are reset on reopen
  assert.equal(library.hidden, false);
  await context.goitaMembers.refresh();
  assert.equal(loads, 2); // ordinary session refresh does not interrupt the library
  click('account');
  await context.goitaMembers.refresh({reloadLibrary: true, root});
  assert.equal(loads, 2); // account remains selected
  click('library');
  member = {...member, member_id: 'different-member'};
  await context.goitaMembers.refresh({reloadLibrary: true, root});
  assert.equal(loads, 3); // no old member selection reused after account change
  console.log('Library refresh UI: reopen reloads, session refresh preserves, account selection and owner isolation passed');
})().catch(error => {console.error(error); process.exitCode = 1;});
