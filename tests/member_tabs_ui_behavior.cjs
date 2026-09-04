const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const script = fs.readFileSync(path.join(__dirname, "../frontend/member.js"), "utf8");

function makeRoot() {
  const listeners = {};
  const root = {
    listeners, tabs: [], nodes: {},
    set innerHTML(html) {
      this.html = html;
      this.nodes = {
        "[data-member-account]": { hidden: false },
        "[data-member-library]": { hidden: true },
        "[data-member-library-slot]": {},
        ".member-status": { textContent: "" },
      };
      this.tabs = html.includes('role="tablist"') ? ["account", "library"].map(action => ({
        dataset: { action }, attributes: { "aria-selected": String(action === "account") },
        tabIndex: action === "account" ? 0 : -1,
        setAttribute(name, value) { this.attributes[name] = value; },
        closest(selector) { return selector === '[role="tab"]' || selector === `[data-action="${action}"]` ? this : null; },
        focus() { root.focused = this; },
        click() { listeners.click({ target: this }); },
      })) : [];
    },
    querySelector(selector) { return this.nodes[selector]; },
    querySelectorAll(selector) { return selector === '[role="tab"]' || selector === "button" ? this.tabs : []; },
    addEventListener(name, handler) { listeners[name] = handler; },
  };
  return root;
}

(async () => {
  const roots = [makeRoot(), makeRoot()];
  let member = { member_id: "test", paid_active: true, must_change_password: false };
  let loads = 0, stops = 0, resets = 0;
  const context = vm.createContext({
    document: {
      querySelectorAll: selector => selector === "[data-member-panel]" ? roots : [],
      getElementById: () => ({ appendChild() {} }),
    },
    fetch: async () => ({ ok: true, json: async () => ({ member }) }),
    resetMemberKifuLibrary() { resets++; },
    mountMemberKifuLibrary() {},
    loadResearchKifuList() { loads++; },
    stopResearchKifuReplay() { stops++; },
  });
  context.window = context;
  vm.runInContext(script, context);
  await new Promise(resolve => setImmediate(resolve));
  for (const [index, root] of roots.entries()) {
    assert.match(root.html, /アカウント/);
    assert.ok(root.html.includes(`id="member-section-${index}-account-tab"`));
    assert.equal(root.tabs[0].attributes["aria-selected"], "true");
    root.tabs[1].click();
    assert.equal(root.nodes["[data-member-library]"].hidden, false);
    assert.equal(root.nodes["[data-member-account]"].hidden, true);
    assert.equal(root.tabs[1].tabIndex, 0);
    const previousLoads = loads;
    root.tabs[1].click();
    assert.equal(loads, previousLoads, "selected tab must not discard the open record");
    await context.goitaMembers.refresh();
    assert.equal(root.tabs[1].attributes["aria-selected"], "true");
    root.listeners.keydown({ target: root.tabs[1], key: "ArrowLeft", preventDefault() {} });
    assert.equal(root.focused, root.tabs[0]);
    assert.equal(root.nodes["[data-member-account]"].hidden, false);
    assert.equal(root.tabs[1].tabIndex, -1);
    root.listeners.keydown({ target: root.tabs[0], key: "End", preventDefault() {} });
    assert.equal(root.tabs[1].attributes["aria-selected"], "true");
    root.tabs[0].click();
  }
  assert.ok(stops >= 4);
  member = null;
  await context.goitaMembers.refresh();
  assert.ok(resets > 0);
  assert.equal(roots[0].tabs.length, 0);
  member = { member_id: "initial", must_change_password: true };
  await context.goitaMembers.refresh();
  assert.equal(roots[0].tabs.length, 0);
  assert.match(roots[0].html, /初回パスワード変更/);
  console.log("Member tabs: lobby/room, refresh, keyboard, repeat click and auth states passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
