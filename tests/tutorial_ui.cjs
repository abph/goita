const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../frontend/index.html'), 'utf8');
for (const script of html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)) {
  new vm.Script(script[1]);
}
function extract(name) {
  const start = html.search(new RegExp(`(?:async )?function ${name}\\(`));
  assert.ok(start >= 0, name);
  const rest = html.slice(start);
  const end = rest.slice(1).search(/\n(?:async )?function /);
  return end < 0 ? rest : rest.slice(0, end + 1);
}
function context() {
  const nodes = new Map();
  return vm.createContext({
    URL, console,
    document: {
      body: {classList: {toggle() {}}},
      getElementById(id) {
        if (!nodes.has(id)) nodes.set(id, {hidden: true, style: {}});
        return nodes.get(id);
      },
    },
    saveTutorialProgress() {},
    tutorialCopy: () => ['title', 'instruction', 'detail'],
  });
}
(async () => {
  const c = context();
  for (const name of ['tutorialChapter', 'tutorialChapterTotal', 'syncTutorialUI']) vm.runInContext(extract(name), c);
  for (const [chapter, completed, explain, royal, practice] of [
    ['basic', false, false, false, false],
    ['basic', true, true, true, false],
    ['royal', false, false, false, false],
    ['royal', true, false, false, true],
  ]) {
    c.syncTutorialUI({tutorial_mode: true, tutorial_chapter: chapter, tutorial_completed: completed});
    assert.equal(!c.document.getElementById('tutorialRoyalExplanation').hidden, explain);
    assert.equal(!c.document.getElementById('tutorialRoyalButton').hidden, royal);
    assert.equal(!c.document.getElementById('tutorialAiPracticeButton').hidden, practice);
  }
  const nav = context();
  let calls = 0;
  nav.startTutorial = async () => {calls++; return true;};
  nav.window = {location: {href: 'https://solo.vrcgoita.com/?tutorial=1&lang=en&from=vrcgoita#guide'},
    history: {replaceState(_state, _title, url) {nav.window.location.href = new URL(url, nav.window.location.href).href;}}};
  vm.runInContext(extract('autoEnterTutorialFromUrl'), nav);
  assert.equal(await nav.autoEnterTutorialFromUrl(), true);
  assert.equal(calls, 1);
  assert.equal(nav.window.location.href, 'https://solo.vrcgoita.com/?lang=en&from=vrcgoita#guide');
  assert.equal(await nav.autoEnterTutorialFromUrl(), false);
  assert.equal(calls, 1);
  nav.window.location.href = 'https://solo.vrcgoita.com/?tutorial=0';
  assert.equal(await nav.autoEnterTutorialFromUrl(), false);
  nav.window.location.href = 'https://solo.vrcgoita.com/?tutorial=1';
  nav.startTutorial = async () => false;
  assert.equal(await nav.autoEnterTutorialFromUrl(), false);

  const start = context();
  let sent;
  let entered;
  Object.assign(start, {
    tutorialStarting: false, clientId: 'test-tutorial', gid: '',
    uiText: x => x, enableBeginnerSupportFromGuide() {},
    storedTutorialProgress: () => 2,
    resetLogAudioSync() {},
    enterRoom: async (...args) => {entered = args;},
    fetch: async (_url, opts) => {sent = JSON.parse(opts.body); return {ok: true, json: async () => ({game_id: 'tutorial-test'})};},
    API: '',
  });
  for (const name of ['tutorialChapter', 'tutorialChapterTotal', 'startTutorial']) vm.runInContext(extract(name), start);
  assert.equal(await start.startTutorial(), true);
  assert.equal(sent.start_step, 3);
  assert.equal(sent.chapter, 'basic');
  assert.equal(entered[0], 'tutorial-test');
  start.storedTutorialProgress = () => 5;
  assert.equal(await start.startTutorial(), true);
  assert.equal(sent.start_step, 1);
  start.fetch = async () => ({ok: false, json: async () => ({detail: 'offline'})});
  assert.equal(await start.startTutorial(), false);
  assert.equal(start.document.getElementById('lobbyFeatureGuideStatus').textContent, 'offline');
  assert.equal(start.tutorialStarting, false);

  const auto = context();
  Object.assign(auto, {cancelAutomaticPass() {}, isSpectator: () => false,
    isPersonalScoreRoom: () => false, isVrcSoloRoom: () => false,
    latestState: {tutorial_mode: true}, updateAutoButtons() {}});
  vm.runInContext(extract('setAutoPlayEnabled'), auto);
  auto.setAutoPlayEnabled(true);
  assert.equal(auto.autoPlayEnabled, false);
  auto.latestState = {tutorial_mode: false};
  auto.setAutoPlayEnabled(true);
  assert.equal(auto.autoPlayEnabled, true);
  assert.ok(!html.includes('チュートリアルを終了する'));
  assert.ok(!html.includes('tutorialExitButton'));
  assert.ok(!html.includes('body.tutorial-room #chatPanel'));
  assert.ok(!html.includes('body.tutorial-room #chatToggle'));
  assert.ok(html.includes('!state.vrc_web && !state.tutorial_mode'));
  assert.ok(html.includes('王（玉）の使い方'));
  console.log('Tutorial UI: direct entry, resume, failure fallback, chapter buttons, and Auto guards passed');
})().catch(error => {console.error(error); process.exitCode = 1;});
