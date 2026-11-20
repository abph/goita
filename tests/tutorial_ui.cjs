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
      body: {classList: {toggle() {}, contains() {return false;}}},
      getElementById(id) {
        if (!nodes.has(id)) nodes.set(id, {hidden: true, style: {}});
        return nodes.get(id);
      },
    },
    saveTutorialProgress() {},
    uiText: x => x,
    isAnimalTutorial: state => state?.tutorial_mode === true && ['basic','royal'].includes(state.tutorial_chapter) && state.tutorial_piece_theme === 'animals',
    currentUiLanguage: () => 'ja', window: {goitaTutorialAnimals: {renderRoyalExplanation() {}}},
    applyBoardViewMode() {}, renderTutorialAnimalLegend() {},
    tutorialCopy: () => ['title', 'instruction', 'detail'],
  });
}
(async () => {
  const c = context();
  for (const name of ['tutorialChapter', 'tutorialChapterTotal', 'syncTutorialUI']) vm.runInContext(extract(name), c);
  for (const [chapter, completed, explain, royal, attack, cooperation, practice] of [
    ['basic', false, false, false, false, false, false],
    ['basic', true, true, true, false, false, false],
    ['royal', false, false, false, false, false, false],
    ['royal', true, false, false, true, false, true],
    ['attack', false, false, false, false, false, false],
    ['attack', true, false, false, false, true, false],
    ['cooperation', false, false, false, false, false, false],
    ['cooperation', true, false, false, false, false, true],
  ]) {
    c.syncTutorialUI({tutorial_mode: true, tutorial_chapter: chapter, tutorial_completed: completed});
    assert.equal(!c.document.getElementById('tutorialRoyalExplanation').hidden, explain);
    assert.equal(!c.document.getElementById('tutorialRoyalButton').hidden, royal);
    assert.equal(!c.document.getElementById('tutorialAttackButton').hidden, attack);
    assert.equal(!c.document.getElementById('tutorialCooperationButton').hidden, cooperation);
    assert.equal(!c.document.getElementById('tutorialAiPracticeButton').hidden, practice);
  }
  c.syncTutorialUI({tutorial_mode:true,tutorial_chapter:'basic',tutorial_piece_theme:'animals',tutorial_completed:true});
  assert.equal(c.document.getElementById('tutorialRoyalExplanation').hidden,true);
  assert.equal(c.document.getElementById('tutorialAnimalLegend').hidden,false);
  assert.equal(c.document.getElementById('tutorialRoyalButton').hidden,false);
  assert.ok(c.document.getElementById('tutorialRoyalButton').textContent.includes('やまねこ'));
  assert.equal(c.document.getElementById('tutorialAnimalRoyalExplanation').hidden,false);
  c.syncTutorialUI({tutorial_mode:true,tutorial_chapter:'royal',tutorial_piece_theme:'animals',tutorial_completed:true});
  assert.equal(c.document.getElementById('tutorialAnimalLegend').hidden,false);
  assert.equal(c.document.getElementById('tutorialAnimalRoyalExplanation').hidden,true);
  assert.ok(c.document.getElementById('tutorialAttackButton').textContent.includes('通常の駒'));
  const royalLink = context();
  let royalRequest;
  royalLink.startTutorial = async (...args) => {royalRequest = args;};
  vm.runInContext(extract('startRoyalTutorial'),royalLink);
  royalLink.latestState = {tutorial_mode:true,tutorial_chapter:'basic',tutorial_piece_theme:'animals'};
  await royalLink.startRoyalTutorial();
  assert.deepEqual(royalRequest,['royal',true,'animals']);
  royalLink.latestState.tutorial_piece_theme = 'standard';
  await royalLink.startRoyalTutorial();
  assert.deepEqual(royalRequest,['royal',true,'standard']);
  const nav = context();
  let calls = 0;
  let requestedChapter;
  nav.startTutorial = async chapter => {calls++; requestedChapter = chapter; return true;};
  nav.window = {location: {href: 'https://solo.vrcgoita.com/?tutorial=1&lang=en&from=vrcgoita#guide'},
    history: {replaceState(_state, _title, url) {nav.window.location.href = new URL(url, nav.window.location.href).href;}}};
  vm.runInContext(extract('autoEnterTutorialFromUrl'), nav);
  assert.equal(await nav.autoEnterTutorialFromUrl(), true);
  assert.equal(calls, 1);
  assert.equal(requestedChapter, 'basic');
  assert.equal(nav.window.location.href, 'https://solo.vrcgoita.com/?lang=en&from=vrcgoita#guide');
  assert.equal(await nav.autoEnterTutorialFromUrl(), false);
  assert.equal(calls, 1);
  nav.window.location.href = 'https://solo.vrcgoita.com/?tutorial=0';
  assert.equal(await nav.autoEnterTutorialFromUrl(), false);
  for (const chapter of ['royal', 'attack', 'cooperation']) {
    nav.window.location.href = `https://solo.vrcgoita.com/?tutorial=${chapter}`;
    assert.equal(await nav.autoEnterTutorialFromUrl(), true);
    assert.equal(requestedChapter, chapter);
  }
  let requestedTheme;
  nav.window.location.href = 'https://solo.vrcgoita.com/?tutorial=animals';
  nav.startTutorial = async (chapter, stay, theme) => {requestedChapter = chapter; requestedTheme = theme; return true;};
  assert.equal(await nav.autoEnterTutorialFromUrl(), true);
  assert.equal(requestedChapter,'basic');
  assert.equal(requestedTheme,'animals');
  nav.window.location.href = 'https://solo.vrcgoita.com/?tutorial=animals-royal';
  assert.equal(await nav.autoEnterTutorialFromUrl(),true);
  assert.equal(requestedChapter,'royal');
  assert.equal(requestedTheme,'animals');
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
  start.storedTutorialProgress = (_chapter, theme) => theme === 'animals' ? 1 : 5;
  assert.equal(await start.startTutorial('basic',false,'animals'),true);
  assert.equal(sent.start_step,2);
  assert.equal(sent.piece_theme,'animals');
  assert.equal(await start.startTutorial('royal',false,'animals'),true);
  assert.equal(sent.chapter,'royal');
  assert.equal(sent.piece_theme,'animals');
  start.fetch = async () => ({ok: false, json: async () => ({detail: 'offline'})});
  assert.equal(await start.startTutorial(), false);
  assert.equal(start.document.getElementById('lobbyFeatureGuideStatus').textContent, 'offline');
  assert.equal(start.tutorialStarting, false);

  const copy = context();
  vm.runInContext(extract('tutorialChapter'), copy);
  vm.runInContext(extract('tutorialCopy'), copy);
  for (const language of ['ja', 'en', 'zh']) {
    copy.currentUiLanguage = () => language;
    for (const [chapter, indices] of [['attack', [0, 5, 6, 11, 12, 16]],
                                   ['cooperation', [3, 4, 9, 10, 14, 18]]]) {
      for (const index of indices) {
        const result = copy.tutorialCopy({tutorial_chapter: chapter, turn: 'A', tutorial_action_index: index});
        assert.equal(result.length, 3);
        assert.ok(result.every(text => typeof text === 'string' && text.length > 0));
      }
      const done = copy.tutorialCopy({tutorial_chapter: chapter, tutorial_completed: true});
      assert.ok(done[0] && done[1]);
    }
  }
  const progress = context();
  vm.runInContext(html.slice(html.indexOf('const TUTORIAL_PROGRESS_KEYS ='), html.indexOf('let tutorialStarting =')), progress);
  const saved = new Map();
  progress.localStorage = {getItem: key => saved.get(key), setItem: (key, value) => saved.set(key, value)};
  for (const name of ['tutorialChapter', 'tutorialChapterTotal', 'tutorialProgressKey', 'storedTutorialProgress', 'saveTutorialProgress'])
    vm.runInContext(extract(name), progress);
  progress.saveTutorialProgress('basic', 5);
  progress.saveTutorialProgress('attack', 2);
  progress.saveTutorialProgress('cooperation', 1);
  assert.equal(progress.storedTutorialProgress('basic'), 5);
  progress.saveTutorialProgress('royal',2,'animals');
  assert.equal(progress.storedTutorialProgress('royal','animals'),2);
  assert.equal(progress.storedTutorialProgress('royal'),0);
  assert.equal(progress.storedTutorialProgress('attack'), 2);
  assert.equal(progress.storedTutorialProgress('cooperation'), 1);
  assert.equal(progress.storedTutorialProgress('royal'), 0);
  assert.equal(progress.storedTutorialProgress('basic', 'animals'), 0);
  progress.saveTutorialProgress('basic', 2, 'animals');
  assert.equal(progress.storedTutorialProgress('basic', 'animals'), 2);
  assert.equal(progress.storedTutorialProgress('basic'), 5);

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
