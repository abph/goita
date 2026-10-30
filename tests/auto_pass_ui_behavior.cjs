const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const pass = {action_type:'pass',block:null,attack:null};
let token = 0;
function initial(changes = {}) {
  return {action_token:`auto-pass-${++token}`, update_version:0, is_started:true, finished:false,
    turn:'A', phase:'receive', attacker:'D', current_attack:'1', dealer:'D', round_count:1,
    owned_human_seats:['A'], human_seats:['A','B','C','D'], ai_seats:[],
    hands:{A:['2','3','4','5','6','7','8','9'], B:{count:8}, C:{count:8}, D:{count:6}},
    init_hands:{A:['2','3','4','5','6','7','8','9']}, face_down_pieces:{A:[]},
    board_public:Object.fromEntries('ABCD'.split('').map(seat => [seat, {
      receive:[null,null,null,null], attack:[null,null,null,null], receive_hidden:[false,false,false,false],
    }])), log:['D: block 1 -> attack 1 (hidden)'], player_names:{A:'テスト',B:'B',C:'C',D:'D'},
    total_team_score:{AC:0,BD:0}, revealed_hand_seats:[], chat_messages:[], ...changes};
}

(async () => {
  const browser = await chromium.launch({channel:'msedge', headless:true});
  try {
    const page = await browser.newPage({viewport:{width:1100,height:900}});
    const errors = [], posts = [];
    let state = initial(), legal = [pass], legalFailure = false;
    page.on('pageerror', error => errors.push(error.message));
    await page.routeWebSocket('**/*', () => {});
    await page.route('**/*', async route => {
      const url = new URL(route.request().url());
      if(url.hostname !== 'goita.test') return route.abort();
      if(url.pathname === '/' || url.pathname.startsWith('/static/')) {
        const root = path.resolve(__dirname, '../frontend');
        const file = path.resolve(root, url.pathname === '/' ? 'index.html' : url.pathname.slice(8));
        return file.startsWith(root + path.sep) && fs.existsSync(file)
          ? route.fulfill({path:file}) : route.fulfill({status:404,body:''});
      }
      if(url.pathname.endsWith('/legal_actions')) return legalFailure
        ? route.fulfill({status:503,json:{detail:'temporarily unavailable'}}) : route.fulfill({json:legal});
      if(url.pathname.endsWith('/state')) return route.fulfill({json:state});
      if(url.pathname.endsWith('/step')) {
        const body = route.request().postDataJSON(); posts.push(body);
        if(body.expected_action_token !== state.action_token) return route.fulfill({status:409,json:{detail:'stale'}});
        state = {...state, action_token:`accepted-${++token}`, update_version:1, turn:'B', log:[...state.log,'A: pass']};
        legal = [];
        return route.fulfill({json:{ok:true,state}});
      }
      return route.fulfill({json:url.pathname === '/api/member/session'
        ? {member:null} : {rooms:[],site_people:[],public_chat_messages:[]}});
    });
    await page.goto('http://goita.test/');
    await page.waitForFunction(() => typeof personalSettings !== 'undefined');
    assert.equal(await page.evaluate(() => personalSettings.autoPassWhenUnable),false);
    await page.evaluate(() => {
      openLobbySettings();
      document.getElementById('lobbyPlaySupportSettingsDetails').open = true;
    });
    assert.equal(await page.locator('#lobbyCheckAutoPassWhenUnable').isChecked(),false);
    await page.locator('#lobbyCheckAutoPassWhenUnable').check();
    await page.evaluate(() => saveLobbyPersonalSettings());
    assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem(PERSONAL_SETTINGS_KEY)).autoPassWhenUnable),true);
    await page.reload();
    await page.waitForFunction(() => typeof personalSettings !== 'undefined');
    assert.equal(await page.evaluate(() => personalSettings.autoPassWhenUnable),true);
    for (const [language, expected] of [['en','Automatically pass when you cannot receive'], ['zh','无法接牌时自动跳过']]) {
      assert.equal(await page.evaluate(language => {
        goitaI18n.setLanguage(language, {dispatch:false});
        return goitaI18n.translate('受けられないとき、自動でパスする');
      }, language),expected);
    }
    await page.evaluate(() => {
      goitaI18n.setLanguage('ja', {dispatch:false}); closeLobbySettings();
      window.autoPassSounds = 0; window.autoPassDisplays = 0;
      playPassSound = () => window.autoPassSounds++;
      const original = showPassAnimation;
      showPassAnimation = phys => {window.autoPassDisplays++; original(phys);};
    });

    async function setup({room='main-e', changes={}, actions=[pass], enabled=true} = {}) {
      state = initial(changes); legal = actions; legalFailure = false;
      await page.evaluate(async ({snapshot,room,enabled}) => {
        cancelImmediateAction(); clearLegalActionsRetry();
        gid = currentWsGid = activeRoomId = room; mySeat = 'A';
        isProcessingCpu = false; autoPlayEnabled = false; latestState = null; latestLegal = [];
        personalSettings.autoPassWhenUnable = enabled;
        personalSettings.autoRevealOwnHand = false; personalSettings.autoRevealAiHands = false;
        personalSettings.enableBeginnerSupport = false; personalSettings.enableSoundEffects = true;
        personalSettings.enableEffects = false; audioUnlocked = true;
        fixedHandMemory = {}; pending = null; lastLogCount = snapshot.log.length;
        logAudioPrimed = true; earlyPassLogKeys.clear();
        document.getElementById('gameView').style.display = 'block';
        document.getElementById('lobbyView').style.display = 'none';
        await refreshFromState(snapshot);
      }, {snapshot:state,room,enabled});
    }
    async function waitForPosts(count) {
      for(let i=0;posts.length<count && i<160;i++) await page.waitForTimeout(10);
      assert.equal(posts.length,count);
      await page.waitForTimeout(60);
    }
    for(const room of ['main-e','room-gold-01','score-auto-pass']) {
      const before = posts.length;
      await setup({room});
      const sounds = await page.evaluate(() => window.autoPassSounds);
      const displays = await page.evaluate(() => window.autoPassDisplays);
      await page.waitForTimeout(500);
      assert.equal(posts.length,before,'wait briefly before passing');
      await page.evaluate(() => refresh()); // Same-turn refresh must not postpone the pass.
      await waitForPosts(before+1);
      assert.deepEqual(posts.at(-1).action,pass);
      assert.match(posts.at(-1).expected_action_token,/^auto-pass-/);
      assert.equal(await page.evaluate(() => latestState.log.at(-1)),'A: pass');
      assert.equal(await page.evaluate(() => window.autoPassSounds),sounds+1);
      assert.equal(await page.evaluate(() => window.autoPassDisplays),displays+1);
    }

    // Natural receive and royal receive must remain a player's choice.
    for(const block of ['1','8','9']) {
      await setup({changes:{current_attack:block === '1' ? '1' : '3'},
        actions:[pass,{action_type:'receive',block,attack:null}]});
      const before = posts.length;
      await page.waitForTimeout(1100);
      assert.equal(posts.length,before);
    }
    await setup({changes:{tutorial_mode:true,tutorial_step:2}});
    let before = posts.length;
    await page.waitForTimeout(1100);
    assert.equal(posts.length,before,'tutorial pass must stay manual');

    // A failed legal request must cancel a previously armed pass, even if the
    // old render still has pass-only buttons available during retries.
    await setup(); before = posts.length;
    legalFailure = true;
    await page.evaluate(() => refresh());
    await page.waitForTimeout(1100);
    assert.equal(posts.length,before);
    assert.equal(await page.evaluate(() => confirmedLegalActionsKey),'');
    legalFailure = false;
    await page.evaluate(() => refresh());
    await waitForPosts(before+1);

    // Saving OFF in the actual room settings cancels an armed timer.
    await setup(); before = posts.length;
    await page.evaluate(async () => {
      syncPersonalSettingsControls();
      document.getElementById('checkAutoPassWhenUnable').checked = false;
      await savePersonalSettings();
    });
    await page.waitForTimeout(1100);
    assert.equal(posts.length,before);
    assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem(PERSONAL_SETTINGS_KEY)).autoPassWhenUnable),false);

    // A simultaneous manual click uses one standard submission, never two.
    await setup(); before = posts.length;
    await page.evaluate(() => onPassClick());
    await waitForPosts(before+1);
    await page.waitForTimeout(1100);
    assert.equal(posts.length,before+1);

    // Leaving a room invalidates the pending timer before any asynchronous
    // voice/seat work can finish.
    await setup(); before = posts.length;
    await page.evaluate(() => returnToLobby());
    await page.waitForTimeout(1100);
    assert.equal(posts.length,before);
    await page.setViewportSize({width:390,height:844});
    await page.evaluate(() => {
      openLobbySettings();
      document.getElementById('lobbyPlaySupportSettingsDetails').open = true;
    });
    await page.locator('#lobbyCheckAutoPassWhenUnable').scrollIntoViewIfNeeded();
    assert.equal(await page.locator('#lobbyCheckAutoPassWhenUnable').isVisible(),true);
    assert.equal(await page.locator('#lobbyCheckAutoPassWhenUnable').isChecked(),true);
    assert.deepEqual(errors,[]);
    console.log('Automatic pass UI: default OFF, both settings/save/reload, translations, public/private/score rooms, royal receives, tutorial, request failure, manual clicks, normal sounds/logs, and room exit passed.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
