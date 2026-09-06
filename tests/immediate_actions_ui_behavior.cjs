const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const clone = value => JSON.parse(JSON.stringify(value));

function initial(phase = 'attack') {
  return {action_token: 'before', update_version: 0, is_started: true, finished: false, turn: 'A', phase,
    attacker: phase === 'receive' ? 'D' : null, current_attack: phase === 'receive' ? '1' : null,
    dealer: 'A', round_count: 1, owned_human_seats: ['A'], human_seats: ['A','B','C','D'], ai_seats: [],
    hands: {A: ['1','1','1','2','3','4','5','9'], B:{count:8}, C:{count:8}, D:{count:8}},
    init_hands: {A: ['1','1','1','2','3','4','5','9']}, face_down_pieces: {A: []},
    board_public: Object.fromEntries(['A','B','C','D'].map(seat => [seat, {receive:[null,null,null,null], attack:[null,null,null,null], receive_hidden:[false,false,false,false]}])),
    log: ['Started'], player_names: {A:'テスト', B:'相手B', C:'相手C', D:'相手D'}, total_team_score:{AC:0,BD:0},
    revealed_hand_seats: [], chat_messages: []};
}

(async () => {
  const browser = await chromium.launch({channel:'msedge', headless:true});
  try {
    const page = await browser.newPage({viewport:{width:1100,height:900}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    let state = initial(), legal = [], posts = [], heldStates = [], heldLegal = [];
    let holdState = false, holdLegal = false, stateReads = 0;
    await page.routeWebSocket('**/*', socket => socket.close());
    await page.route('**/*', async route => {
      const url = new URL(route.request().url());
      if(url.hostname !== 'goita.test') return route.abort();
      if(url.pathname === '/' || url.pathname.startsWith('/static/')) {
        const root = path.resolve(__dirname, '../frontend');
        const file = path.resolve(root, url.pathname === '/' ? 'index.html' : url.pathname.slice(8));
        return file.startsWith(root + path.sep) && fs.existsSync(file) ? route.fulfill({path:file}) : route.fulfill({status:404,body:''});
      }
      if(url.pathname.endsWith('/step')) {posts.push(route); return;}
      if(url.pathname.endsWith('/state')) {
        stateReads++;
        if(holdState) {heldStates.push(route); return;}
        return route.fulfill({json:state});
      }
      if(url.pathname.endsWith('/legal_actions')) {
        if(holdLegal) {heldLegal.push(route); return;}
        return route.fulfill({json:legal});
      }
      return route.fulfill({json:url.pathname === '/api/member/session' ? {member:null} : {rooms:[],site_people:[],public_chat_messages:[]}});
    });
    await page.goto('http://goita.test/');
    await page.waitForFunction(() => typeof latestState !== 'undefined');
    await page.evaluate(() => {
      window.pieceSoundsPlayed = 0; window.passesShown = 0; window.passSoundsPlayed = 0;
      playPieceSound = () => window.pieceSoundsPlayed++;
      playPassSound = () => window.passSoundsPlayed++;
      const original = showPassAnimation;
      showPassAnimation = phys => {window.passesShown++; original(phys);};
      window.lastAlternateBoard = null;
      renderBoard3D = state => {window.lastAlternateBoard = structuredClone(state);};
    });
    async function setup(nextState, actions) {
      state = clone(nextState); legal = actions;
      await page.evaluate(({state, legal}) => {
        cancelImmediateAction(); clearLegalActionsRetry();
        gid = 'immediate-test'; mySeat = 'A'; activeRoomId = gid; isProcessingCpu = false;
        latestState = state; latestLegal = legal; pending = null; fixedHandMemory = {};
        isGameStarted = true; lastLogCount = state.log.length; logAudioPrimed = true; audioUnlocked = true; suppressLogAudioUntil = 0;
        earlyPieceSoundLogKeys.clear(); earlyPassLogKeys.clear();
        personalSettings.enableEffects = false; personalSettings.enableSoundEffects = true;
        personalSettings.enableBeginnerSupport = false; personalSettings.autoRevealOwnHand = false; personalSettings.autoRevealAiHands = false;
        document.getElementById('gameView').style.display = 'block';
        document.getElementById('lobbyView').style.display = 'none';
        closeChatPanel('lobby'); closeChatPanel('room');
        renderBoard(state); renderHands(state); renderActions();
      }, {state, legal});
    }
    async function waitForPost(count) {
      await page.waitForFunction(() => actionIsPending());
      for(let i=0;posts.length<count && i<100;i++) await new Promise(resolve => setTimeout(resolve,10));
      assert.equal(posts.length,count);
    }

    const combined = {action_type:'attack_after_block',block:'1',attack:'1'};
    await setup(initial(), [combined]);
    holdState = true;
    await page.evaluate(() => {window.oldRefresh = refresh();});
    for(let i=0;heldStates.length===0 && i<100;i++) await new Promise(resolve => setTimeout(resolve,10));
    await page.evaluate(() => onHandPieceClick('1', 0));
    assert.equal(posts.length,0,'selecting the hidden piece is only a preview');
    await page.evaluate(() => {window.movePromise = onHandPieceClick('1',1);});
    await waitForPost(1);
    assert.deepEqual(await page.evaluate(() => immediateActionView(latestState).board_public.A.attack), ['1',null,null,null]);
    assert.equal(await page.evaluate(() => latestState.hands.A.length),8,'authoritative data must stay untouched');
    assert.equal(await page.locator('#handsArea .hand.face-down').count(),2);
    assert.equal(await page.locator('#board .cell.attack.slot').count(),1);
    assert.equal(await page.evaluate(() => window.lastAlternateBoard.hands.A.length),6);
    assert.equal(await page.evaluate(() => window.pieceSoundsPlayed),1);
    await page.setViewportSize({width:390,height:844});
    fs.mkdirSync(path.resolve(__dirname,'../results'),{recursive:true});
    await page.locator('.board-wrap').screenshot({path:path.resolve(__dirname,'../results/immediate-action-mobile.png')});
    await page.setViewportSize({width:1100,height:900});
    assert.equal(await page.evaluate(combined => submitAction(combined), combined),false,'rapid repeated clicks must not send');
    const readsBeforeAck = stateReads;
    await page.evaluate(() => refresh());
    assert.equal(stateReads,readsBeforeAck,'WebSocket notification must not duplicate the action response fetch');
    await heldStates.shift().fulfill({json:initial()}); holdState = false;
    await page.evaluate(() => window.oldRefresh);
    assert.equal(await page.locator('#board .cell.attack.slot').count(),1,'an older GET must not erase the immediate move');
    assert.equal(posts[0].request().postDataJSON().expected_action_token,'before');
    state = await page.evaluate(() => structuredClone(immediateActionView(latestState)));
    Object.assign(state,{action_token:'after',update_version:1,turn:'B',phase:'receive',log:['Started','A: block 1 -> attack 1 (hidden)']});
    await page.evaluate(() => queueImmediateActionUpdate({action_token:'after',update_version:1}));
    legal = []; holdLegal = true;
    await posts[0].fulfill({json:{ok:true,state}});
    await page.waitForFunction(() => latestState.action_token === 'after');
    assert.equal(await page.evaluate(() => window.pieceSoundsPlayed),1,'confirmed logs must not play the same sound again');
    assert.equal(stateReads,readsBeforeAck);
    for(let i=0;heldLegal.length===0 && i<100;i++) await new Promise(resolve => setTimeout(resolve,10));
    holdLegal = false; await heldLegal.shift().fulfill({json:legal});
    await page.evaluate(() => window.movePromise);

    const pass = {action_type:'pass',block:null,attack:null};
    await setup(initial('receive'),[pass]);
    const priorPasses = await page.evaluate(() => window.passesShown);
    await page.evaluate(() => {window.movePromise = onPassClick();});
    await waitForPost(2);
    assert.equal(await page.locator('.pass-anim-wrapper').count(),1,'pass is visible while POST is still held');
    assert.equal(await page.evaluate(() => window.passesShown),priorPasses+1);
    assert.equal(await page.locator('#handsArea button.pass').isDisabled(),true);
    state = {...state,action_token:'pass-done',turn:'B',log:['Started','A: pass']}; legal = [];
    holdLegal = true;
    await posts[1].fulfill({json:{ok:true,state}});
    await page.waitForFunction(() => latestState.action_token === 'pass-done');
    assert.equal(await page.evaluate(() => window.passesShown),priorPasses+1,'slow legal lookup must not replay the pass');
    for(let i=0;heldLegal.length===0 && i<100;i++) await new Promise(resolve => setTimeout(resolve,10));
    holdLegal = false; await heldLegal.shift().fulfill({json:[]});
    await page.evaluate(() => window.movePromise);

    await setup(initial(),[combined]);
    await page.evaluate(combined => {window.movePromise = submitAction(combined,[{slotIndex:0,piece:'1',hidden:true},{slotIndex:1,piece:'1'}]);},combined);
    await waitForPost(3);
    await posts[2].fulfill({status:409,json:{detail:'stale'}});
    await page.evaluate(() => window.movePromise);
    assert.equal(await page.locator('#board .cell.attack.slot').count(),0,'rejection restores the confirmed board');
    assert.equal(await page.locator('#handsArea .hand.face-down').count(),0,'rejection restores duplicate-piece slots');

    await setup(initial('receive'),[pass]);
    await page.evaluate(() => {window.movePromise = onPassClick();});
    await waitForPost(4);
    state = {...state,action_token:'recovered',turn:'B',log:['Started','A: pass']}; legal = [];
    const shownBeforeRecovery = await page.evaluate(() => window.passesShown);
    await posts[3].abort('failed');
    await page.evaluate(() => window.movePromise);
    assert.equal(await page.evaluate(() => latestState.action_token),'recovered');
    assert.equal(posts.length,4,'connection recovery must never resend the move');
    assert.equal(await page.evaluate(() => window.passesShown),shownBeforeRecovery);

    // Other players' pass notifications also render before the legal-action request returns.
    await setup(initial('receive'),[pass]); holdLegal = true;
    const other = {...initial('receive'),action_token:'other-pass',turn:'B',log:['Started','D: pass']};
    const previousPasses = await page.evaluate(() => window.passesShown);
    await page.evaluate(state => {window.updatePromise = refreshFromState(state);},other);
    await page.waitForFunction(count => window.passesShown === count+1,previousPasses);
    for(let i=0;heldLegal.length===0 && i<100;i++) await new Promise(resolve => setTimeout(resolve,10));
    holdLegal=false; await heldLegal.shift().fulfill({json:[]}); await page.evaluate(() => window.updatePromise);

    // A newer socket notification arriving before the POST response must not be lost.
    await setup(initial('receive'),[pass]);
    await page.evaluate(() => {window.movePromise = onPassClick();});
    await waitForPost(5);
    const ownPass = {...state,action_token:'own-pass',update_version:1,turn:'B',log:['Started','A: pass']};
    state = {...ownPass,action_token:'next-pass',update_version:2,turn:'C',log:['Started','A: pass','B: pass']}; legal=[];
    await page.evaluate(() => queueImmediateActionUpdate({action_token:'next-pass',update_version:2}));
    await posts[4].fulfill({json:{ok:true,state:ownPass}});
    await page.evaluate(() => window.movePromise);
    assert.equal(await page.evaluate(() => latestState.action_token),'next-pass');

    // Receiving unlocks a follow-up attack even when the legal-action response is slow.
    const receive = {action_type:'receive',block:'1',attack:null};
    await setup(initial('receive'),[receive,pass]);
    await page.evaluate(() => {window.receivePromise = onHandPieceClick('1',0);});
    await waitForPost(6);
    assert.equal(await page.locator('#board .cell.slot:not(.attack)').count(),1);
    assert.equal(await page.evaluate(() => immediateActionView(latestState).board_public.A.receive_hidden[0]),false);
    state = await page.evaluate(() => structuredClone(immediateActionView(latestState)));
    Object.assign(state,{action_token:'received',update_version:1,phase:'attack',log:['Started','A: receive 1']});
    holdLegal = true;
    await posts[5].fulfill({json:{ok:true,state}});
    await page.waitForFunction(() => latestState.action_token === 'received' && latestLegal.some(action => action.attack === '2'));
    for(let i=0;heldLegal.length===0 && i<100;i++) await new Promise(resolve => setTimeout(resolve,10));
    await page.evaluate(() => {window.attackPromise = onHandPieceClick('2',3);});
    await waitForPost(7);
    assert.equal(await page.evaluate(() => immediateActionView(latestState).board_public.A.attack[0]),'2');
    assert.equal(await page.evaluate(() => immediateActionView(latestState).hands.A.length),6);
    state = await page.evaluate(() => structuredClone(immediateActionView(latestState)));
    Object.assign(state,{action_token:'attacked',update_version:2,turn:'B',phase:'receive',log:['Started','A: receive 1','A: attack 2']});
    holdLegal = false; legal = [];
    await posts[6].fulfill({json:{ok:true,state}});
    await page.evaluate(() => window.attackPromise);
    await heldLegal.shift().fulfill({json:[{action_type:'attack',block:null,attack:'2'}]});
    await page.evaluate(() => window.receivePromise);
    assert.equal(await page.evaluate(() => latestState.action_token),'attacked','an older receive response must not erase the follow-up attack');

    await setup(initial(),[combined]);
    await page.evaluate(combined => {window.movePromise = submitAction(combined);},combined);
    await waitForPost(8);
    await page.evaluate(() => {cancelImmediateAction(); gid='another-room'; latestState={action_token:'other-room'};});
    await posts[7].fulfill({json:{ok:true,state:other}});
    await page.evaluate(() => window.movePromise);
    assert.equal(await page.evaluate(() => latestState.action_token),'other-room','late POST must not overwrite a different room');
    assert.deepEqual(errors,[]);
    console.log('Immediate actions: delayed POST/GET, duplicate-piece slots, alternate-board snapshot, no double sends/sounds/passes, early feedback, rejection, lost responses and room changes passed.');
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode=1;});
