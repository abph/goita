const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({channel:'msedge', headless:true});
  try {
    const page = await browser.newPage({viewport:{width:1100,height:900}});
    const errors = [], calls = [], dialogs = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('dialog', async dialog => {dialogs.push(dialog.message()); await dialog.accept();});
    let expired = false;
    const state = {action_token:'before',update_version:0,is_started:false,finished:false,turn:null,phase:'',
      dealer:'A',round_count:1,owned_human_seats:['A'],human_seats:['A'],ai_seats:['B','C','D'],owner_name:'スコアアタック',
      hands:{A:['1','1','1','2','3','4','5','9'],B:{count:8},C:{count:8},D:{count:8}},
      init_hands:{A:['1','1','1','2','3','4','5','9']},face_down_pieces:{A:[]},
      board_public:Object.fromEntries(['A','B','C','D'].map(seat=>[seat,{receive:[null,null,null,null],attack:[null,null,null,null],receive_hidden:[false,false,false,false]}])),
      log:[],player_names:{A:'自分',B:'',C:'',D:''},total_team_score:{AC:0,BD:0},revealed_hand_seats:[],chat_messages:[]};
    const result = {attempt_id:'one',original:{AC:0,BD:30},actual:{AC:0,BD:30},improvement:0,
      attempt_no:1,is_best:true,ranking:[]};
    await page.routeWebSocket('**/*', socket => socket.close());
    await page.route('**/*', route => {
      const request = route.request(), url = new URL(request.url());
      if (url.hostname !== 'goita.test') return route.abort();
      if (url.pathname === '/' || url.pathname.startsWith('/static/')) {
        const root = path.resolve(__dirname,'../frontend');
        const file = path.resolve(root,url.pathname === '/' ? 'index.html' : url.pathname.slice(8));
        return file.startsWith(root + path.sep) && fs.existsSync(file) ? route.fulfill({path:file}) : route.fulfill({status:404,body:''});
      }
      calls.push({path:url.pathname,method:request.method(),headers:request.headers()});
      if (url.pathname === '/api/score-attack/enter') return route.fulfill({json:{game_id:'score-personal'}});
      if (url.pathname.endsWith('/trace_random_start')) {
        Object.assign(state,{is_started:true,turn:'A',phase:'attack',trace_mode:true,trace_attempt_id:'one',log:['Started']});
        return route.fulfill({json:{ok:true}});
      }
      if (url.pathname.endsWith('/score_reset')) {
        Object.assign(state,{is_started:false,finished:false,turn:null,phase:'',trace_mode:false,trace_attempt_id:'',log:[]});
        return route.fulfill({json:{ok:true}});
      }
      if (url.pathname.endsWith('/state')) return expired ? route.fulfill({status:410,json:{detail:'expired'}}) : route.fulfill({json:state});
      if (url.pathname.endsWith('/legal_actions')) return route.fulfill({json:state.is_started && !state.finished ? [{action_type:'attack_after_block',block:'1',attack:'1'}] : []});
      if (url.pathname.endsWith('/history')) return route.fulfill({json:{offset:0,limit:30,total:1,records:[{attempt_id:'one',challenge_label:'課題001',improvement:0,attempt_no:1,finished_at:1900000000,is_best:true}]}});
      if (url.pathname.endsWith('/trace_results/one')) return route.fulfill({json:result});
      return route.fulfill({json:url.pathname === '/api/member/session' ? {member:null} : {rooms:[],site_people:[],public_chat_messages:[]}});
    });
    await page.goto('http://goita.test/');
    assert.equal(await page.getByRole('button',{name:'ランキングを見る',exact:true}).count(),0);
    await page.locator('#scoreAttackEntry').click();
    await page.locator('#debugTraceModal').waitFor({state:'visible'});
    assert.equal(page.url(),'http://goita.test/');
    assert.equal(await page.evaluate(()=>mySeat),'A');
    assert.equal(await page.locator('#debugTraceModal .settings-help-text').count(),0);
    await page.locator('#debugTraceRandomButton').click();
    await page.locator('#debugTraceModal').waitFor({state:'hidden'});
    for (const id of ['btnSeatA','btnSeatB','btnSeatC','btnSeatD']) assert.equal(await page.locator(`#${id}`).isDisabled(),true);
    assert.equal(await page.locator('#handRevealPanel').isVisible(),false);
    assert.equal(await page.locator('.auto-play').first().isVisible(),false);
    await page.evaluate(()=>openSeatMenu('B'));
    assert.equal(await page.locator('#seatMenuModal').isVisible(),false);
    await page.evaluate(()=>openSettingsModal(gid));
    for (const id of ['matchSettingsTab','roomManagementTab','forceResetSettingsRow']) assert.equal(await page.locator(`#${id}`).isVisible(),false);
    await page.evaluate(()=>closeSettings());
    await page.reload();
    await page.waitForFunction(()=>gid === 'score-personal' && mySeat === 'A' && latestState?.is_started);
    assert.equal(page.url(),'http://goita.test/');
    assert.equal(await page.locator('#debugTraceModal').isVisible(),false);
    await page.evaluate(()=>returnToLobby());
    await page.locator('#scoreAttackEntry').click();
    await page.waitForFunction(()=>gid === 'score-personal' && mySeat === 'A');
    assert.ok(!calls.some(call=>/\/(release|claim|auto_step|reveal_hand|reset_config)$/.test(call.path)));
    Object.assign(state,{finished:true,turn:'B',winner:'B',last_round_score:30,total_team_score:{AC:0,BD:30},trace_original_score_after:{AC:0,BD:30}});
    await page.evaluate(()=>refresh());
    await page.locator('#traceResultContent').waitFor({state:'visible'});
    await page.locator('#traceResultTitle button').click();
    await page.locator('#traceHistoryRows button').waitFor({state:'visible'});
    assert.equal(await page.locator('.trace-history-scroll table').evaluate(el=>getComputedStyle(el).fontSize),'13px');
    await page.locator('#traceHistoryTitle button').click();
    await page.locator('#debugTraceModal').waitFor({state:'visible'});
    await page.evaluate(()=>closeDebugTrace());
    await page.locator('#btnNewGame').click();
    await page.locator('#debugTraceModal').waitFor({state:'visible'});
    assert.ok(calls.some(call=>call.path.endsWith('/score_reset')));
    await page.setViewportSize({width:390,height:844});
    await page.locator('#debugTraceModal button.settings-modal-close').click();
    await page.evaluate(()=>openTraceHistory());
    await page.locator('#traceHistoryRows button').waitFor();
    assert.ok(await page.locator('#traceHistoryTitle button').isVisible());
    await page.locator('#traceHistoryTitle button').click();
    expired = true;
    await page.evaluate(()=>refresh());
    await page.waitForFunction(()=>document.getElementById('lobbyView').style.display === 'block' && !sessionStorage.getItem('goitaScoreRoomOpen'));
    assert.ok(dialogs.some(message=>message.includes('ルームは終了')));
    assert.ok(calls.filter(call=>call.path.startsWith('/games/score-')).every(call=>call.headers['x-goita-member']==='1'));
    assert.deepEqual(errors,[]);
    console.log('SCORE_ROOMS_UI_OK: lobby entry, unchanged URL, A seat, fixed controls, reload, return, history navigation, reset, mobile, expiry');
  } finally {
    await browser.close();
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
