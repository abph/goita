const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const board = Object.fromEntries(['A','B','C','D'].map(seat => [seat, {
  receive:[null,null,null,null], attack:[null,null,null,null], receive_hidden:[false,false,false,false],
}]));
function initial() {
  return {action_token:'waiting', update_version:0, is_started:false, finished:false, turn:null, phase:'',
    dealer:'A', round_count:1, owned_human_seats:[], human_seats:['A'], ai_seats:['B','D'],
    hands:Object.fromEntries(['A','B','C','D'].map(s => [s,{count:0}])), init_hands:{}, face_down_pieces:{},
    board_public:board, log:[], player_names:{A:'VRChat Player',B:'AI B',C:'サイト参加待ち',D:'AI D'},
    total_team_score:{AC:0,BD:0}, revealed_hand_seats:[], chat_messages:[],
    vrc_web:{enabled:true,active:true,waiting:true,joined:false,owned:false,online:false,can_join:true,can_takeover:true,reconnect_seconds:0}};
}

(async () => {
  const browser = await chromium.launch({channel:'msedge',headless:true});
  try {
    const page = await browser.newPage({viewport:{width:1100,height:1000}});
    const errors = [], calls = [];
    let state = initial(), legal = [], posts = 0;
    page.on('pageerror', error => errors.push(error.message));
    page.on('dialog', dialog => dialog.accept());
    await page.routeWebSocket('**/*', socket => socket.close());
    await page.route('**/*', async route => {
      const url = new URL(route.request().url());
      if (url.hostname !== 'goita.test') return route.abort();
      if (url.pathname === '/' || url.pathname.startsWith('/static/')) {
        const root = path.resolve(__dirname,'../frontend');
        const file = path.resolve(root,url.pathname === '/' ? 'index.html' : url.pathname.slice(8));
        return file.startsWith(root + path.sep) && fs.existsSync(file) ? route.fulfill({path:file}) : route.fulfill({status:404,body:''});
      }
      calls.push(url.pathname);
      if (url.pathname.endsWith('/web/heartbeat')) return route.fulfill({json:state.vrc_web});
      if (url.pathname.endsWith('/state')) return route.fulfill({json:state});
      if (url.pathname.endsWith('/legal_actions')) return route.fulfill({json:legal});
      if (url.pathname.endsWith('/claim')) {
        assert.equal(url.searchParams.get('seat'),'C');
        state.human_seats = ['A','C']; state.owned_human_seats = ['C'];
        Object.assign(state.vrc_web,{owned:true,joined:true,online:true,can_takeover:false});
        return route.fulfill({json:{ok:true}});
      }
      if (url.pathname.endsWith('/release')) {
        state.owned_human_seats = [];
        Object.assign(state.vrc_web,{owned:false,online:false,can_takeover:true,can_join:true});
        return route.fulfill({json:{ok:true}});
      }
      if (url.pathname.endsWith('/set_name')) {
        state.player_names.C = route.request().postDataJSON().name || 'サイトのプレイヤー';
        return route.fulfill({json:{ok:true}});
      }
      if (url.pathname.endsWith('/step')) {
        const body = route.request().postDataJSON();
        assert.equal(body.player,'C'); posts++;
        state.turn = 'A'; state.update_version++; state.action_token = 'after';
        return route.fulfill({json:{ok:true,state}});
      }
      return route.fulfill({json:url.pathname === '/api/member/session' ? {member:null} : {rooms:[],site_people:[],public_chat_messages:[]}});
    });
    await page.goto('http://goita.test/?room=vrc-solo-table1');
    await page.waitForFunction(() => document.getElementById('vrcCoopPanel')?.hidden === false);
    const join = page.getByRole('button',{name:'C席で参加・再接続',exact:true});
    await join.click();
    await page.waitForFunction(() => mySeat === 'C' && document.querySelector('#vrcCoopPanel p').textContent.includes('もう一度START'));
    assert.equal(calls.filter(p => p.endsWith('/claim')).length,1);
    Object.assign(state,{is_started:true,turn:'C',phase:'receive',action_token:'playing',
      hands:{A:{count:8},B:{count:8},C:['1','1','2','3','4','5','6','9'],D:{count:8}},
      init_hands:{C:['1','1','2','3','4','5','6','9']},face_down_pieces:{C:[]}});
    state.vrc_web.waiting = false;
    legal = [{action_type:'pass',block:null,attack:null}];
    await page.evaluate(() => refresh());
    await page.waitForFunction(() => document.querySelector('#vrcCoopPanel p').textContent.includes('C席で対局中'));
    assert.equal(await page.locator('.auto-play').count(),0,'unsupported auto-play control is hidden on linked tables');
    await page.screenshot({path:path.resolve(__dirname,'../analysis/vrc-coop-browser.png'),fullPage:true});
    await page.evaluate(async () => {
      await submitAction({action_type:'pass',block:null,attack:null});
    });
    assert.equal(posts,1);
    await page.reload();
    await page.waitForFunction(() => mySeat === 'C');
    assert.equal(calls.filter(p => p.endsWith('/claim')).length,1,'reload resumes C without taking another seat');
    await page.getByRole('button',{name:'C席から退出',exact:true}).click();
    await page.waitForFunction(() => mySeat === 'W');
    assert.equal(await page.locator('#gameView').isVisible(),true,'leaving C stays on the linked board');
    Object.assign(state.vrc_web,{can_join:false,joined:true,online:true,owned:false});
    await page.evaluate(() => refresh());
    assert.equal(await join.isDisabled(),true);
    assert.match(await page.locator('#vrcCoopPanel p').textContent(),/C席は使用中/);
    Object.assign(state.vrc_web,{active:false,ended:true,can_join:false});
    state.is_started = false;
    await page.evaluate(() => refresh());
    assert.match(await page.locator('#vrcCoopPanel p').textContent(),/利用が終了/);
    await page.evaluate(() => {
      isProcessingCpu = false;
      scheduleAutomaticTurn({is_started:true,finished:false,turn:'B',ai_seats:['B','D']});
      if (isProcessingCpu) throw new Error('Browser must not schedule bridge AI');
      setAutoPlayEnabled(true);
      if (autoPlayEnabled) throw new Error('Browser auto-play must not bypass the cooperative turn controller');
    });
    assert.deepEqual(errors,[]);
    console.log('VRChat/browser cooperative UI passed: join, play, reload, leave, busy seat, end notice, server-controlled AI.');
  } finally { await browser.close(); }
})().catch(error => {console.error(error);process.exitCode=1;});
