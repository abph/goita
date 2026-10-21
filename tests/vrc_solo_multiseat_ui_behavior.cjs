const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({channel:'msedge',headless:true});
  try {
    const owners = {}, departed = {}, locked = {}, offline = {}, names = {};
    const errors = [], moves = [];
    let started = false, active = true, turn = 'B';
    const hand = ['1','1','2','3','4','5','6','9'];
    const clone = value => JSON.parse(JSON.stringify(value));
    function meta(client) {
      const ownedSeat = Object.keys(owners).find(s => owners[s] === client);
      const seats = Object.fromEntries(['B','C','D'].map(seat => [seat, {
        joined:!!owners[seat], owned:active && owners[seat] === client && !departed[seat],
        online:active && !!owners[seat] && !departed[seat] && !offline[seat],
        is_ai:started && !owners[seat], reconnect_seconds:offline[seat] ? 60 : 120,
        can_join:active && (!ownedSeat || ownedSeat === seat) && (owners[seat] === client || (!started && !owners[seat] && !locked[seat])),
        can_takeover:active && !!departed[seat],
      }]));
      return {enabled:true,multi:true,active,ended:!active,waiting:active && !started,seats,...seats.C};
    }
    function state(client) {
      const owned = Object.keys(owners).filter(s => owners[s] === client && active && !departed[s]);
      return {action_token:'multi-' + moves.length, update_version:moves.length,
        is_started:started && active, finished:false, turn:started ? turn : null, phase:'receive',
        dealer:'A', round_count:1, owned_human_seats:owned,
        human_seats:['A',...Object.keys(owners)],ai_seats:started ? ['B','C','D'].filter(s => !owners[s]) : [],
        hands:Object.fromEntries(['A','B','C','D'].map(s => [s,started && owned.includes(s) ? hand : {count:started ? 8 : 0}])),
        init_hands:Object.fromEntries(owned.map(s => [s,hand])),face_down_pieces:Object.fromEntries(owned.map(s => [s,[]])),
        board_public:Object.fromEntries(['A','B','C','D'].map(s => [s,{receive:[null,null,null,null],attack:[null,null,null,null],receive_hidden:[false,false,false,false]}])),
        log:[],player_names:{A:'VRChat Player',B:names.B || '参加待ち',C:names.C || '参加待ち',D:names.D || '参加待ち'},
        total_team_score:{AC:40,BD:20},revealed_hand_seats:[],chat_messages:[],vrc_web:meta(client)};
    }
    async function openPage() {
      const page = await browser.newPage({viewport:{width:1100,height:1100}});
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
        const body = route.request().method() === 'POST' && route.request().postData() ? route.request().postDataJSON() : {};
        const client = url.searchParams.get('client_id') || body.client_id || '';
        const seat = url.searchParams.get('seat') || body.player || body.seat;
        if (url.pathname.endsWith('/web/heartbeat')) return route.fulfill({json:meta(client)});
        if (url.pathname.endsWith('/state')) return route.fulfill({json:state(client)});
        if (url.pathname.endsWith('/legal_actions')) return route.fulfill({json:started && active && owners[turn] === client ? [{action_type:'pass',block:null,attack:null}] : []});
        if (url.pathname.endsWith('/claim')) {
          if (owners[seat] && owners[seat] !== client) return route.fulfill({status:409,json:{detail:'使用中'}});
          if (started && owners[seat] !== client) return route.fulfill({status:409,json:{detail:'開始済み'}});
          owners[seat] = client; departed[seat] = false;
          return route.fulfill({json:{ok:true}});
        }
        if (url.pathname.endsWith('/release')) { departed[seat] = true; return route.fulfill({json:{ok:true}}); }
        if (url.pathname.endsWith('/set_name')) { names[seat] = body.name || `サイトの人${seat}`; return route.fulfill({json:{ok:true}}); }
        if (url.pathname.endsWith('/step')) {
          assert.equal(owners[body.player],client);
          assert.equal(body.player,turn);
          moves.push(body.player);
          turn = ({B:'D',D:'C',C:'A'})[turn];
          return route.fulfill({json:{ok:true,state:state(client)}});
        }
        return route.fulfill({json:url.pathname === '/api/member/session' ? {member:null} : {rooms:[],site_people:[],public_chat_messages:[]}});
      });
      await page.goto('http://goita.test/?room=vrc-solo-table1');
      await page.waitForFunction(() => document.getElementById('vrcCoopPanel')?.hidden === false);
      return page;
    }
    const pages = {};
    for (const seat of ['B','D','C']) {
      const page = await openPage();
      pages[seat] = page;
      await page.getByRole('button',{name:`${seat}席で参加・再接続`,exact:true}).click();
      await page.waitForFunction(s => mySeat === s,seat);
      for (const other of ['B','C','D'].filter(s => s !== seat))
        assert.equal(await page.getByRole('button',{name:`${other}席で参加・再接続`,exact:true}).isDisabled(),true);
    }
    started = true;
    for (const [seat,page] of Object.entries(pages)) {
      await page.evaluate(() => refresh());
      const visible = await page.evaluate(() => ({hands:latestState.hands,owned:latestState.owned_human_seats}));
      assert.deepEqual(visible.owned,[seat]);
      for (const other of ['A','B','C','D']) assert.equal(Array.isArray(visible.hands[other]),other === seat);
    }
    for (const seat of ['B','D','C']) {
      await pages[seat].evaluate(() => refresh());
      await pages[seat].evaluate(() => submitAction({action_type:'pass',block:null,attack:null}));
    }
    assert.deepEqual(moves,['B','D','C']);
    await pages.D.reload();
    await pages.D.waitForFunction(() => mySeat === 'D');
    await pages.D.getByRole('button',{name:'D席から退出',exact:true}).click();
    await pages.D.waitForFunction(() => mySeat === 'W');
    assert.equal(await pages.D.locator('#gameView').isVisible(),true);
    await pages.D.getByRole('button',{name:'D席で参加・再接続',exact:true}).click();
    await pages.D.waitForFunction(() => mySeat === 'D');
    const spectator = await openPage();
    for (const seat of ['B','C','D'])
      assert.equal(await spectator.getByRole('button',{name:`${seat}席で参加・再接続`,exact:true}).isDisabled(),true);
    offline.B = true;
    await spectator.evaluate(() => refresh());
    assert.match(await spectator.getByRole('group',{name:'B席の参加',exact:true}).textContent(),/再接続待ち/);
    await pages.C.screenshot({path:path.resolve(__dirname,'../analysis/vrc-multiseat-browser.png'),fullPage:true});
    active = false;
    await pages.B.evaluate(() => refresh());
    await pages.B.waitForFunction(() => mySeat === 'W');
    assert.match(await pages.B.locator('#vrcCoopPanel p').textContent(),/利用が終了/);
    assert.deepEqual(errors,[]);
    console.log('B/C/D browser UI passed: three clients, seat selection, own hands, turns, one seat per client, D reconnect, busy spectator, independent presence and end notice.');
  } finally { await browser.close(); }
})().catch(error => {console.error(error);process.exitCode=1;});
