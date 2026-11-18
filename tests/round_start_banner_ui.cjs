const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

function fixture(round=1, started=true) {
  return {is_started:started, finished:false, turn:null, phase:'receive', dealer:'C',
    attacker:'C', current_attack:'1', round_count:round,
    board_public:Object.fromEntries([...'ABCD'].map(seat=>[seat, {
      receive:['1',null,null,null], attack:['1',null,null,null], receive_hidden:[true,false,false,false],
    }])), hands:Object.fromEntries([...'ABCD'].map(seat=>[seat,{count:6}])),
    init_hands:{}, face_down_pieces:{}, player_names:{A:'A',B:'B',C:'プレイヤー名',D:'D'},
    human_seats:['A','B','C','D'], ai_seats:[], revealed_hand_seats:[],
    total_team_score:{AC:20,BD:30}, log:[], chat_messages:[]};
}

(async()=>{
  const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-swiftshader']});
  const output=path.resolve('.server_deps_divergent/round-start-ui');
  fs.mkdirSync(output,{recursive:true});
  try {
    const page=await browser.newPage({viewport:{width:1250,height:1000}});
    const errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.routeWebSocket('**/*',()=>{});
    let state=fixture();
    await page.route('**/*',route=>{
      const url=new URL(route.request().url());
      if(url.hostname!=='goita.test') return route.abort();
      if(url.pathname==='/' || url.pathname.startsWith('/static/')) {
        const root=path.resolve(__dirname,'../frontend');
        const file=path.resolve(root,url.pathname==='/'?'index.html':url.pathname.slice(8));
        return file.startsWith(root+path.sep) && fs.existsSync(file)
          ?route.fulfill({path:file}):route.fulfill({status:404,body:''});
      }
      if(url.pathname.endsWith('/state')) return route.fulfill({json:state});
      if(url.pathname.endsWith('/legal_actions')) return route.fulfill({json:[]});
      return route.fulfill({json:url.pathname==='/api/member/session'?{member:null}:{rooms:[],site_people:[],public_chat_messages:[]}});
    });
    await page.goto('http://goita.test/');
    await page.waitForFunction(()=>typeof personalSettings!=='undefined' && window.goitaBoardPixel);
    await page.evaluate(()=>{
      gid=activeRoomId=currentWsGid='room-gold-01'; mySeat=WATCH;
      document.getElementById('lobbyView').style.display='none';
      document.getElementById('gameView').style.display='block';
      personalSettings.enableEffects=false;
      audioUnlocked=true;
      window.startSoundCount=0;
      nextRoundSound.play=()=>{window.startSoundCount++;return Promise.resolve();};
      hideRoundStartBanner(true);
    });
    async function refresh(snapshot) {
      state=snapshot;
      await page.evaluate(snapshot=>refreshFromState(snapshot),snapshot);
    }
    const banner=page.locator('#roundStartBanner');
    await refresh(fixture());
    assert.equal(await banner.isVisible(),false,'joining an ongoing round is silent');
    await refresh(fixture(2,false));
    assert.equal(await banner.isVisible(),false,'dealing is not the start');
    await refresh(fixture(2));
    assert.equal(await banner.isVisible(),true);
    assert.equal(await page.locator('#roundStartTitle').textContent(),'第2局　開始');
    assert.equal(await page.locator('#roundStartDealer').textContent(),'親：C（プレイヤー名）');
    assert.equal(await page.evaluate(()=>window.startSoundCount),1);
    const timer=await page.evaluate(()=>roundStartBannerTimer);
    await refresh(fixture(2));
    assert.equal(await page.evaluate(()=>roundStartBannerTimer),timer,'repeated updates do not restart the fade');
    assert.equal(await page.evaluate(()=>window.startSoundCount),1);
    await page.waitForFunction(()=>document.getElementById('roundStartBanner').hidden);
    await refresh(fixture(2));
    assert.equal(await banner.isVisible(),false,'reconnection/current-state refresh does not repeat the banner');
    // Notifications appear for seated clients as well as spectators and never intercept input.
    for(const width of [1250,390]) {
      await page.setViewportSize({width,height:1000});
      for(const mode of ['2d','pixel-color','3d']) {
        await page.evaluate(({mode,snapshot})=>{
          mySeat='A'; personalSettings.boardViewMode=mode; applyBoardViewMode();
          hideRoundStartBanner(true); syncRoundStartBanner({...snapshot,round_count:1});
          latestState=snapshot; renderBoard(snapshot); syncRoundStartBanner(snapshot);
        },{mode,snapshot:fixture(2)});
        await page.waitForTimeout(250);
        const geometry=await page.evaluate(()=>{
          const wrap=document.querySelector('#gameView .board-wrap');
          const banner=document.getElementById('roundStartBanner');
          const w=wrap.getBoundingClientRect(), b=banner.getBoundingClientRect();
          const border=parseFloat(getComputedStyle(wrap).borderLeftWidth);
          return {left:b.left-(w.left+border),right:(w.right-border)-b.right,
            center:(b.top+b.bottom-w.top-w.bottom)/2,pointer:getComputedStyle(banner).pointerEvents};
        });
        assert.ok(Math.abs(geometry.left)<1 && Math.abs(geometry.right)<1,'band spans board frame');
        assert.ok(Math.abs(geometry.center)<1,'band is vertically centered');
        assert.equal(geometry.pointer,'none');
        await page.locator('#gameView .board-wrap').screenshot({path:path.join(output,`${mode}-${width}.png`)});
      }
    }
    await page.evaluate(snapshot=>{
      hideRoundStartBanner(true); personalSettings.enableSoundEffects=false;
      syncRoundStartBanner({...snapshot,is_started:false});
      syncRoundStartBanner(snapshot);
    },fixture(3));
    assert.equal(await banner.isVisible(),true,'essential notice still works with effects/sounds disabled');
    const soundsBefore=await page.evaluate(()=>window.startSoundCount);
    for(const language of ['en','zh']) {
      await page.evaluate(({snapshot,language})=>{
        goitaI18n.setLanguage(language,{dispatch:false});
        hideRoundStartBanner(true); syncRoundStartBanner({...snapshot,is_started:false});
        syncRoundStartBanner(snapshot);
      },{snapshot:fixture(4),language});
      assert.equal(await page.locator('#roundStartTitle').textContent(),language==='en'?'Round 4 — Start':'第4局　开始');
      assert.match(await page.locator('#roundStartDealer').textContent(),/C（プレイヤー名）/);
    }
    assert.equal(await page.evaluate(()=>window.startSoundCount),soundsBefore,'sound preference is respected');
    for(const special of ['practice_replay_active','tutorial_mode']) {
      await page.evaluate(({snapshot,special})=>{
        hideRoundStartBanner(true); syncRoundStartBanner({...snapshot,[special]:true,is_started:false});
        syncRoundStartBanner({...snapshot,[special]:true});
      },{snapshot:fixture(5),special});
      assert.equal(await banner.isVisible(),false);
    }
    // Match reset, room changes, and a new first-round start.
    await page.evaluate(snapshot=>{
      hideRoundStartBanner(true); syncRoundStartBanner({...snapshot,finished:true});
      syncRoundStartBanner({...snapshot,is_started:false}); syncRoundStartBanner(snapshot);
    },fixture());
    assert.equal(await banner.isVisible(),true);
    await page.evaluate(snapshot=>{gid='room-gold-02';syncRoundStartBanner(snapshot);},fixture(2));
    assert.equal(await banner.isVisible(),false);
    await page.evaluate(snapshot=>{
      gid='room-gold-01'; hideRoundStartBanner(true);
      syncRoundStartBanner({...snapshot,is_started:false}); syncRoundStartBanner(snapshot);
      hideRoundStartBanner(true);
    },fixture(2));
    assert.equal(await banner.isVisible(),false);
    assert.deepEqual(errors,[]);
    console.log('Round start banner: authoritative transitions, reconnect/duplicate suppression, sound preferences, localization, PC/mobile and 2D/pixel/3D passed');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
