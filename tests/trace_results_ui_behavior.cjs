const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({channel:'msedge', headless:true});
  try {
    const page = await browser.newPage({viewport:{width:1100,height:950}});
    const errors = [], calls = [];
    page.on('pageerror', error => errors.push(error.message));
    const result = {attempt_id:'first', original:{AC:0,BD:30},actual:{AC:40,BD:0},improvement:70,
      ranked:true,guest:true,expires_at:2000000000,own_rank:1,total:2,attempt_no:1,is_best:true,challenge_label:'課題001',
      ranking:[{rank:1,name:'<img src=x onerror=alert(1)>',guest:true,improvement:70,finished_at:1900000000,self:true,attempt_no:3},
        {rank:2,name:'会員さん',guest:false,improvement:20,finished_at:1900000010,self:false,attempt_no:1}]};
    const payload = {round_index:1, winner:'B', gained_score:20,
      hand:{p0:'しし王金金銀飛香',p1:'しし玉銀銀銀香馬',p2:'しししし金飛香香',p3:'しし角角金馬馬馬'},
      player_names:{},anonymous:true,my_seat:'A',game:[['1','馬','銀'],['0','銀','金']],
      score_before:{AC:0,BD:0},score_after:{AC:0,BD:20}};
    await page.routeWebSocket('**/*', socket => socket.close());
    await page.route('**/*', route => {
      const url = new URL(route.request().url());
      if (url.hostname !== 'goita.test') return route.abort();
      if (url.pathname === '/' || url.pathname.startsWith('/static/')) {
        const root = path.resolve(__dirname,'../frontend');
        const file = path.resolve(root, url.pathname === '/' ? 'index.html' : url.pathname.slice(8));
        return file.startsWith(root + path.sep) && fs.existsSync(file) ? route.fulfill({path:file}) : route.fulfill({status:404,body:''});
      }
      if (url.pathname.includes('/trace_')) {
        calls.push({path:url.pathname, body:route.request().postData(), headers:route.request().headers()});
        if (url.pathname.endsWith('/history')) {
          const offset=Number(url.searchParams.get('offset'));
          const records=Array.from({length:31},(_,i)=>({attempt_id:`old-${i}`,challenge_label:'課題001',improvement:i,
            attempt_no:31-i,finished_at:1900000000-i*86400,is_best:i===0}));
          return route.fulfill({json:{total:31,offset,limit:30,records:records.slice(offset,offset+30)}});
        }
        if (url.pathname.endsWith('/latest')) return route.fulfill({json:{attempt_id:'first'}});
        if (url.pathname.endsWith('/original')) return route.fulfill({json:{payload}});
        if (url.pathname.endsWith('/retry') || url.pathname.endsWith('/trace_random_start')) return route.fulfill({json:{ok:true}});
        return route.fulfill({json:result});
      }
      return route.fulfill({json:url.pathname === '/api/member/session' ? {member:null} : {rooms:[],site_people:[],public_chat_messages:[]}});
    });
    await page.goto('http://goita.test/');
    const resetBehavior = await page.evaluate(async () => {
      gid='debug'; mySeat='A'; refresh=async()=>{};
      const starts=[]; startNewGame=async keepScore=>starts.push(keepScore);
      const button=document.getElementById('btnNewGame');
      updateRoundResetButton({finished:true,trace_mode:true,match_finished:false},true,false);
      const scoreLabel=button.textContent;
      await button.onclick();
      updateRoundResetButton({finished:true,trace_mode:false,match_finished:false},true,false);
      const ordinaryLabel=button.textContent;
      await button.onclick();
      return {scoreLabel,ordinaryLabel,starts};
    });
    assert.deepEqual(resetBehavior,{scoreLabel:'リセット',ordinaryLabel:'次の一局へ',starts:[false,true]});
    const practiceBehavior = await page.evaluate(async () => {
      gid='room-gold-01'; mySeat='A'; practiceReplayRequestInFlight=false;
      const actions=[]; practiceReplayAction=async (action,sceneIndex)=>actions.push([action,sceneIndex]);
      const state={finished:true,match_finished:false,practice_replay_active:false,practice_replay_available:true};
      updateRoundResetButton(state,true,false);
      updatePracticeReplayButtons(state,true,false);
      const normal={
        nextVisible:document.getElementById('btnNewGame').style.display !== 'none',
        replayVisible:document.getElementById('btnPracticeReplay').style.display !== 'none',
        returnVisible:document.getElementById('btnPracticeReturn').style.display !== 'none',
        sceneVisible:document.getElementById('btnPracticeScene').style.display !== 'none',
        replayLabel:document.getElementById('btnPracticeReplay').textContent,
        sceneLabel:document.getElementById('btnPracticeScene').textContent,
      };
      await document.getElementById('btnPracticeReplay').onclick();
      state.practice_replay_active=true;
      updateRoundResetButton(state,true,false);
      updatePracticeReplayButtons(state,true,false);
      const practice={
        nextVisible:document.getElementById('btnNewGame').style.display !== 'none',
        replayVisible:document.getElementById('btnPracticeReplay').style.display !== 'none',
        returnVisible:document.getElementById('btnPracticeReturn').style.display !== 'none',
        sceneVisible:document.getElementById('btnPracticeScene').style.display !== 'none',
        returnLabel:document.getElementById('btnPracticeReturn').textContent,
      };
      await document.getElementById('btnPracticeReturn').onclick();
      return {normal,practice,actions};
    });
    assert.deepEqual(practiceBehavior,{
      normal:{nextVisible:true,replayVisible:true,returnVisible:false,sceneVisible:true,replayLabel:'もう一度練習する',sceneLabel:'場面を指定して練習する'},
      practice:{nextVisible:false,replayVisible:false,returnVisible:true,sceneVisible:false,returnLabel:'元のゲームに戻る'},
      actions:[['start',undefined],['return',undefined]],
    });
    const publicPracticeBehavior = await page.evaluate(() => {
      gid='main';
      const state={finished:true,practice_replay_active:false,practice_replay_available:true};
      updatePracticeReplayButtons(state,true,false);
      return {
        replayVisible:document.getElementById('btnPracticeReplay').style.display !== 'none',
        sceneVisible:document.getElementById('btnPracticeScene').style.display !== 'none',
      };
    });
    assert.deepEqual(publicPracticeBehavior,{replayVisible:false,sceneVisible:false});
    await page.evaluate(() => {
      gid='room-gold-01'; mySeat='A';
      latestState={
        finished:true,practice_replay_active:false,practice_replay_available:true,
        log:['Game start. dealer=A','A: block 2 -> attack 3','B: pass','C: receive 3','C: attack 4','Round finished. winner=C gained=10'],
        log_turn_numbers:[null,1,2,3,3,null],
      };
      window.practiceSceneCalls=[];
      practiceReplayAction=async (action,sceneIndex)=>window.practiceSceneCalls.push([action,sceneIndex]);
      openPracticeScenePicker();
    });
    assert.equal(await page.locator('#practiceSceneModal').isVisible(),true);
    assert.equal(await page.locator('#practiceSceneList .practice-scene-turn').count(),3);
    assert.match(await page.locator('#practiceSceneList .practice-scene-turn').nth(2).textContent(),/C手番3/);
    await page.locator('#practiceSceneList .practice-scene-turn').nth(2).getByRole('button').click();
    assert.deepEqual(await page.evaluate(()=>window.practiceSceneCalls),[['scene',2]]);
    await page.locator('#practiceSceneModal .settings-modal-close').click();
    await page.evaluate(()=>{gid='debug';mySeat='A';});
    await page.evaluate(()=>openDebugTrace());
    assert.equal(await page.locator('#debugTraceModal input[type=file]').count(),0);
    await page.locator('#debugTraceRandomButton').click();
    await page.locator('#debugTraceModal').waitFor({state:'hidden'});
    await page.evaluate(() => {gid='debug';mySeat='A';refresh=async()=>{};syncTraceResult({trace_mode:true,finished:true,trace_attempt_id:'first'});});
    await page.locator('#traceResultContent').waitFor({state:'visible'});
    assert.equal(await page.locator('#traceOriginalScore').textContent(),'BD：30点');
    assert.equal(await page.locator('#traceActualScore').textContent(),'AC：40点');
    assert.match(await page.locator('#traceImprovement').textContent(),/\+70点/);
    assert.equal(await page.locator('#traceRankingRows tr').count(),2);
    assert.equal(await page.locator('#traceRankingRows img').count(),0);
    for (const selector of ['#traceRetention','#traceChallengeLabel','#traceRankingSummary','#traceNextButton']) {
      assert.equal(await page.locator(selector).count(),0);
    }
    for (const name of ['ランキングを更新','挑戦履歴へ','別の棋譜に挑戦']) {
      assert.equal(await page.locator('#traceResultModal').getByRole('button',{name,exact:true}).count(),0);
    }
    assert.match(await page.locator('#traceRankingRows tr').first().locator('td').last().textContent(),/^\d{4}\/\d{2}\/\d{2}$/);
    assert.equal(await page.locator('#traceRankingbest,#traceRankingfirst,#traceRetryButton').count(),0);
    assert.match(await page.locator('#traceRankingRows tr').first().textContent(),/3回目/);
    await page.locator('#traceResultModal .settings-modal-close').click();
    await page.evaluate(()=>openDebugTrace());
    await page.getByRole('button',{name:'挑戦履歴・ランキング',exact:true}).click();
    await page.locator('#traceHistoryRows tr').first().waitFor();
    assert.equal(await page.locator('#traceHistoryRows tr').count(),30);
    await page.locator('#traceHistoryNext').click();
    await page.waitForFunction(()=>document.querySelectorAll('#traceHistoryRows tr').length===1);
    await page.locator('#traceHistoryRows button').click();
    await page.locator('#traceResultContent').waitFor({state:'visible'});
    assert.ok(calls.some(call=>call.path.endsWith('/old-30')));
    assert.equal(await page.locator('#traceHistoryModal').isVisible(),false);
    await page.evaluate(()=>openTraceResult('first'));
    await page.locator('#traceViewOriginalButton').click();
    await page.locator('#traceOriginalModal').waitFor({state:'visible'});
    assert.ok(await page.locator('#traceOriginalBoard .research-kifu-piece').count());
    assert.equal(await page.locator('#traceOriginalBoard .is-self').count(),1);
    assert.equal(await page.locator('#traceOriginalPosition').count(),0);
    assert.equal(await page.locator('#traceOriginalModal').getByRole('button',{name:'結果に戻る',exact:true}).count(),0);
    await page.locator('#traceOriginalPlay').click();
    assert.equal(await page.locator('#traceOriginalPlay').textContent(),'停止');
    await page.locator('#traceOriginalBack').click();
    assert.equal(await page.locator('#traceOriginalModal').isVisible(),false);
    assert.equal(await page.locator('#traceResultModal').isVisible(),true);
    assert.equal(await page.locator('#traceViewOriginalButton').evaluate(element=>element===document.activeElement),true);
    await page.evaluate(()=>closeTraceResult());
    await page.evaluate(() => syncTraceResult({trace_mode:true,finished:true,trace_attempt_id:'first'}));
    assert.equal(await page.locator('#traceResultModal').isVisible(),false);
    result.attempt_id='practice';result.ranked=false;result.attempt_no=2;result.is_best=false;
    result.actual={AC:30,BD:0};result.original={AC:30,BD:0};result.improvement=0;
    await page.evaluate(() => syncTraceResult({trace_mode:true,finished:true,trace_attempt_id:'practice'}));
    await page.locator('#traceResultContent').waitFor({state:'visible'});
    assert.equal(await page.locator('#traceRecordKind').count(),0);
    assert.equal(await page.getByText('AC：自分のペア ／ BD：相手のペア',{exact:true}).count(),0);
    assert.equal(await page.locator('#traceHistoryModal th').count(),4);
    assert.equal(await page.locator('#traceActualScore').textContent(),'AC：30点');
    assert.equal(await page.locator('#traceOriginalScore').textContent(),'AC：30点');
    assert.equal(await page.locator('#traceImprovement').textContent(),'元の対局との差　±0点');
    await page.screenshot({path:path.join(__dirname,'../.codex_deps/score-result-simple-desktop.png')});
    await page.setViewportSize({width:390,height:700});
    const size=await page.locator('#traceResultModal .modal-content').boundingBox();
    assert.ok(size.width<=390 && size.height<=700);
    await page.screenshot({path:path.join(__dirname,'../.codex_deps/trace-results-mobile.png')});
    await page.locator('#traceViewOriginalButton').click();
    await page.locator('#traceOriginalModal').waitFor({state:'visible'});
    await page.screenshot({path:path.join(__dirname,'../.codex_deps/score-original-simple-mobile.png')});
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('#traceOriginalModal').isVisible(),false);
    await page.evaluate(()=>openTraceHistory());
    await page.locator('#traceHistoryRows tr').first().waitFor();
    const historySize=await page.locator('#traceHistoryModal .modal-content').boundingBox();
    assert.ok(historySize.width<=390 && historySize.height<=700);
    await page.screenshot({path:path.join(__dirname,'../.codex_deps/score-history-mobile.png')});
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('#traceHistoryModal').isVisible(),false);
    assert.ok(calls.every(call => call.headers['x-goita-member']==='1'));
    await page.evaluate(() => {gid='main'; syncTraceResult({trace_mode:true,finished:true,trace_attempt_id:'other'});});
    assert.equal(await page.locator('#traceResultModal').isVisible(),false);
    assert.deepEqual(errors,[]);
    console.log('TRACE_RESULTS_UI_OK: compact scores, ranking, top back navigation, private replay, retry, history, mobile, debug-only');
  } finally {await browser.close();}
})().catch(error => {console.error(error);process.exit(1);});
