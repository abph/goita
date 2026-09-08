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
    await page.evaluate(()=>openDebugTrace());
    assert.equal(await page.locator('#debugTraceModal input[type=file]').count(),0);
    await page.locator('#debugTraceRandomButton').click();
    await page.locator('#debugTraceModal').waitFor({state:'hidden'});
    await page.evaluate(() => {gid='debug';mySeat='A';refresh=async()=>{};syncTraceResult({trace_mode:true,finished:true,trace_attempt_id:'first'});});
    await page.locator('#traceResultContent').waitFor({state:'visible'});
    assert.equal(await page.locator('#traceOriginalBD').textContent(),'30点');
    assert.equal(await page.locator('#traceActualAC').textContent(),'40点');
    assert.match(await page.locator('#traceImprovement').textContent(),/\+70点/);
    assert.equal(await page.locator('#traceRankingRows tr').count(),2);
    assert.equal(await page.locator('#traceRankingRows img').count(),0);
    assert.match(await page.locator('#traceRetention').textContent(),/ゲストの記録期限/);
    assert.match(await page.locator('#traceRankingRows tr').first().locator('td').last().textContent(),/^\d{4}\/\d{2}\/\d{2}$/);
    assert.equal(await page.locator('#traceRankingbest').getAttribute('aria-pressed'),'true');
    await page.locator('#traceRankingfirst').click();
    await page.waitForFunction(()=>document.getElementById('traceRankingfirst').getAttribute('aria-pressed')==='true');
    await page.getByRole('button',{name:'挑戦履歴へ',exact:true}).click();
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
    await page.locator('#traceOriginalPlay').click();
    assert.equal(await page.locator('#traceOriginalPlay').textContent(),'停止');
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('#traceOriginalModal').isVisible(),false);
    await page.locator('#traceRetryButton').click();
    await page.locator('#traceResultModal').waitFor({state:'hidden'});
    assert.ok(calls.some(call => call.path.endsWith('/first/retry')));
    await page.evaluate(() => syncTraceResult({trace_mode:true,finished:true,trace_attempt_id:'first'}));
    assert.equal(await page.locator('#traceResultModal').isVisible(),false);
    result.attempt_id='practice';result.ranked=false;result.attempt_no=2;result.is_best=false;
    await page.evaluate(() => syncTraceResult({trace_mode:true,finished:true,trace_attempt_id:'practice'}));
    await page.locator('#traceResultContent').waitFor({state:'visible'});
    assert.equal(await page.locator('#traceRecordKind').textContent(),'2回目の挑戦');
    await page.setViewportSize({width:390,height:700});
    const size=await page.locator('#traceResultModal .modal-content').boundingBox();
    assert.ok(size.width<=390 && size.height<=700);
    await page.screenshot({path:path.join(__dirname,'../.codex_deps/trace-results-mobile.png')});
    await page.locator('#traceNextButton').click();
    assert.ok(calls.some(call => call.path.endsWith('/trace_random_start')));
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
    console.log('TRACE_RESULTS_UI_OK: scores, ranking, private replay, retry, next, mobile, debug-only');
  } finally {await browser.close();}
})().catch(error => {console.error(error);process.exit(1);});
