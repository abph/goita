const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({channel:'msedge',headless:true});
  try {
    const page = await browser.newPage({viewport:{width:1100,height:900}});
    const errors = [], calls = [];
    let fail = false, empty = false, holdDaily = false, heldRoute, notifyHeld;
    const held = new Promise(resolve => {notifyHeld = resolve;});
    const data = period => ({period,start_date:'2026-09-07',end_date:'2026-09-13',total:empty ? 0 : 1,
      ranking:empty ? [] : [{rank:1,name:'<img src=x onerror=alert(1)>',guest:true,self:true,score:period === 'weekly' ? 80 : -70,games:3}]});
    page.on('pageerror', error => errors.push(error.message));
    await page.routeWebSocket('**/*', socket => socket.close());
    await page.route('**/*', route => {
      const url = new URL(route.request().url());
      if (url.hostname !== 'goita.test') return route.abort();
      if (url.pathname === '/' || url.pathname.startsWith('/static/')) {
        const root = path.resolve(__dirname,'../frontend');
        const file = path.resolve(root,url.pathname === '/' ? 'index.html' : url.pathname.slice(8));
        return file.startsWith(root + path.sep) && fs.existsSync(file) ? route.fulfill({path:file}) : route.fulfill({status:404,body:''});
      }
      calls.push({path:url.pathname,headers:route.request().headers()});
      if (url.pathname === '/api/score-attack/rankings') {
        const period = url.searchParams.get('period');
        if (holdDaily && period === 'daily') {heldRoute = route; notifyHeld(); return;}
        return fail ? route.fulfill({status:503,json:{detail:'一時的に読み込めません。'}}) : route.fulfill({json:data(period)});
      }
      return route.fulfill({json:url.pathname === '/api/member/session' ? {member:null} : {rooms:[],site_people:[],public_chat_messages:[]}});
    });
    await page.goto('http://goita.test/');
    const section = page.locator('section.score-attack-entry');
    assert.equal(await section.locator('h2').textContent(),'スコアアタック');
    const play = await section.locator('#scoreAttackEntry').boundingBox();
    const rank = await section.locator('#scoreRankingsEntry').boundingBox();
    assert.ok(rank.x > play.x && rank.y === play.y);
    await page.locator('#scoreRankingsEntry').click();
    await page.locator('#scoreRankingsRows tr').waitFor();
    assert.match(await page.locator('#scoreRankingsRows').textContent(),/-70点/);
    assert.equal(await page.locator('#scoreRankingsRows img').count(),0);
    await page.locator('#scorePeriodweekly').click();
    await page.waitForFunction(()=>document.getElementById('scoreRankingsRows').textContent.includes('80点'));
    assert.match(await page.locator('#scoreRankingsNote').textContent(),/マイナスの日は0点/);
    assert.equal(page.url(),'http://goita.test/');
    assert.ok(!calls.some(call=>call.path === '/api/score-attack/enter' || call.path.includes('/games/score-')));
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('#scoreRankingsEntry').evaluate(el=>el===document.activeElement),true);
    holdDaily = true;
    await page.evaluate(()=>{void openScoreRankings('daily');});
    await held;
    await page.evaluate(()=>openScoreRankings('weekly'));
    await heldRoute.fulfill({json:data('daily')});
    await page.waitForTimeout(100);
    assert.match(await page.locator('#scoreRankingsRows').textContent(),/80点/);
    holdDaily = false;
    await page.setViewportSize({width:390,height:700});
    const box = await page.locator('#scoreRankingsModal .modal-content').boundingBox();
    assert.ok(box.width <= 390 && box.height <= 700);
    await page.screenshot({path:path.join(__dirname,'../.codex_deps/score-weekly-mobile.png')});
    empty = true;
    await page.evaluate(()=>openScoreRankings());
    assert.match(await page.locator('#scoreRankingsStatus').textContent(),/まだありません/);
    fail = true;
    await page.evaluate(()=>openScoreRankings());
    assert.match(await page.locator('#scoreRankingsStatus').textContent(),/一時的に/);
    assert.ok(calls.filter(call=>call.path.endsWith('/rankings')).every(call=>call.headers['x-goita-member']==='1'));
    assert.deepEqual(errors,[]);
    console.log('SCORE_PERIOD_UI_OK: lobby-only ranking, daily/weekly, stale response, empty/error, mobile, escaped names');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
