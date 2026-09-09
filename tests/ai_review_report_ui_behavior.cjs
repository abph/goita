const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const fixture = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../results/ai_review_report/sample.json'), 'utf8'));

(async () => {
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1100, height: 900}});
    const errors = [];
    let requests = 0, failSnapshot = false;
    page.on('pageerror', error => errors.push(error.message));
    await page.routeWebSocket('**/*', socket => socket.close());
    await page.route('**/*', async route => {
      const url = new URL(route.request().url());
      if(url.hostname !== 'goita.test') return route.abort();
      if(url.pathname === '/' || url.pathname.startsWith('/static/')) {
        const root = path.resolve(__dirname, '../frontend');
        const file = path.resolve(root, url.pathname === '/' ? 'index.html' : url.pathname.slice(8));
        return file.startsWith(root + path.sep) && fs.existsSync(file) ? route.fulfill({path: file}) : route.fulfill({status: 404, body: ''});
      }
      if(url.pathname.endsWith('/ai_review_snapshot')) {
        requests++;
        assert.equal(url.searchParams.get('round_id'), 'review-round');
        return route.fulfill(failSnapshot ? {status: 409, json: {detail: '局が切り替わりました。最新のログから選び直してください。'}} : {json: fixture});
      }
      return route.fulfill({json: url.pathname === '/api/member/session' ? {member: null} : {rooms: [], site_people: [], public_chat_messages: []}});
    });
    await page.goto('http://goita.test/');
    await page.waitForFunction(() => typeof openAiReviewReport === 'function');
    await page.evaluate(fixture => {
      gid = DEBUG_GID; mySeat = 'A'; activeRoomId = gid;
      latestState = {owned_human_seats: ['A'], review_round_id: 'review-round', log: fixture.log};
      document.getElementById('gameView').style.display = 'block';
      document.getElementById('lobbyView').style.display = 'none';
      document.getElementById('logCard').style.display = 'block';
      document.querySelector('#logCard details').open = true;
      gameLogViewMode = 'normal'; renderGameLog(fixture.log);
    }, fixture);
    const headers = await page.locator('#log .game-log-entry-header').allTextContents();
    assert.deepEqual(headers.map(text => text.replace('この手を報告', '')), ['C手番1', 'D手番2', 'A手番3', 'B手番4']);
    await page.getByRole('button', {name: 'D手番2を報告', exact: true}).click();
    await page.locator('#aiReviewDecision').waitFor({state: 'visible'});
    assert.equal(await page.locator('#aiReviewDecision option').count(), 2);
    assert.equal(await page.locator('#aiReviewTurns button[aria-pressed=true] strong').textContent(), 'D手番2');
    const beforeReceive = await page.locator('#aiReviewPosition tr.ai-review-actor td').nth(1).textContent();
    await page.locator('#aiReviewDecision').selectOption('3');
    const beforeAttack = await page.locator('#aiReviewPosition tr.ai-review-actor td').nth(1).textContent();
    assert.equal(beforeReceive.split('し').length - beforeAttack.split('し').length, 1);
    assert.match(await page.locator('#aiReviewPosition').textContent(), /伏せ（非公開）/);
    await page.screenshot({path: path.resolve(__dirname, '../results/ai_review_report/select-desktop.png')});
    await page.locator('#aiReviewNext').click();
    await page.locator('#aiReviewNext').click();
    assert.equal(await page.locator('#aiReviewStep2').isVisible(), true); // Required fields block progress.
    await page.locator('#aiReviewPreferred').fill('しを攻める');
    await page.locator('#aiReviewReason').fill('しで継続したい。<script>悪意のない入力例</script>');
    await page.locator('#aiReviewPlan').fill('手駒全体から次の攻めを考える');
    // The live game changes while the user writes; the report must not follow it.
    await page.evaluate(() => {latestState = {owned_human_seats: ['A'], review_round_id: 'new-round', log: []};});
    await page.locator('#aiReviewNext').click();
    assert.match(await page.locator('#aiReviewPreview').textContent(), /D手番2/);
    const downloadEvent = page.waitForEvent('download');
    await page.locator('#aiReviewSave').click();
    const download = await downloadEvent;
    const exported = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
    assert.equal(exported.target.decision_log_index, 3);
    assert.equal(exported.target.turn_number, 2);
    assert.equal(exported.round_id, 'review-round');
    assert.equal(exported.human_review.preferred_action, 'しを攻める');
    assert.equal(exported.decisions[2].candidate_evaluations.chosen.attack, '7');
    assert.deepEqual(exported.log, fixture.log);
    assert.match(exported.human_review.reason, /<script>/);
    assert.match(download.suggestedFilename(), /_D_turn2\.json$/);
    await page.locator('#aiReviewClose').click();
    await page.locator('#aiReviewReportOpen').click();
    assert.equal(await page.locator('#aiReviewStep3').isVisible(), true);
    assert.equal(requests, 1); // Reopening resumes the same frozen draft.
    await page.locator('#aiReviewBack').click();
    await page.locator('#aiReviewUnknown').check();
    await page.locator('#aiReviewNext').click();
    await page.evaluate(() => {Object.defineProperty(navigator, 'clipboard', {configurable: true, value: {writeText: async text => {window.copiedReport = text;}}});});
    await page.locator('#aiReviewCopy').click();
    assert.match(await page.evaluate(() => window.copiedReport), /"preferred_action_unknown": true/);
    await page.setViewportSize({width: 390, height: 844});
    await page.locator('#aiReviewBack').click();
    await page.screenshot({path: path.resolve(__dirname, '../results/ai_review_report/input-mobile.png')});
    assert.equal(await page.evaluate(() => {
      const dialog = document.getElementById('aiReviewReportDialog');
      return dialog.scrollWidth <= dialog.clientWidth && dialog.getBoundingClientRect().right <= innerWidth;
    }), true);
    await page.locator('#aiReviewClose').click();
    failSnapshot = true;
    await page.evaluate(() => {latestState.review_round_id = 'review-round'; openAiReviewReport(3);});
    await page.waitForFunction(() => document.getElementById('aiReviewStatus').textContent.includes('局が切り替わりました'));
    assert.equal(await page.locator('#aiReviewSave').isVisible(), false);
    await page.locator('#aiReviewClose').click();
    await page.evaluate(() => {
      gid = 'main'; renderGameLog(latestState.log);
    });
    assert.equal(await page.locator('#aiReviewReportOpen').isVisible(), false);
    assert.deepEqual(errors, []);
    console.log('AI_REVIEW_REPORT_UI_OK');
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exit(1);});
