// Offline browser check; use freshly generated review.html / result.html.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require('playwright');

(async () => {
  const reviewPath = path.resolve(process.argv[2] || 'private_data/ai_decision_check/review.html');
  const output = path.resolve(process.argv[3] || '.server_deps_divergent/decision-check-ui');
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.PLAYWRIGHT_BROWSER_EXECUTABLE || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
  try {
    const context = await browser.newContext({acceptDownloads: true, viewport: {width: 1100, height: 850}});
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.route('http://**/*', route => route.abort());
    await page.route('https://**/*', route => route.abort());
    await page.goto(pathToFileURL(reviewPath).href);
    await page.getByRole('heading', {name: 'AI判断チェック・正解の確認', exact: true}).waitFor();
    const headingBefore = await page.locator('#main h2').textContent();
    assert(await page.getByText('B手番2：しで受ける → しで攻める', {exact: true}).isVisible());
    const row = page.locator('.routes tr').filter({hasText: '銀で受ける → 香で攻める'});
    await row.locator('select').selectOption('best');
    // Selecting a grade alone is not approval and does not leave the case.
    assert.match(await page.locator('#main .notice').textContent(), /下書き/);
    assert.equal(await page.locator('#main h2').textContent(), headingBefore);
    await page.locator('textarea').fill('UI検証：公開情報から味方への引き継ぎを優先する。');
    await page.getByRole('button', {name: 'この回答を確定', exact: true}).click();
    assert.match(await page.locator('#main .notice').textContent(), /確定済み/);
    // Editing an already confirmed answer revokes its approval.
    await page.locator('textarea').fill('UI検証：理由を修正した下書き。');
    assert.match(await page.locator('#main .notice').textContent(), /下書き/);
    await page.getByRole('button', {name: 'この回答を確定', exact: true}).click();
    const downloaded = page.waitForEvent('download');
    await page.getByRole('button', {name: '確定済み回答を書き出す', exact: true}).click();
    const download = await downloaded;
    const answerPath = path.join(output, 'browser-answers.json');
    await download.saveAs(answerPath);
    const answer = JSON.parse(fs.readFileSync(answerPath, 'utf8'));
    const d = answer.cases.find(c => c.answer.reason === 'UI検証：理由を修正した下書き。');
    assert(d && d.answer.confirmed && d.style === 'safe');
    assert(d.answer.best.some(r => JSON.stringify(r) === JSON.stringify([['receive', '4', null], ['attack', null, '2']])));
    await page.setViewportSize({width: 390, height: 844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({path: path.join(output, 'review-mobile.png'), fullPage: true});
    const resultPath = path.join(path.dirname(reviewPath), 'result.html');
    if (fs.existsSync(resultPath)) {
      await page.goto(pathToFileURL(resultPath).href);
      await page.getByRole('heading', {name: 'AI判断チェック・実行結果', exact: true}).waitFor();
      const count = await page.locator('#main > .card').count();
      assert(count > 0);
      await page.locator('#toolbar select').selectOption('attention');
      await page.locator('#toolbar select').selectOption('all');
      assert.equal(await page.locator('#main > .card').count(), count);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      await page.setViewportSize({width: 1100, height: 850});
      await page.screenshot({path: path.join(output, 'results-desktop.png'), fullPage: true});
    }
    assert.deepEqual(errors, []);
    console.log('AI判断チェック UI: approval, draft invalidation, JSON export, filters, mobile layout passed');
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error);process.exitCode = 1;});
