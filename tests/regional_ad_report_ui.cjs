// Run with Playwright available on NODE_PATH. All HTTP responses are local fixtures.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');
const root = path.join(__dirname, '..');
const title = '<img src=x onerror=alert(1)> ごいた会';
const items = [
  {id:'a', title, status:'掲載中', impressions:100, clicks:2, click_rate:2, viewers:4,
    daily:[{date:'2026-11-01', impressions:100, clicks:2}],
    surfaces:[{surface:'public_room', impressions:100, clicks:2, click_rate:2}],
    regions:[{region:'関東', impressions:100, clicks:2, click_rate:2}]},
  {id:'b', title:'別の広告', status:'終了', impressions:10, clicks:3, click_rate:30, viewers:1,
    daily:[{date:'2026-11-03', impressions:10, clicks:3}], surfaces:[], regions:[]},
  {id:'c', title:'未表示', status:'停止中', impressions:0, clicks:0, click_rate:null, viewers:0,
    daily:[], surfaces:[], regions:[]},
];
const report = period => ({period, period_start:period === 'all' ? '2026-10-01' : '2026-11-01',
  period_end:'2026-11-03', summary:{impressions:110, clicks:5, click_rate:4.55}, items});

(async () => {
  const browser = await chromium.launch({headless:true,
    ...(process.env.EDGE_PATH ? {executablePath:process.env.EDGE_PATH} : {})});
  try {
    const page = await browser.newPage({viewport:{width:1280,height:1000}});
    const errors = [], requests = [];
    let mode = '', releaseSlow, slowSeen;
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', async route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/admin/') return route.fulfill({contentType:'text/html',
        body:fs.readFileSync(path.join(root,'frontend/admin.html'),'utf8')});
      if (url.pathname.startsWith('/static/')) {
        const file = path.join(root,'frontend',path.basename(url.pathname));
        return route.fulfill({contentType:file.endsWith('.js') ? 'application/javascript' : 'image/svg+xml',
          body:fs.existsSync(file) ? fs.readFileSync(file) : ''});
      }
      if (url.pathname === '/admin/api/regional-ads/report') {
        const period = url.searchParams.get('period'); requests.push(period);
        if (mode === 'race' && period === '7') {
          slowSeen(); await new Promise(resolve => {releaseSlow=resolve;});
        }
        if (mode === 'error' || mode === '401') return route.fulfill({status:mode === '401' ? 401 : 500,
          json:{detail:'テスト用の取得エラー'}});
        return route.fulfill({json:mode === 'empty' ? {...report(period),items:[],summary:{impressions:0,clicks:0,click_rate:null}} : report(period)});
      }
      return route.fulfill({json:{}});
    });
    await page.goto('http://ad-report.test/admin/');
    await page.locator('[data-tab="regionalAds"]').click();
    await page.locator('#regionalAdReportStatus').filter({hasText:'3件の広告'}).waitFor();
    assert.deepEqual(requests,['30']);
    assert.match(await page.locator('#regionalAdReportSummary').textContent(),/110回.*5回.*4.55%/s);
    assert.equal(await page.locator('#regionalAdSettingsPanel').isVisible(),false);
    const titles = () => page.locator('#regionalAdReportRows button').allTextContents();
    assert.deepEqual(await titles(),[title,'別の広告','未表示']);
    await page.locator('[data-ad-sort="click_rate"] button').click();
    assert.deepEqual(await titles(),['別の広告',title,'未表示']);
    await page.locator('[data-ad-sort="click_rate"] button').click();
    assert.deepEqual(await titles(),[title,'別の広告','未表示']);
    await page.getByRole('button',{name:title,exact:true}).click();
    assert.equal(await page.locator('#regionalAdReportDetailTitle').textContent(),title);
    assert.equal(await page.locator('#regionalAdReportDetailTitle img').count(),0);
    assert.equal(await page.locator('#regionalAdReportCharts svg').count(),2);
    assert.equal(await page.locator('#regionalAdReportDaily tbody tr').count(),3);
    assert.match(await page.locator('#regionalAdReportDaily tbody tr').nth(1).textContent(),/2026-11-0200/);
    assert.match(await page.locator('#regionalAdReportSurfaces').textContent(),/公開部屋/);
    assert.match(await page.locator('#regionalAdReportRegions').textContent(),/関東/);
    assert.match(await page.locator('#regionalAdReportBrowsers').textContent(),/全閲覧者数ではありません/);

    await page.locator('#regionalAdShowSettings').click();
    assert.equal(await page.locator('#regionalAdTwiplaUrl').isVisible(),true);
    assert.equal(await page.locator('#regionalAdReportPanel').isVisible(),false);
    await page.locator('#regionalAdShowReport').click();
    await page.locator('#regionalAdReportStatus').filter({hasText:'3件の広告'}).waitFor();
    await page.locator('[data-ad-period="all"]').click();
    await page.locator('#regionalAdReportRange').filter({hasText:'全期間'}).waitFor();
    assert.match(await page.locator('#regionalAdReportBrowsers').textContent(),/閲覧ブラウザー数/);

    // A slow earlier request must not replace a newer period's data.
    mode='race'; const pending = new Promise(resolve => {slowSeen=resolve;});
    await page.locator('[data-ad-period="7"]').click(); await pending;
    await page.locator('[data-ad-period="30"]').click();
    await page.locator('#regionalAdReportStatus').filter({hasText:'3件の広告'}).waitFor();
    releaseSlow();
    await page.waitForLoadState('networkidle');
    assert.equal(await page.locator('[data-ad-period="30"]').getAttribute('aria-pressed'),'true');
    assert.match(await page.locator('#regionalAdReportRange').textContent(),/^集計期間/);

    // Desktop and mobile widths: only the comparison table scrolls horizontally.
    for (const width of [1280,390,320]) {
      await page.setViewportSize({width,height:1000});
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),true,`overflow at ${width}px`);
      assert.equal(await page.locator('#regionalAdReportCharts svg').count(),2);
      if (process.env.AD_REPORT_SCREENSHOTS) await page.screenshot({path:path.join(process.env.AD_REPORT_SCREENSHOTS,`ad-report-${width}.png`),fullPage:true});
    }
    await page.getByRole('button',{name:'未表示',exact:true}).click();
    assert.match(await page.locator('#regionalAdReportCharts').textContent(),/実績はありません/);
    mode='empty'; await page.locator('#regionalAdReportRefresh').click();
    await page.locator('#regionalAdReportStatus').filter({hasText:'0件の広告'}).waitFor();
    assert.match(await page.locator('#regionalAdReportRows').textContent(),/広告はまだありません/);
    mode='error'; await page.locator('#regionalAdReportRefresh').click();
    await page.locator('#regionalAdReportStatus').filter({hasText:'取得エラー'}).waitFor();
    assert.match(await page.locator('#regionalAdReportRows').textContent(),/再試行/);
    assert.equal(await page.locator('#regionalAdReportDetail').isVisible(),false);
    mode='401'; await page.locator('#regionalAdReportRefresh').click();
    await page.locator('#authView').waitFor({state:'visible'});
    assert.deepEqual(errors,[]);
    console.log('Ad report UI: periods, sorting, details, race, errors, login, settings and desktop/mobile layout passed');
  } finally { await browser.close(); }
})().catch(error => {console.error(error);process.exitCode=1;});
