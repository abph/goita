const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../frontend/admin.html'), 'utf8');
const start = html.indexOf('function analyticsRangeParams()');
const end = html.indexOf('function surveyLabel(', start);
assert.ok(start >= 0 && end > start);
assert.ok(html.includes('id="analyticsExport"'));
assert.ok(html.includes('addEventListener("click", exportAnalytics)'));
const elements = {analyticsDays: {value: '7'}, analyticsStartDate: {value: ''},
  analyticsEndDate: {value: ''}, analyticsExport: {disabled: false}, analyticsStatus: {textContent: ''}};
const calls = [], downloads = [], blobs = [], revoked = [], rendered = [];
let fail = false, logins = 0;
const data = {period_start:'2026-10-16', period_end:'2026-10-22', visitors:[{visitor_number:'visitor_0001'}], sessions:[{}]};
const context = vm.createContext({URLSearchParams, Blob,
  URL: {createObjectURL: blob => {blobs.push(blob); return 'blob:export';}, revokeObjectURL: url => revoked.push(url)},
  document: {getElementById: id => elements[id], body: {appendChild() {}},
    createElement: () => ({click() {downloads.push({name: this.download, href: this.href});}, remove() {}})},
  api: async url => {calls.push(url); if(fail) throw Object.assign(new Error('login required'), {status:401}); return data;},
  showLogin: () => logins++, renderAnalytics: value => rendered.push(value),
  setTimeout: callback => callback(),
});
vm.runInContext(html.slice(start, end), context);
(async () => {
  await context.exportAnalytics();
  assert.equal(calls[0], '/admin/api/analytics/export?days=7');
  assert.equal(downloads[0].name, 'goita-usage-2026-10-16-2026-10-22.json');
  assert.deepEqual(JSON.parse(await blobs[0].text()), data);
  assert.equal(elements.analyticsExport.disabled, false);
  assert.deepEqual(revoked, ['blob:export']);
  assert.match(elements.analyticsStatus.textContent, /1件の匿名利用者・1件の訪問/);

  elements.analyticsDays.value = 'custom';
  elements.analyticsStartDate.value = '2026-10-01';
  elements.analyticsEndDate.value = '2026-10-10';
  await context.exportAnalytics();
  assert.equal(calls[1], '/admin/api/analytics/export?start_date=2026-10-01&end_date=2026-10-10');
  await context.loadAnalytics();
  assert.equal(calls[2], '/admin/api/analytics?start_date=2026-10-01&end_date=2026-10-10&recent_limit=100');
  assert.equal(rendered.length, 1);

  elements.analyticsEndDate.value = '';
  await context.exportAnalytics();
  assert.equal(calls.length, 3);
  assert.match(elements.analyticsStatus.textContent, /開始日と終了日/);
  assert.equal(elements.analyticsExport.disabled, false);
  elements.analyticsEndDate.value = '2026-09-30';
  await context.exportAnalytics();
  assert.equal(calls.length, 3);
  assert.match(elements.analyticsStatus.textContent, /終了日は開始日以降/);

  elements.analyticsDays.value = '30'; fail = true;
  await context.exportAnalytics();
  assert.equal(logins, 1);
  assert.equal(downloads.length, 2);
  assert.equal(elements.analyticsExport.disabled, false);
  console.log('Analytics export UI: preset/custom range, full JSON download, validation and login failures passed');
})().catch(error => {console.error(error);process.exitCode = 1;});
