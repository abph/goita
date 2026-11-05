const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../frontend/admin.html'), 'utf8');
const start = html.indexOf('let regionalAds = []');
const end = html.indexOf('function renderRegionalAdCarouselSets()', start);
assert.ok(start > 0 && end > start);
assert.ok(html.includes('addEventListener("click", importRegionalAdFromTwipla)'));
assert.match(html, /id="regionalAdEndsAt"[^>]*required/);
const fields = Object.fromEntries(['Title','Message','Url','BackgroundTheme','StartsAt','EndsAt',
  'Enabled','TwiplaUrl','TwiplaStatus','TwiplaEvent','TwiplaImport'].map(name => [
    `regionalAd${name}`, {value:'', textContent:'', hidden:false, checked:false, disabled:false},
  ]));
const targets = [{value:'関東', checked:false}, {value:'中部', checked:false}];
const surfaces = [{value:'public_room', checked:false}, {value:'score_attack', checked:false}];
const requests = [];
let fail = null, deferred = null, logins = 0;
const result = {
  event:{title:'はまりばカフェ ごいた会', venue:'はまりばカフェ', starts_at:'2066-11-22T11:00:00+09:00', ends_at:'2066-11-22T16:30:00+09:00'},
  draft:{title:'はまりばカフェ ごいた会', message:'11/22 11:00〜16:30\n初心者歓迎。', url:'https://twipla.jp/events/747979',
    starts_at:'2066-11-05T09:00:00+09:00', ends_at:'2066-11-22T16:30:00+09:00'}, warnings:[],
};
const context = vm.createContext({Date,
  document:{getElementById:id => fields[id], querySelectorAll:selector => selector.includes('Targets') ? targets : surfaces},
  api: async (url, options) => {
    requests.push({url, options});
    if (fail) throw fail;
    if (deferred) return await new Promise(resolve => {deferred.resolve = resolve;});
    return JSON.parse(JSON.stringify(result));
  },
  showLogin: () => logins++,
});
vm.runInContext(html.slice(start, end), context);
const f = name => fields[`regionalAd${name}`];
(async () => {
  context.resetRegionalAdForm();
  await context.importRegionalAdFromTwipla();
  assert.equal(requests.length, 0);
  f('TwiplaUrl').value = 'https://www.twipla.jp/events/747979';
  f('BackgroundTheme').value = 'green'; targets[0].checked = true; f('Enabled').checked = true;
  await context.importRegionalAdFromTwipla();
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, '/admin/api/regional-ads/import-twipla');
  assert.equal(requests[0].options.method, 'POST');
  assert.deepEqual(JSON.parse(requests[0].options.body), {url:'https://www.twipla.jp/events/747979'});
  assert.equal(f('EndsAt').value, '2066-11-22T16:30');
  assert.equal(f('StartsAt').value, '2066-11-05T09:00');
  assert.equal(f('Title').value, result.draft.title);
  assert.equal(f('Url').value, result.draft.url);
  assert.equal(f('Enabled').checked, false, 'import cannot silently publish');
  assert.equal(f('BackgroundTheme').value, 'green');
  assert.equal(targets[0].checked, true); assert.equal(targets[1].checked, false);
  assert.ok(surfaces.every(item => item.checked));
  assert.match(f('TwiplaStatus').textContent, /まだ保存・掲載していません/);
  assert.match(f('TwiplaEvent').textContent, /はまりばカフェ/);

  result.draft.ends_at = null; result.event.ends_at = null;
  result.warnings = ['終了時刻を取得できませんでした。掲載終了を入力してください。'];
  await context.importRegionalAdFromTwipla();
  assert.equal(f('EndsAt').value, '');
  assert.match(f('TwiplaStatus').textContent, /掲載終了を入力/);
  assert.match(f('TwiplaEvent').textContent, /終了時刻未取得/);

  f('Title').value = '手入力を保持'; fail = new Error('取得できませんでした');
  await context.importRegionalAdFromTwipla();
  assert.equal(f('Title').value, '手入力を保持');
  assert.equal(f('TwiplaImport').disabled, false);
  assert.match(f('TwiplaStatus').textContent, /取得できません/);
  fail = Object.assign(new Error('管理者ログインが必要です'), {status:401});
  await context.importRegionalAdFromTwipla(); assert.equal(logins, 1);
  fail = null;

  deferred = {};
  const pending = context.importRegionalAdFromTwipla();
  assert.equal(f('TwiplaImport').disabled, true);
  context.resetRegionalAdForm();
  deferred.resolve(result); await pending;
  assert.equal(f('Title').value, '', 'a late response cannot overwrite a reset/editor switch');

  f('TwiplaUrl').value = 'https://twipla.jp/events/747979'; deferred = {};
  const changedUrl = context.importRegionalAdFromTwipla();
  f('TwiplaUrl').value = 'https://twipla.jp/events/123';
  deferred.resolve(result); await changedUrl;
  assert.equal(f('Title').value, '');
  assert.match(f('TwiplaStatus').textContent, /URLが変更/);
  assert.equal(f('TwiplaImport').disabled, false);
  assert.ok(requests.every(item => item.url.endsWith('/import-twipla')), 'no save request on import');
  console.log('TwiPla import UI: draft only, JST dates, missing end, failure preservation, login and stale responses passed');
})().catch(error => {console.error(error);process.exitCode = 1;});
