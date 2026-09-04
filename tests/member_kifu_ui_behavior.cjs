const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../frontend/index.html'), 'utf8');
const source = html.slice(html.indexOf('async function researchKifuApi('), html.indexOf('function researchKifuDateLabel('));

(async () => {
  let captured;
  let reset = 0;
  let refreshed = 0;
  const context = vm.createContext({
    uiText: text => text,
    resetMemberKifuLibrary: () => reset++,
    window: {goitaMembers: {refresh: () => refreshed++}},
    fetch: async (url, options) => {
      captured = {url, options};
      return {ok: true, json: async () => ({records: []})};
    },
  });
  vm.runInContext('let memberKifuRevision = 0;' + source, context);
  await context.researchKifuApi('/list');
  assert.equal(captured.url, '/api/member/kifu/list');
  assert.equal(captured.options.credentials, 'same-origin');
  assert.equal(captured.options.cache, 'no-store');
  assert.equal(captured.options.headers['X-Goita-Member'], '1');
  assert.deepEqual(JSON.parse(captured.options.body), {});

  let finish;
  context.fetch = () => new Promise(resolve => {finish = resolve;});
  const pending = context.researchKifuApi('/record');
  vm.runInContext('memberKifuRevision++;', context);
  finish({ok: true, json: async () => ({record: {payload: 'old account data'}})});
  await assert.rejects(pending, /ログイン状態/);

  context.fetch = async () => ({ok: false, status: 401, json: async () => ({detail: 'expired'})});
  await assert.rejects(context.researchKifuApi('/list'), /expired/);
  assert.equal(reset, 1);
  assert.equal(refreshed, 1);

  // Compile every inline script as well as exercising the reused text exporter.
  for (const script of html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)) new vm.Script(script[1]);
  const exportSource = html.slice(html.indexOf('function researchKifuYamlValue('), html.indexOf('function downloadSelectedResearchKifu('));
  vm.runInContext(exportSource, context);
  const text = context.researchKifuDownloadText({payload: {dealer: 'A', hand: {p0: 'し', p1: 'し', p2: 'し', p3: 'し'}, game: [['0','し','し']], player_names: {A: 'Alice'}}});
  assert.match(text, /version: 1\.0/);
  assert.match(text, /Alice/);
  console.log('Member library UI: authenticated requests, stale-response rejection, expiry, and export passed');
})().catch(error => {console.error(error); process.exitCode = 1;});
