const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../frontend/index.html'), 'utf8');
const initialization = html.slice(html.indexOf('let analyticsVisitRecorded = false;'), html.indexOf('function randomAnalyticsToken('));
const tracking = html.slice(html.indexOf('function trackAnalytics('), html.indexOf('async function deleteAnalyticsHistory('));
let allowed = false;
let contexts = 0;
const events = [];
const context = vm.createContext({
  personalSettings: {enableAnalytics: true}, API: '',
  document: {getElementById: () => ({style: {display: 'none'}})},
  analyticsRoomType: () => 'main',
  analyticsContext: () => { contexts++; return {}; },
  fetch: (url, options) => {events.push(JSON.parse(options.body).event); return Promise.resolve();},
  navigator: {sendBeacon: () => {events.push('beacon');}}, Blob, console,
  goitaMembers: {shouldRecordAnalytics: () => allowed},
});
context.window = context;
vm.runInContext(initialization + tracking, context);
// Both unresolved sessions and authenticated operators suppress all paths.
context.onMemberSessionReady();
context.trackAnalytics('site_visit');
context.trackAnalytics('room_leave', {}, 'main', true);
assert.equal(contexts, 0);
assert.deepEqual(events, []);
allowed = true;
context.onMemberSessionReady();
context.onMemberSessionReady();
assert.deepEqual(events, ['site_visit']);
context.trackAnalytics('heartbeat');
assert.deepEqual(events, ['site_visit', 'heartbeat']);
allowed = false;
context.trackAnalytics('room_leave', {}, 'main', true);
assert.equal(events.length, 2);
allowed = true;
context.personalSettings.enableAnalytics = false;
context.trackAnalytics('heartbeat');
assert.equal(events.length, 2);
console.log('Operator analytics UI: initial session gate, single visit, operator exclusion, beacon and opt-out passed');
