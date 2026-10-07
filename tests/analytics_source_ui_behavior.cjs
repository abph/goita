const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const html = fs.readFileSync(path.join(__dirname, '../frontend/index.html'), 'utf8');
const start = html.indexOf('function analyticsAttribution()');
const end = html.indexOf('function analyticsRoomType(', start);
assert.ok(start >= 0 && end > start);

const values = new Map();
const sourceKey = 'goita_analytics_source';
const sessionKey = 'goita_analytics_session_id';
const context = vm.createContext({
  URL,
  URLSearchParams,
  location: {search: '?utm_source=vrcgoita'},
  document: {referrer: ''},
  sessionStorage: {
    getItem: key => values.has(key) ? values.get(key) : null,
    setItem: (key, value) => values.set(key, value),
    removeItem: key => values.delete(key),
  },
  ANALYTICS_SOURCE_KEY: sourceKey,
  ANALYTICS_SESSION_KEY: sessionKey,
});

vm.runInContext(html.slice(start, end), context);
values.set(sourceKey, JSON.stringify({source: 'direct', medium: '', campaign: ''}));
values.set(sessionKey, 'session_direct');
let attribution = context.analyticsAttribution();
assert.equal(attribution.source, 'vrcgoita');
assert.equal(Object.hasOwn(attribution, 'referrer_url'), false);
assert.equal(values.has(sessionKey), false);

values.set(sessionKey, 'session_vrcgoita');
attribution = context.analyticsAttribution();
assert.equal(values.get(sessionKey), 'session_vrcgoita');

values.clear();
context.location.search = '';
context.document.referrer = 'https://vrcgoita.com/ai/?member=secret#chapter';
attribution = context.analyticsAttribution();
assert.equal(attribution.source, 'vrcgoita');
assert.equal(Object.hasOwn(attribution, 'referrer_url'), false);

values.clear();
context.document.referrer = 'https://example.com/another-page?secret=yes';
attribution = context.analyticsAttribution();
assert.equal(attribution.source, 'example.com');
assert.equal(Object.hasOwn(attribution, 'referrer_url'), false);

console.log('Analytics source: coarse attribution and VRCとごいた canonicalization passed');
