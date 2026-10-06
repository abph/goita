const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const html = fs.readFileSync(path.join(__dirname, '../frontend/index.html'), 'utf8');
const start = html.indexOf('function analyticsAttribution()');
const end = html.indexOf('function analyticsRoomType(', start);
assert.ok(start >= 0 && end > start);

const values = new Map();
const context = vm.createContext({
  URL,
  URLSearchParams,
  location: {search: ''},
  document: {referrer: 'https://vrcgoita.com/goita/rules/?member=secret#chapter'},
  sessionStorage: {
    getItem: key => values.has(key) ? values.get(key) : null,
    setItem: (key, value) => values.set(key, value),
  },
  ANALYTICS_SOURCE_KEY: 'goita_analytics_source',
});

vm.runInContext(html.slice(start, end), context);
const attribution = context.analyticsAttribution();
assert.equal(attribution.source, 'vrcgoita.com');
assert.equal(attribution.referrer_url, 'https://vrcgoita.com/goita/rules/');

context.document.referrer = 'https://example.com/another-page';
assert.equal(
  context.analyticsAttribution().referrer_url,
  'https://vrcgoita.com/goita/rules/',
);

console.log('Analytics referrer: path capture, sensitive-part removal and session first-touch passed');
