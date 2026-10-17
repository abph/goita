const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(
  path.join(__dirname, "../frontend/regionalAds.js"),
  "utf8",
);

const storage = () => {
  const values = new Map();
  return {
    getItem: key => values.has(key) ? values.get(key) : null,
    setItem: (key, value) => values.set(key, String(value)),
  };
};

const adElement = {hidden:true, dataset:{}};
const linkElement = {
  _href: "",
  addEventListener: () => {},
  removeAttribute(name) { if (name === "href") this._href = ""; },
  hasAttribute(name) { return name === "href" && !!this._href; },
  set href(value) { this._href = value; },
  get href() { return this._href; },
};
const elements = {
  regionalAd: adElement,
  regionalAdTitle: {textContent:""},
  regionalAdMessage: {textContent:""},
  regionalAdLink: linkElement,
  regionalAdClose: {addEventListener:() => {}},
};
const metricBodies = [];
const context = vm.createContext({
  Blob,
  Date,
  JSON,
  console,
  localStorage: storage(),
  sessionStorage: storage(),
  navigator: {},
  document: {
    getElementById: id => elements[id] || null,
    addEventListener: (_name, callback) => callback(),
  },
  fetch: async (url, options = {}) => {
    if (String(url).startsWith("/api/regional-ad?")) {
      return {
        ok: true,
        json: async () => ({
          ad: {id:"regional-test", title:"広告", message:"本文", url:"https://example.com/"},
        }),
      };
    }
    if (url === "/api/regional-ad/metric") {
      metricBodies.push(JSON.parse(options.body));
      return {ok:true};
    }
    throw new Error(`Unexpected request: ${url}`);
  },
  window: {goitaI18n:{refresh:() => {}}},
});

vm.runInContext(source, context);

(async () => {
  await context.window.goitaRegionalAds.sync("public_room", "room-a", "one");
  await context.window.goitaRegionalAds.sync("score_attack", "room-a", "two");
  await context.window.goitaRegionalAds.sync("public_room", "room-b", "three");

  const impressions = metricBodies.filter(item => item.event === "impression");
  assert.equal(impressions.length, 3);
  assert.equal(impressions[0].viewer_measurement, true);
  assert.equal(impressions[0].viewer_first, true);
  assert.equal(impressions[0].surface_viewer_first, true);
  assert.equal(impressions[1].viewer_first, false);
  assert.equal(impressions[1].surface_viewer_first, true);
  assert.equal(impressions[2].viewer_first, false);
  assert.equal(impressions[2].surface_viewer_first, false);

  await context.window.goitaRegionalAds.sync("score_attack", "room-b", "four");
  assert.equal(metricBodies.filter(item => item.event === "impression").length, 3);
  assert.equal(adElement.hidden, true);
  console.log("Regional ads: unique-browser estimate and daily frequency cap passed");
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
