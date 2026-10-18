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

class TestElement {
  constructor(tagName = "div") {
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.dataset = {};
    this.style = {};
    this.hidden = false;
    this.className = "";
    this.attributes = new Map();
    this.listeners = new Map();
    this.classList = {
      toggle: (name, enabled) => {
        const names = new Set(this.className.split(/\s+/).filter(Boolean));
        if (enabled) names.add(name); else names.delete(name);
        this.className = [...names].join(" ");
      },
    };
  }
  addEventListener(name, callback) { this.listeners.set(name, callback); }
  append(...children) { this.children.push(...children); }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  contains(target) { return target === this || this.children.some(child => child.contains?.(target)); }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  querySelectorAll(selector) {
    const matches = child => selector.startsWith(".")
      ? child.className.split(/\s+/).includes(selector.slice(1))
      : child.tagName.toLowerCase() === selector.toLowerCase();
    const found = [];
    const visit = child => {
      if (matches(child)) found.push(child);
      child.children?.forEach(visit);
    };
    this.children.forEach(visit);
    return found;
  }
}

let renderedPosition = "";
const elements = {
  regionalAd: new TestElement("aside"),
  regionalAdClose: new TestElement("button"),
  regionalAdViewport: new TestElement("div"),
  regionalAdTrack: new TestElement("div"),
  regionalAdDots: new TestElement("div"),
  regionalAdSlot: new TestElement("div"),
  regionalAdTopAnchor: {after: () => { renderedPosition = "top"; }},
  regionalAdBottomAnchor: {before: () => { renderedPosition = "bottom"; }},
};
elements.regionalAd.hidden = true;

const metricBodies = [];
const scheduled = new Map();
let timerNumber = 0;
const context = vm.createContext({
  Blob,
  Date,
  JSON,
  console,
  localStorage: storage(),
  sessionStorage: storage(),
  navigator: {},
  document: {
    hidden: false,
    activeElement: null,
    getElementById: id => elements[id] || null,
    createElement: tag => new TestElement(tag),
    addEventListener: (name, callback) => { if (name === "DOMContentLoaded") callback(); },
  },
  fetch: async (url, options = {}) => {
    if (String(url).startsWith("/api/regional-ad?")) {
      return {
        ok: true,
        json: async () => ({
          position: "bottom",
          ads: [
            {id:"regional-a", title:"広告A", message:"本文A", url:"https://example.com/a"},
            {id:"regional-b", title:"広告B", message:"本文B", url:"https://example.com/b"},
            {id:"regional-c", title:"広告C", message:"本文C", url:"https://example.com/c"},
          ],
        }),
      };
    }
    if (url === "/api/regional-ad/metric") {
      metricBodies.push(JSON.parse(options.body));
      return {ok:true};
    }
    throw new Error(`Unexpected request: ${url}`);
  },
  window: {
    goitaI18n:{refresh:() => {}},
    matchMedia:() => ({matches:false}),
    setTimeout: callback => { const id = ++timerNumber; scheduled.set(id, callback); return id; },
    clearTimeout: id => scheduled.delete(id),
  },
});

vm.runInContext(source, context);

function runNextTimer() {
  const [id, callback] = scheduled.entries().next().value || [];
  assert.ok(callback, "carousel timer should be scheduled");
  scheduled.delete(id);
  callback();
}

(async () => {
  await context.window.goitaRegionalAds.sync("public_room", "room-a", "one");
  assert.equal(elements.regionalAd.hidden, false);
  assert.equal(renderedPosition, "bottom");
  assert.equal(elements.regionalAdSlot.dataset.position, "bottom");
  assert.equal(elements.regionalAdTrack.children.length, 3);
  assert.equal(elements.regionalAdDots.children.length, 3);
  assert.equal(elements.regionalAdTrack.style.transform, "translateX(-0%)");

  runNextTimer();
  assert.equal(elements.regionalAdTrack.style.transform, "translateX(-100%)");
  runNextTimer();
  assert.equal(elements.regionalAdTrack.style.transform, "translateX(-200%)");
  runNextTimer();

  const impressions = metricBodies.filter(item => item.event === "impression");
  assert.equal(impressions.length, 3, "one page context counts each visible ad only once");
  assert.deepEqual(impressions.map(item => item.ad_id), ["regional-a", "regional-b", "regional-c"]);
  assert.ok(impressions.every(item => item.viewer_measurement === true));
  assert.ok(impressions.every(item => item.viewer_first === true));

  await context.window.goitaRegionalAds.sync("public_room", "room-b", "two");
  const nextImpressions = metricBodies.filter(item => item.event === "impression");
  assert.equal(nextImpressions.length, 4);
  assert.equal(nextImpressions[3].ad_id, "regional-b", "the starting ad rotates between visits");
  assert.equal(nextImpressions[3].viewer_first, false);
  assert.equal(nextImpressions[3].surface_viewer_first, false);

  runNextTimer();
  runNextTimer();
  await context.window.goitaRegionalAds.sync("public_room", "room-c", "three");
  runNextTimer();
  runNextTimer();
  assert.equal(metricBodies.filter(item => item.event === "impression").length, 9);

  await context.window.goitaRegionalAds.sync("public_room", "room-d", "four");
  assert.equal(elements.regionalAd.hidden, true, "ads disappear after three daily views each");
  assert.equal(metricBodies.filter(item => item.event === "impression").length, 9);

  console.log("Regional ads: carousel rotation, fair start, deduplication, and daily cap passed");
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
