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
    this.style.setProperty = (key, value) => { this.style[key] = value; };
    this.style.getPropertyValue = key => this.style[key] || "";
    this.style.removeProperty = key => { delete this.style[key]; };
    this.hidden = false;
    this.className = "";
    this.attributes = new Map();
    this.listeners = new Map();
    this.classList = {
      contains: name => this.className.split(/\s+/).includes(name),
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
  gameView: new TestElement("div"),
  chatPanel: new TestElement("div"),
  logCard: new TestElement("div"),
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
const playLayout = new TestElement("div");
const mainColumn = new TestElement("div");
mainColumn.className = "play-main-column";
playLayout.append(mainColumn, elements.chatPanel);
elements.chatPanel.parentElement = playLayout;
elements.chatPanel.after = slot => {
  renderedPosition = "side";
  slot.previousElementSibling = elements.chatPanel;
};
mainColumn.getBoundingClientRect = () => ({width:594});
elements.chatPanel.getBoundingClientRect = () => ({
  width: elements.gameView.classList.contains("desktop-log-visible") ? 220 : 240,
});
elements.logCard.getBoundingClientRect = () => ({width:280});

const metricBodies = [];
const scheduled = new Map();
const animationFrames = [];
const windowListeners = new Map();
let timerNumber = 0;
const viewport = {clientWidth:900};
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
    documentElement: viewport,
    querySelector: selector => selector === ".play-layout" ? playLayout : mainColumn,
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
            {id:"regional-a", title:"広告A", message:"本文A", url:"https://example.com/a", background_theme:"blue"},
            {id:"regional-b", title:"広告B", message:"本文B", url:"https://example.com/b", background_theme:"green"},
            {id:"regional-c", title:"広告C", message:"本文C", url:"https://example.com/c", background_theme:"orange"},
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
    innerWidth:900,
    getComputedStyle: () => ({columnGap:"16px"}),
    addEventListener: (name, callback) => windowListeners.set(name, callback),
    requestAnimationFrame: callback => { animationFrames.push(callback); return animationFrames.length; },
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

function resize(width) {
  viewport.clientWidth = width;
  context.window.innerWidth = width;
  elements.regionalAdSlot.previousElementSibling = null;
  windowListeners.get("resize")();
  while (animationFrames.length) animationFrames.shift()();
}

(async () => {
  await context.window.goitaRegionalAds.sync("public_room", "room-a", "one");
  assert.equal(elements.regionalAd.hidden, false);
  assert.equal(renderedPosition, "bottom");
  assert.equal(elements.regionalAdSlot.dataset.position, "bottom");
  assert.equal(elements.regionalAdTrack.children.length, 3);
  assert.deepEqual(
    elements.regionalAdTrack.children.map(item => item.dataset.theme),
    ["blue", "green", "orange"],
  );
  assert.equal(elements.regionalAdDots.children.length, 3);
  assert.equal(elements.regionalAdTrack.style.transform, "translateX(-0%)");

  runNextTimer();
  assert.equal(elements.regionalAdTrack.style.transform, "translateX(-100%)");
  runNextTimer();
  assert.equal(elements.regionalAdTrack.style.transform, "translateX(-200%)");
  runNextTimer();

  let impressions = metricBodies.filter(item => item.event === "impression");
  assert.equal(impressions.length, 1, "automatic slides do not count as impressions");
  assert.deepEqual(impressions.map(item => item.ad_id), ["regional-a"]);
  assert.ok(impressions.every(item => item.viewer_measurement === true));
  assert.ok(impressions.every(item => item.viewer_first === true));

  resize(1200);
  assert.equal(elements.regionalAdSlot.dataset.position, "side", "spare width puts the same ad beside chat");
  assert.equal(elements.regionalAdSlot.previousElementSibling, elements.chatPanel);
  assert.equal(elements.gameView.style.getPropertyValue("--regional-ad-layout-width"), "1066px");
  elements.gameView.classList.toggle("desktop-log-visible", true);
  resize(1200);
  assert.equal(elements.regionalAdSlot.dataset.position, "bottom", "the desktop log must also fit");
  assert.equal(elements.gameView.classList.contains("regional-ad-side-visible"), false);
  resize(1440);
  assert.equal(elements.regionalAdSlot.dataset.position, "side");
  assert.equal(elements.gameView.style.getPropertyValue("--regional-ad-layout-width"), "1342px");
  elements.chatPanel.parentElement = new TestElement("aside");
  resize(1440);
  assert.equal(elements.regionalAdSlot.dataset.position, "bottom", "a waiting-room chat outside the play layout is not used for side ads");
  elements.chatPanel.parentElement = playLayout;
  resize(390);
  assert.equal(elements.regionalAdSlot.dataset.position, "bottom", "mobile keeps the configured ad position");
  assert.equal(elements.gameView.style.getPropertyValue("--regional-ad-layout-width"), "");
  assert.equal(metricBodies.filter(item => item.event === "impression").length, 1, "moving the slot does not add impressions");
  elements.gameView.classList.toggle("desktop-log-visible", false);

  elements.regionalAdViewport.listeners.get("touchstart")({changedTouches:[{clientX:100}]});
  elements.regionalAdViewport.listeners.get("touchend")({changedTouches:[{clientX:20}]});
  impressions = metricBodies.filter(item => item.event === "impression");
  assert.equal(impressions.length, 2, "a user swipe counts the selected ad");
  assert.equal(impressions[1].ad_id, "regional-b");

  await context.window.goitaRegionalAds.sync("public_room", "room-b", "two");
  const nextImpressions = metricBodies.filter(item => item.event === "impression");
  assert.equal(nextImpressions.length, 3);
  assert.equal(nextImpressions[2].ad_id, "regional-b", "the starting ad rotates between visits");
  assert.equal(nextImpressions[2].viewer_first, false);
  assert.equal(nextImpressions[2].surface_viewer_first, false);

  for (const [roomId, exposureId] of [
    ["room-c", "three"], ["room-d", "four"], ["room-e", "five"],
    ["room-f", "six"], ["room-g", "seven"], ["room-h", "eight"],
    ["room-i", "nine"],
  ]) {
    await context.window.goitaRegionalAds.sync("public_room", roomId, exposureId);
  }
  assert.equal(elements.regionalAd.hidden, true, "ads disappear after three daily views each");
  assert.equal(metricBodies.filter(item => item.event === "impression").length, 9);

  console.log("Regional ads: responsive sidebar, automatic rotation, explicit impression counting, and daily cap passed");
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
