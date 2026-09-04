const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const root = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(root, "frontend/index.html"), "utf8");
const memberScript = fs.readFileSync(path.join(root, "frontend/member.js"), "utf8");

function element() {
  return {
    children: [],
    replaceChildren() { this.children = []; },
    setAttribute() {},
    appendChild(child) { this.children.push(child); },
    append(...children) { this.children.push(...children); },
    addEventListener() {},
  };
}

async function check(member, gid, roomCount) {
  const pickers = { chatStampPicker: element(), lobbyChatStampPicker: element(), memberKifuParking: element(), researchKifuPanel: element() };
  const context = vm.createContext({
    document: {
      querySelectorAll: () => [],
      getElementById: id => pickers[id],
      createElement: element,
    },
    fetch: async () => ({ ok: true, json: async () => ({ member }) }),
    uiText: text => text,
    buildChatStampVisual: () => element(),
    resetMemberKifuLibrary() {},
    sendChatStamp() {},
    gid,
    PRIVATE_ROOM_IDS: new Set(["private"]),
    MAIN_ROOM_IDS: new Set(["main"]),
    DEBUG_GID: "debug",
  });
  context.window = context;
  vm.runInContext(html.slice(html.indexOf("const CHAT_STAMP_DEFINITIONS"), html.indexOf("const CHAT_STAMP_ASSET_VERSION")), context);
  vm.runInContext(html.slice(html.indexOf("function initializeChatStampPickers()"), html.indexOf("function sendChatStamp(kind")), context);
  vm.runInContext(memberScript, context);
  // Let the initial asynchronous session request render its result.
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(pickers.chatStampPicker.children.length, roomCount);
  assert.equal(pickers.lobbyChatStampPicker.children.length, 4);
  return context;
}

(async () => {
  await check(null, "main", 4);
  const paid = { paid_active: true, must_change_password: false, paid_until: null };
  await check(paid, "main", 10);
  await check({ ...paid, must_change_password: true }, "main", 4);
  await check({ ...paid, paid_active: false }, "main", 4);
  await check({ ...paid, paid_until: "2000-01-01" }, "main", 4);
  await check(null, "private", 10);
  await check(null, "debug", 10);
  const context = await check(paid, "main", 10);
  context.fetch = async () => ({ ok: true, json: async () => ({ member: null }) });
  await context.goitaMembers.refresh();
  assert.equal(context.document.getElementById("chatStampPicker").children.length, 4);
  console.log("Member stamp UI: 8 scenarios passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
