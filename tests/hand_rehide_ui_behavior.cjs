const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.join(__dirname, "../frontend/index.html"), "utf8");
const calls = [];
const panel = { style: {}, children: [], appendChild(child) { this.children.push(child); }, set innerHTML(_) { this.children = []; } };
const context = vm.createContext({
  URLSearchParams, API: "", gid: "main", mySeat: "A", clientId: "client-a",
  personalSettings: { autoRevealOwnHand: true, autoRevealAiHands: true },
  autoRevealHandsInFlight: false, autoRevealHandsRoundKey: "",
  isSpectator: () => false, isMainRoomId: () => true, isCurrentClientHost: () => true,
  isSeatHandRevealed: (state, seat) => state.revealed_hand_seats.includes(seat),
  uiText: text => text,
  document: { getElementById: () => panel, createElement: () => ({}) },
  fetch: async url => { calls.push(new URL(url, "http://test")); return { ok: true }; },
  refresh: async () => {},
  confirmHandReveal: async () => { throw new Error("Hiding must not ask for confirmation"); },
});
vm.runInContext(html.slice(html.indexOf("async function requestSeatHandReveal("), html.indexOf("async function submitAction(")), context);

(async () => {
  const state = { is_started: true, finished: true, owned_human_seats: ["A"], ai_seats: ["B", "C"],
    revealed_hand_seats: ["A", "B"], auto_reveal_blocked_seats: ["C"], round_count: 1 };
  context.renderHandRevealControls(state);
  assert.equal(panel.children[0].textContent, "自分の手札を非公開に戻す");
  assert.equal(panel.children[0].disabled, false);
  assert.equal(panel.children[1].textContent, "B（AI） 手札を非公開に戻す");
  await panel.children[0].onclick();
  assert.equal(calls[0].searchParams.get("visible"), "false");
  await context.maybeAutoRevealOwnAndAiHands(state);
  assert.equal(calls.length, 1, "automatic reveal must skip manually hidden hands");
  context.autoRevealHandsRoundKey = "";
  await context.maybeAutoRevealOwnAndAiHands({ ...state, auto_reveal_blocked_seats: [] });
  assert.equal(calls[1].searchParams.get("target"), "C");
  assert.equal(calls[1].searchParams.get("automatic"), "true");
  context.renderHandRevealControls({ ...state, finished: false, revealed_hand_seats: [] });
  assert.equal(panel.children[0].disabled, true);
  assert.equal(panel.children.length, 2, "public AI controls stay unavailable during play");
  console.log("Hand visibility UI: hide, labels, public restrictions and auto suppression passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
