const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../frontend/soloDiscord.js'), 'utf8');
const SHOWN_DAY_KEY = 'goita-solo-discord-prompt-day';

function harness({saved = new Map(), blocked = false} = {}) {
  const nodes = Object.fromEntries(['soloDiscordPrompt', 'soloDiscordOpen', 'soloDiscordDismiss'].map(id => [id, {
    hidden:true, events:{}, addEventListener(name, callback) { this.events[name] = callback; },
  }]));
  const documentEvents = {};
  const windowEvents = {};
  const metrics = [];
  const opened = [];
  let now = Date.parse('2026-10-20T14:59:00Z');
  class ClockDate extends Date { static now() { return now; } }
  const context = vm.createContext({
    Date:ClockDate,
    localStorage:{
      getItem(key) { if (blocked) throw new Error('disabled'); return saved.get(key) || null; },
      setItem(key, value) { if (blocked) throw new Error('disabled'); saved.set(key, String(value)); },
    },
    document:{
      hidden:false,
      getElementById:id => nodes[id],
      addEventListener(name, callback) {
        if (name === 'DOMContentLoaded') callback(); else documentEvents[name] = callback;
      },
    },
    window:{
      addEventListener:(name, callback) => { windowEvents[name] = callback; },
      trackAnalytics:(event, properties, roomType) => metrics.push({event, properties, roomType}),
      openSiteInfo:key => opened.push(key),
    },
  });
  vm.runInContext(source, context);
  return {
    api:context.window.goitaSoloDiscord, nodes, metrics, saved, opened, documentEvents, windowEvents,
    setHidden(value) { context.document.hidden = value; },
    setTime(value) { now = Date.parse(value); },
    click(id) { nodes[id].events.click(); },
  };
}

const state = changes => ({
  is_started:true, finished:true, round_count:2,
  human_seats:['A'], ai_seats:['B','C','D'], owned_human_seats:['A'], ...changes,
});
const context = changes => ({roomId:'main-e', roomType:'public_ai', seat:'A', normalRoom:true, ...changes});

// Only the sole human player in a normal game sees the invitation after round two.
for (const [changes, room] of [
  [{finished:false}, {}], [{round_count:1}, {}], [{is_started:false}, {}],
  [{owned_human_seats:[]}, {}], [{}, {seat:'WATCH'}],
  [{human_seats:['A','C'],ai_seats:['B','D']}, {}],
  [{ai_seats:['A','B','C']}, {}], [{ai_seats:['B','C','C']}, {}],
  [{trace_mode:true}, {}], [{tutorial_mode:true}, {}], [{practice_replay_active:true}, {}],
  [{}, {roomType:'none'}], [{}, {roomType:'score_attack'}], [{}, {normalRoom:false}],
]) {
  const h = harness();
  h.api.sync(state(changes), context(room));
  assert.equal(h.nodes.soloDiscordPrompt.hidden, true);
  assert.equal(h.metrics.length, 0);
  assert.equal(h.saved.size, 0, 'ineligible games do not consume the daily invitation');
}

const h = harness();
h.api.sync(state(), context());
assert.equal(h.nodes.soloDiscordPrompt.hidden, false);
assert.equal(h.saved.get(SHOWN_DAY_KEY), '2026-10-20');
assert.equal(h.metrics.length, 1);
assert.equal(h.metrics[0].event, 'discord_solo_prompt_shown');
for (let i = 0; i < 5; i++) h.api.sync(state(), context());
assert.equal(h.nodes.soloDiscordPrompt.hidden, false, 'refreshes keep the same invitation visible');
assert.equal(h.metrics.length, 1);
h.click('soloDiscordOpen');
assert.deepEqual(h.opened, ['discord']);
assert.deepEqual(h.metrics.map(item => item.event), [
  'discord_solo_prompt_shown', 'discord_solo_prompt_clicked', 'discord_info_opened',
]);
assert.ok(h.metrics.every(item => item.roomType === 'public_ai'));
assert.ok(h.metrics.every(item => Object.keys(item.properties).length === 0), 'no gameplay or player details are sent');
assert.equal(h.nodes.soloDiscordPrompt.hidden, true);
h.api.sync(state({round_count:3}), context());
assert.equal(h.nodes.soloDiscordPrompt.hidden, true, 'clicking the invitation suppresses further invitations today');

const reloaded = harness({saved:h.saved});
reloaded.api.sync(state(), context({roomId:'private-a',roomType:'private'}));
assert.equal(reloaded.nodes.soloDiscordPrompt.hidden, true, 'daily limit survives reloads and switching rooms');
reloaded.setTime('2026-10-20T15:00:00Z');
reloaded.api.sync(state(), context({roomId:'private-a',roomType:'private'}));
assert.equal(reloaded.nodes.soloDiscordPrompt.hidden, false, 'a new Japanese calendar day permits one invitation');
reloaded.click('soloDiscordDismiss');
reloaded.api.sync(state(), context({roomId:'private-a',roomType:'private'}));
assert.equal(reloaded.nodes.soloDiscordPrompt.hidden, true);

const background = harness();
background.setHidden(true);
background.api.sync(state(), context());
assert.equal(background.saved.size, 0, 'background tabs do not consume the invitation');
background.setHidden(false);
background.documentEvents.visibilitychange();
assert.equal(background.nodes.soloDiscordPrompt.hidden, false);
background.api.sync(state({finished:false,round_count:3}), context());
assert.equal(background.nodes.soloDiscordPrompt.hidden, true, 'starting the next round hides the invitation');

const navigation = harness();
navigation.api.sync(state(), context());
navigation.api.reset();
assert.equal(navigation.nodes.soloDiscordPrompt.hidden, true);
navigation.documentEvents.visibilitychange();
assert.equal(navigation.nodes.soloDiscordPrompt.hidden, true, 'returning to the lobby clears the previous game context');

const otherTab = harness();
otherTab.api.sync(state(), context());
otherTab.windowEvents.storage({key:SHOWN_DAY_KEY,newValue:'2026-10-20'});
assert.equal(otherTab.nodes.soloDiscordPrompt.hidden, true);

const withoutStorage = harness({blocked:true});
withoutStorage.api.sync(state(), context());
assert.equal(withoutStorage.nodes.soloDiscordPrompt.hidden, false);
withoutStorage.click('soloDiscordDismiss');
withoutStorage.api.sync(state({round_count:3}), context());
assert.equal(withoutStorage.nodes.soloDiscordPrompt.hidden, true, 'memory still suppresses repeats when storage is unavailable');

console.log('Solo Discord: eligibility, daily limit, dismissal, navigation, visibility, and click tracking passed');
