const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const elements = {};
function element() {
  const classes = new Set();
  return {hidden: false, textContent: '', attrs: {}, events: {},
    classList: {add: (...v) => v.forEach(c => classes.add(c)), remove: (...v) => v.forEach(c => classes.delete(c)), toggle: (c, on) => on ? classes.add(c) : classes.delete(c)},
    setAttribute(name, value) {this.attrs[name] = value;}, removeAttribute(name) {delete this.attrs[name];},
    addEventListener(name, listener) {this.events[name] = listener;},
  };
}
for (const id of ['lobbyWhisper', 'lobbyWhisperMessage', 'lobbyWhisperClose', 'label']) elements[id] = element();
let sequence = 0;
const timers = new Map();
const opened = [];
const context = vm.createContext({
  document: {readyState: 'complete', getElementById: id => elements[id], querySelector: () => elements.label},
  setTimeout: fn => {const id = ++sequence; timers.set(id, fn); return id;}, clearTimeout: id => timers.delete(id),
  open: (...args) => opened.push(args),
});
context.window = context;
vm.runInContext(fs.readFileSync(path.join(__dirname, '../frontend/lobbyWhisper.js'), 'utf8'), context);
const api = context.goitaLobbyWhisper;
api.setRoomVisibility(true);
assert.equal(elements.lobbyWhisper.hidden, true); // no special-message flash before settings arrive
const custom = {enabled: true, mode: 'custom', room_id: 'main', label: '大会', message: '<b>来週開催</b>', url: 'https://example.com/event'};
api.setRoomContext(true, null, custom);
assert.equal(elements.lobbyWhisper.hidden, false);
assert.equal(elements.label.textContent, '大会');
assert.equal(elements.lobbyWhisperMessage.textContent, '<b>来週開催</b>');
assert.equal(elements.lobbyWhisperMessage.attrs['data-i18n-ignore'], '');
elements.lobbyWhisper.events.click();
assert.deepEqual(opened[0], ['https://example.com/event', '_blank', 'noopener,noreferrer']);
api.dismiss();
api.setRoomContext(true, null, custom);
assert.equal(elements.lobbyWhisper.hidden, true);
api.setRoomContext(true, null, {...custom, message: '変更した文章', url: ''});
assert.equal(elements.lobbyWhisper.hidden, false);
elements.lobbyWhisper.events.click();
assert.equal(elements.lobbyWhisperMessage.textContent, '変更した文章'); // custom mode has no surprise behavior
api.setRoomContext(true, null, {enabled: true, mode: 'whisper', room_id: 'main'});
assert.equal(elements.label.textContent, '1222のつぶやき');
assert.equal(elements.lobbyWhisperMessage.textContent, 'こんにちは！');
for (let i = 0; i < 100; i++) {
  elements.lobbyWhisper.events.click();
  const callbacks = [...timers.values()]; timers.clear(); callbacks.forEach(fn => fn());
}
assert.match(elements.lobbyWhisperMessage.textContent, /100回目おめでとう/);
api.setRoomContext(true, null, {enabled: false, mode: 'whisper'});
assert.equal(elements.lobbyWhisper.hidden, true);
api.setRoomContext(false, {enabled: true, label: '個室', message: '従来のお知らせ', url: ''});
assert.equal(elements.label.textContent, '個室');
assert.equal(elements.lobbyWhisperMessage.textContent, '従来のお知らせ');
console.log('Notices UI: custom/link, dismissal, updates, legacy 100-click behavior, disabled and private notices passed');
