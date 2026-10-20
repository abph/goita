const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Element {
  constructor() { this.style = {}; this.children = []; this.textContent = ''; this.hidden = false; }
  append(child) { this.children.push(child); }
  replaceChildren(...children) { this.children = children; }
  querySelector() { return new Element(); }
  focus() {}
}
const elements = new Map();
const element = id => {
  if (!elements.has(id)) elements.set(id, new Element());
  return elements.get(id);
};
const calls = [];
let response;
const context = vm.createContext({
  document: {getElementById:element, createElement:()=>new Element(), addEventListener() {}},
  window: {addEventListener() {}}, setTimeout, clearTimeout,
  gid:'score-test', mySeat:'W', clientId:'client', API:'',
  isScoreAttackRoom:()=>true, uiText:value=>value,
  syncScoreAttackMemberPrompt() {}, refresh:async()=>{}, setHint() {},
  fetch: async (url, init) => { calls.push({url,init}); return response; },
});
for (const file of ['kifu-files.js','trace-results.js']) {
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../frontend',file),'utf8'), context);
}

(async () => {
  // A seat update arriving after the menu opens must enable both start buttons.
  context.openDebugTrace();
  assert.equal(element('debugTraceDivergentButton').disabled, true);
  context.mySeat = 'A';
  context.syncDebugTraceMenu();
  assert.equal(element('debugTraceRandomButton').disabled, false);
  assert.equal(element('debugTraceDivergentButton').disabled, false);
  assert.equal(element('debugTraceStatus').textContent, '');

  let resolve;
  response = new Promise(done => {resolve=done;});
  const pending = context.startRandomDebugTrace('divergent');
  assert.equal(element('debugTraceRandomButton').disabled, true);
  assert.equal(element('debugTraceDivergentButton').disabled, true);
  await context.startRandomDebugTrace();
  assert.equal(calls.length, 1); // No duplicate starts across the two buttons.
  assert.equal(JSON.parse(calls[0].init.body).selection_mode, 'divergent');
  resolve({ok:false,json:async()=>({detail:'No candidates'})});
  await pending;
  assert.equal(element('debugTraceStatus').textContent, 'No candidates');
  assert.equal(element('debugTraceDivergentButton').disabled, false);
  assert.equal(element('debugTraceModal').style.display, 'flex');
  response = {ok:true,json:async()=>({ok:true})};
  await context.startRandomDebugTrace();
  assert.equal(JSON.parse(calls.at(-1).init.body).selection_mode, 'random');

  const result = {actual:{AC:20,BD:0},original:{AC:0,BD:0},improvement:20,ranking:[],
    comparison:{players:3,distribution:[{improvement:-50,players:2},{improvement:30,players:1}]}};
  response = {ok:true,json:async()=>result};
  await context.openTraceResult('attempt');
  assert.equal(element('traceComparison').hidden, false);
  assert.deepEqual(element('traceComparisonScores').children.map(node=>node.textContent), ['-50点 × 2人', '+30点 × 1人']);
  assert.equal(element('traceComparisonOwnScore').textContent, '+20点');
  result.comparison = null;
  await context.openTraceResult('normal-attempt');
  assert.equal(element('traceComparison').hidden, true); // Never retain stale comparisons.
  console.log('Divergent score attack UI behavior passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
