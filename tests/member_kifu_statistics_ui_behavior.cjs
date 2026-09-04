const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(require('node:path').join(__dirname, '../frontend/index.html'), 'utf8');
const source = html.slice(html.indexOf('let researchKifuStatisticsRevision ='), html.indexOf('function researchKifuSeatLabel('));
class Element {
  constructor() { this.children = []; this.hidden = true; this.disabled = false; this.textContent = ''; }
  append(...children) { this.children.push(...children); }
  replaceChildren() { this.children = []; this.textContent = ''; }
}
(async () => {
  const elements = {researchKifuStatistics: new Element(), researchKifuStatisticsButton: new Element()};
  let calls = 0;
  const stats = {total: 8, counted: 5, wins: 3, losses: 2, win_rate: 60,
    points_for: 90, points_against: 50, point_difference: 40, self_finishes: 1,
    partner_finishes: 2, unset: 1, spectator: 1, incomplete: 1};
  const context = vm.createContext({
    document: {getElementById: id => elements[id], createElement: () => new Element()},
    uiText: text => text,
    researchKifuApi: async (path, body) => {
      calls++;
      assert.equal(path, '/statistics');
      assert.equal(body, undefined); // No tag filter or user-supplied owner.
      return {statistics: stats};
    },
  });
  vm.runInContext(source, context);
  assert.equal(calls, 0); // Only button action requests statistics.
  const pending = context.createResearchKifuStatistics();
  assert.equal(elements.researchKifuStatisticsButton.disabled, true);
  await pending;
  assert.equal(elements.researchKifuStatisticsButton.disabled, false);
  assert.equal(elements.researchKifuStatistics.hidden, false);
  const values = elements.researchKifuStatistics.children[1].children.map(e => e.textContent);
  assert.equal(values[9], '60.0%');
  assert.equal(values[1], '8');
  stats.win_rate = null;
  await context.createResearchKifuStatistics();
  assert.equal(elements.researchKifuStatistics.children[1].children[9].textContent, '-');
  let finish;
  context.researchKifuApi = () => new Promise(resolve => {finish = resolve;});
  const stale = context.createResearchKifuStatistics();
  context.invalidateResearchKifuStatistics(); // Edit, reload or account change.
  finish({statistics: stats});
  await stale;
  assert.equal(elements.researchKifuStatistics.hidden, true);
  assert.equal(elements.researchKifuStatistics.children.length, 0);
  context.researchKifuApi = async () => {throw new Error('offline');};
  await context.createResearchKifuStatistics();
  assert.equal(elements.researchKifuStatistics.textContent, 'offline');
  assert.equal(elements.researchKifuStatisticsButton.disabled, false);
  for (const name of ['saveCurrentResearchKifu', 'importResearchKifuFile', 'saveResearchKifuEdit']) {
    const start = html.indexOf('async function ' + name + '(');
    const next = html.indexOf('\nfunction ', start);
    assert.match(html.slice(start, next), /my_seat: document.getElementById\("researchKifuMySeat(?:Edit)?"\).value/);
  }
  console.log('Statistics UI: on-demand rendering, zero sample, invalidation, error recovery and seat inputs passed');
})().catch(error => {console.error(error); process.exitCode = 1;});
