const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../frontend/index.html'), 'utf8');
const playback = html.slice(html.indexOf('function researchKifuReplayFrames('), html.indexOf('function canShareResearchKifuNow('));
const exporter = html.slice(html.indexOf('function researchKifuYamlValue('), html.indexOf('function downloadSelectedResearchKifu('));

function harness(payload, {mode='recorded', speed='1'} = {}) {
  let now = 0;
  const timers = [];
  const board = [];
  const nodes = Object.fromEntries([
    'researchKifuReplayButton', 'researchKifuApplyButton', 'researchKifuReplayPause',
    'researchKifuReplayNext', 'researchKifuReplayMode', 'researchKifuReplaySpeed', 'researchKifuDetailStatus',
  ].map(id => [id, {textContent:'', value:'', disabled:false, classList:{toggle(){}}, querySelector(){ return this.option ||= {}; }}]));
  nodes.researchKifuReplayMode.value = mode;
  nodes.researchKifuReplaySpeed.value = speed;
  const context = vm.createContext({
    performance:{now:()=>now}, setTimeout:(fn, ms)=>timers.push({fn, due:now + ms}),
    document:{getElementById:id=>nodes[id]}, uiText:s=>s, currentUiLanguage:()=> 'ja',
    RESEARCH_KIFU_SEATS:['A','B','C','D'], researchKifuMoveLabel:row=>row.join(':'),
    renderResearchKifuBoard:(payload, options)=>board.push({now, options}),
  });
  vm.runInContext(`let selectedResearchKifu = ${JSON.stringify({payload})};
    let researchKifuReplayToken=0, researchKifuReplayRunning=false,
      researchKifuReplayPaused=false, researchKifuReplayAdvance=false;`, context);
  vm.runInContext(exporter + playback, context);
  async function microtasks() { for (let i=0;i<10;i++) await Promise.resolve(); }
  return {context, nodes, board,
    async advance(ms) {
      const target = now + ms;
      await microtasks();
      while (true) {
        timers.sort((a,b)=>a.due-b.due);
        if (!timers.length || timers[0].due > target) break;
        const timer=timers.shift(); now=timer.due; timer.fn(); await microtasks();
      }
      now=target; await microtasks();
    },
    frames:()=>JSON.parse(JSON.stringify(context.researchKifuReplayFrames(payload))),
    running:()=>vm.runInContext('researchKifuReplayRunning', context),
  };
}

const payload = {moves:[['0','し','銀'],['1','パス',''],['2','銀',''],['2','','金']],
  game:[['0','し','銀'],['2','銀','金']], move_times:[2.3,30,45,1]};

(async () => {
  const h = harness(payload);
  assert.deepEqual(h.frames().map(f=>f.recordedWait), [2.3,0,30,45,1]);
  assert.deepEqual(h.frames()[2].latestAction, {seat:'B',kind:'pass'});
  for (let seat=0; seat<4; seat++) {
    const rows=[[String(seat),'し','銀'],[String(seat),'馬','金'],[String(seat),'パス',''],[String((seat+1)%4),'パス','']];
    const frames=h.context.researchKifuReplayFrames({moves:rows});
    assert.deepEqual(JSON.parse(JSON.stringify(frames[4].latestAction)),
      {seat:'ABCD'[seat],kind:'pass'}, 'pass identifies the player independently of their placed pieces');
    assert.equal(frames[5].latestAction.seat,'ABCD'[(seat+1)%4], 'consecutive passes replace the previous pass marker');
  }
  assert.match(h.context.researchKifuDownloadText({payload}), /move_times: \[2.3,30,45,1\]/);
  assert.match(h.context.researchKifuDownloadText({payload}), /"1","パス",""/);
  const run = h.context.toggleResearchKifuReplay();
  await h.advance(2250);
  assert.equal(h.board.length,1,'no piece is shown before its recorded action time');
  assert.equal(h.board[0].options.hideHands,true,'unused hands are hidden from the first replay frame');
  await h.advance(50);
  assert.equal(h.board.length,3,'combined block and attack are simultaneous');
  assert.equal(h.board.at(-1).now,2300);
  await h.advance(1000);
  h.context.pauseResearchKifuReplay();
  await h.advance(40000);
  assert.equal(h.board.length,3,'pause preserves the current board even during long waits');
  h.context.advanceResearchKifuReplay();
  await h.advance(50);
  assert.equal(h.board.length,4,'next advances only one action while paused');
  await h.advance(40000);
  assert.equal(h.board.length,4);
  h.context.pauseResearchKifuReplay();
  await h.advance(44900);
  assert.equal(h.board.length,4);
  await h.advance(100);
  assert.equal(h.board.length,5,'the 45-second interval belongs before the receive');
  await h.advance(2000);
  await run;
  assert.ok(h.board.slice(0,-1).every(frame=>frame.options.hideHands===true),'unused hands stay hidden throughout playback');
  assert.equal(h.board.at(-1).options,undefined,'normal final board is restored when playback finishes');
  assert.equal(h.running(),false);
  assert.equal(h.nodes.researchKifuReplayPause.disabled,true);

  const faster=harness(payload,{speed:'2'});
  const fastRun=faster.context.toggleResearchKifuReplay();
  await faster.advance(1100); assert.equal(faster.board.length,1);
  await faster.advance(50); assert.equal(faster.board.length,3);
  await faster.advance(15000); assert.equal(faster.board.length,4);
  faster.context.stopResearchKifuReplay();
  const count=faster.board.length;
  await faster.advance(100000); await fastRun;
  assert.equal(faster.board.length,count,'stopped replay cannot render stale frames');

  const legacy={moves:payload.moves,game:payload.game};
  const old=harness(legacy);
  old.context.updateResearchKifuReplayControls();
  assert.equal(old.nodes.researchKifuReplayMode.value,'normal');
  assert.equal(old.nodes.researchKifuReplayMode.disabled,true);
  assert.ok(old.frames().every(frame=>frame.recordedWait===null));
  assert.doesNotMatch(old.context.researchKifuDownloadText({payload:legacy}),/move_times:/);
  const oldRun=old.context.toggleResearchKifuReplay();
  await old.advance(600); assert.equal(old.board.length,1);
  await old.advance(50); assert.equal(old.board.length,2);
  await old.advance(5000); await oldRun;
  assert.equal(old.running(),false);

  for(const times of [[1],[-1,0,0,0],[NaN,0,0,0],['2',0,0,0],[null,null,null,null]]) {
    assert.equal(h.context.researchKifuMoveTimes({...payload,move_times:times}),null);
  }
  for (const script of html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)) new vm.Script(script[1]);
  console.log('Kifu timing UI: interval alignment, pauses, skip, speed, cancellation, old records and export passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
