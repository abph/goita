const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const emptyBoard = () => Object.fromEntries([... 'ABCD'].map(s => [s, {
  receive: [null,null,null,null], attack: [null,null,null,null], receive_hidden: [false,false,false,false],
}]));
function fixture() {
  const board = emptyBoard();
  board.A.receive = ['1','3',null,null]; board.A.attack = ['1','4',null,null]; board.A.receive_hidden[0] = true;
  board.B.receive[0] = '4'; board.B.attack[0] = '5';
  board.C.receive[0] = '1'; board.C.attack[0] = '2';
  board.D.receive[0] = '2'; board.D.attack[0] = '3';
  return {is_started:true,finished:false,turn:null,phase:'receive',attacker:'B',current_attack:'5',dealer:'A',
    board_public:board,board_attack_numbers:{A:[1,4,null,null],B:[5,null,null,null],C:[2,null,null,null],D:[3,null,null,null]},
    hands:{A:['1','2','3','4','5','6','7','8'],B:{count:6},C:{count:6},D:{count:6}},
    init_hands:{},face_down_pieces:{A:['1']},player_names:{A:'A',B:'B',C:'C',D:'D'},
    human_seats:['A','B','C','D'],ai_seats:[],revealed_hand_seats:[],total_team_score:{AC:20,BD:30},
    round_count:1,log:Array(200).fill('B: pass'),chat_messages:[]};
}

(async () => {
  const browser = await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-swiftshader']});
  const output = path.resolve('.server_deps_divergent/board-numbers-ui');
  fs.mkdirSync(output,{recursive:true});
  try {
    const page = await browser.newPage({viewport:{width:1250,height:1000}});
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.routeWebSocket('**/*', () => {});
    const routeLocal = route => {
      const url = new URL(route.request().url());
      if(url.hostname !== 'goita.test') return route.abort();
      if(url.pathname === '/' || url.pathname.startsWith('/static/')) {
        const root = path.resolve(__dirname,'../frontend');
        const file = path.resolve(root,url.pathname==='/'?'index.html':url.pathname.slice(8));
        return file.startsWith(root+path.sep) && fs.existsSync(file)
          ? route.fulfill({path:file}) : route.fulfill({status:404,body:''});
      }
      if(url.pathname.endsWith('/state')) return route.fulfill({json:fixture()});
      if(url.pathname.endsWith('/legal_actions')) return route.fulfill({json:[]});
      return route.fulfill({json:url.pathname==='/api/member/session'?{member:null}:{rooms:[],site_people:[],public_chat_messages:[]}});
    };
    await page.route('**/*', routeLocal);
    await page.goto('http://goita.test/');
    await page.waitForFunction(() => typeof personalSettings !== 'undefined' && window.goitaBoardPixel);
    assert.equal(await page.evaluate(() => personalSettings.showBoardMoveNumbers),false);
    await page.evaluate(() => {openLobbySettings(); document.getElementById('lobbyPlaySupportSettingsDetails').open=true;});
    assert.equal(await page.locator('#lobbyCheckShowBoardMoveNumbers').isChecked(),false);
    await page.locator('#lobbyCheckShowBoardMoveNumbers').check();
    await page.evaluate(() => saveLobbyPersonalSettings());
    assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem(PERSONAL_SETTINGS_KEY)).showBoardMoveNumbers),true);
    await page.reload();
    await page.waitForFunction(() => typeof personalSettings !== 'undefined' && window.goitaBoardPixel);
    assert.equal(await page.evaluate(() => personalSettings.showBoardMoveNumbers),true);
    const other=await browser.newPage();
    await other.routeWebSocket('**/*',()=>{});
    await other.route('**/*',routeLocal);
    await other.goto('http://goita.test/');
    await other.waitForFunction(()=>typeof personalSettings!=='undefined');
    assert.equal(await other.evaluate(()=>personalSettings.showBoardMoveNumbers),false);
    await other.close();
    async function setup(snapshot=fixture(),seat='W') {
      await page.evaluate(({snapshot,seat}) => {
        cancelImmediateAction(); gid=activeRoomId=currentWsGid='room-gold-01';mySeat=seat;
        latestState=snapshot; latestLegal=[]; pending=null; fixedHandMemory={};
        personalSettings.enableEffects=false;personalSettings.enableSoundEffects=false;
        document.getElementById('lobbyView').style.display='none';
        document.getElementById('gameView').style.display='block';
        document.getElementById('lobbySettingsModal').style.display='none';
        renderBoard(snapshot);
      },{snapshot,seat});
    }
    await setup();
    const markers=page.locator('#board .board-move-number');
    assert.equal(await markers.count(),5);
    assert.deepEqual((await markers.allTextContents()).map(Number).sort((a,b)=>a-b),[1,2,3,4,5]);
    const before=await page.locator('#board').boundingBox();
    // Changing one's viewing seat rotates labels together with the pieces.
    for(const seat of ['A','B','C','D','W']) {
      await setup(fixture(),seat);
      const checks=await page.evaluate(() => [...document.querySelectorAll('#board .board-move-number')].map(e=>({
        expected:getSeatToPhys()[e.dataset.seat],actual:e.className.split('phys-')[1],text:e.textContent,
        label:e.getAttribute('aria-label'),pointer:getComputedStyle(e).pointerEvents,
      })));
      for(const c of checks){assert.equal(c.actual,c.expected);assert.match(c.label,/\d/);assert.equal(c.pointer,'none');}
    }
    // A receive-only state has no new completed attack number.
    const receiving=fixture();receiving.board_public.C.receive[1]='5';
    await setup(receiving);
    assert.equal(await markers.count(),5);
    const projection=await page.evaluate(snapshot=>{
      const after=projectImmediateAction(snapshot,'A',{action_type:'attack_after_block',block:'5',attack:'2'});
      return {after:after.board_attack_numbers.A[2],before:snapshot.board_attack_numbers.A[2]};
    },fixture());
    assert.deepEqual(projection,{after:6,before:null});
    // A late spectator needs no old log lines to reconstruct the same labels.
    await setup();
    assert.equal(await markers.count(),5);
    for(const width of [1250,390]) {
      await page.setViewportSize({width,height:1000});
      await page.locator('#board').screenshot({path:path.join(output,`board-${width}.png`)});
    }
    await page.setViewportSize({width:1250,height:1000});
    // Both canvas renderers receive exactly the same public numbering.
    for(const mode of ['pixel-color','pixel-mono','3d']) {
      await page.evaluate(mode=>{personalSettings.boardViewMode=mode;applyBoardViewMode();renderBoard(latestState);},mode);
      const numbers=await page.evaluate(()=>buildBoard3DSnapshot(latestState).pieces.filter(p=>p.moveNumber).map(p=>p.moveNumber).sort((a,b)=>a-b));
      assert.deepEqual(numbers,[1,2,3,4,5]);
      if(mode==='3d') assert.equal(await page.evaluate(()=>window.goitaBoard3D.isAvailable()),true);
      await page.waitForTimeout(350);
      await page.locator(mode==='3d'?'#board3d':'#boardPixel').screenshot({path:path.join(output,mode+'.png')});
    }
    await page.evaluate(()=>{
      personalSettings.boardViewMode='2d';applyBoardViewMode();
      syncPersonalSettingsControls();document.getElementById('checkShowBoardMoveNumbers').checked=false;
    });
    await page.evaluate(()=>savePersonalSettings());
    assert.equal(await markers.count(),0);
    assert.equal(await page.evaluate(()=>JSON.parse(localStorage.getItem(PERSONAL_SETTINGS_KEY)).showBoardMoveNumbers),false);
    const after=await page.locator('#board').boundingBox();
    assert.equal(after.width,before.width);assert.equal(after.height,before.height);
    await page.evaluate(()=>{personalSettings.showBoardMoveNumbers=true;renderBoard(latestState);});
    const next=fixture();next.round_count=2;next.board_public=emptyBoard();next.board_public.B.receive[0]='1';next.board_public.B.attack[0]='7';
    next.board_attack_numbers={A:[null,null,null,null],B:[1,null,null,null],C:[null,null,null,null],D:[null,null,null,null]};
    await setup(next);
    assert.equal(await markers.count(),1);assert.equal(await markers.textContent(),'1');
    for(const [language,expected] of [['en','Show move numbers on the board'],['zh','在棋盘上显示出牌顺序编号']]) {
      assert.equal(await page.evaluate(language=>{goitaI18n.setLanguage(language,{dispatch:false});return goitaI18n.translate('盤面に手順番号を表示する');},language),expected);
    }
    assert.deepEqual(errors,[]);
    console.log('Board move numbers UI: settings, persistence, spectators, rotations, receive/attack pairing, round reset, 2D/pixel/3D and translations passed');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
