const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
(async () => {
  const browser = await chromium.launch({channel:'msedge',headless:true});
  try {
    const page = await browser.newPage({viewport:{width:1100,height:1000}});
    const errors=[], requests=[];
    page.on('pageerror',error=>errors.push(error.message));
    const rounds=Array.from({length:10},(_,index)=>({round_index:index+1,winner:'B',gained_score:20,
      hand:{p0:'しし王金金銀飛香',p1:'しし玉銀銀銀香馬',p2:'しししし金飛香香',p3:'しし角角金馬馬馬'},
      player_names:{A:'Sample A',B:'Sample B',C:'Sample C',D:'Sample D'},
      game:[['1','馬','銀'],['0','銀','金']],score_before:{AC:0,BD:0},score_after:{AC:0,BD:20}}));
    let hold=false, held;
    await page.routeWebSocket('**/*',socket=>socket.close());
    await page.route('**/*',route=>{
      const url=new URL(route.request().url());
      if(url.hostname!=='goita.test') return route.abort();
      if(url.pathname==='/' || url.pathname.startsWith('/static/')) {
        const root=path.resolve(__dirname,'../frontend');
        const file=path.resolve(root,url.pathname==='/'?'index.html':url.pathname.slice(8));
        return file.startsWith(root+path.sep)&&fs.existsSync(file)?route.fulfill({path:file}):route.fulfill({status:404,body:''});
      }
      requests.push(url.pathname+url.search);
      if(url.pathname==='/kifu/preview') {
        if(hold) {held=route; return;}
        const text=route.request().postDataJSON().kifu_text;
        return text==='invalid'?route.fulfill({status:400,json:{detail:'形式が違います'}}):route.fulfill({json:{rounds}});
      }
      if(url.pathname.endsWith('/kifu')) return route.fulfill({body:'version: 1.0\n',contentType:'text/plain'});
      return route.fulfill({json:url.pathname==='/api/member/session'?{member:null}:{rooms:[],site_people:[],public_chat_messages:[]}});
    });
    await page.goto('http://goita.test/');
    assert.equal(await page.locator('#debugTraceMenuItem').evaluate(element=>getComputedStyle(element).display),'none');
    await page.evaluate(()=>{gid='debug'; mySeat='A'; syncDebugTraceMenu();});
    assert.notEqual(await page.locator('#debugTraceMenuItem').evaluate(element=>getComputedStyle(element).display),'none');
    await page.evaluate(()=>openDebugTrace());
    assert.equal(await page.locator('#debugTraceModal').isVisible(),true);
    assert.equal(await page.locator('#debugTraceRandomButton').isDisabled(),false);
    await page.locator('#debugTraceInput').setInputFiles(path.join(__dirname,'fixtures/external_match.yaml'));
    await page.waitForFunction(()=>document.getElementById('debugTraceRound').options.length===10);
    assert.equal(await page.locator('#debugTraceStartButton').isDisabled(),false);
    await page.evaluate(()=>closeDebugTrace());
    assert.equal(await page.locator('#debugTraceModal').isVisible(),false);
    await page.locator('[data-header-menu-toggle]').first().click();
    await page.getByRole('button',{name:'棋譜ファイル',exact:true}).first().click();
    assert.equal(await page.locator('#kifuFileSave').isDisabled(),true);
    await page.locator('#kifuFileInput').setInputFiles(path.join(__dirname,'fixtures/external_match.yaml'));
    await page.waitForFunction(()=>document.getElementById('kifuFileRound').options.length===10);
    assert.equal(await page.locator('#kifuFileBoard .research-kifu-seat-label').count(),4);
    assert.match(await page.locator('#kifuFileMove').textContent(),/最初の配牌/);
    await page.locator('#kifuFileRound').selectOption('9');
    assert.match(await page.locator('#kifuFileBoard').textContent(),/第10局/);
    await page.getByRole('button',{name:'進む',exact:true}).click();
    assert.match(await page.locator('#kifuFileMove').textContent(),/^1 /);
    await page.locator('#kifuFilePlay').click();
    assert.equal(await page.locator('#kifuFilePlay').textContent(),'停止');
    await page.locator('#kifuFilePlay').click();
    await page.getByRole('button',{name:'最後',exact:true}).click();
    assert.match(await page.locator('#kifuFileBoard').textContent(),/BD 20/);
    await page.setViewportSize({width:390,height:844});
    await page.locator('#kifuFileRound').selectOption('0');
    await page.locator('#kifuFilesModal .modal-content').screenshot({path:path.resolve(__dirname,'../results/kifu-files-mobile.png')});
    await page.locator('#kifuFileInput').setInputFiles({name:'invalid.yaml',mimeType:'text/plain',buffer:Buffer.from('invalid')});
    await page.waitForFunction(()=>document.getElementById('kifuFileStatus').textContent==='形式が違います');
    hold=true;
    await page.locator('#kifuFileInput').setInputFiles({name:'pending.yaml',mimeType:'text/plain',buffer:Buffer.from('pending')});
    for(let i=0;!held&&i<100;i++) await new Promise(resolve=>setTimeout(resolve,10));
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('#kifuFilesModal').isVisible(),false);
    assert.equal(await page.evaluate(()=>kifuFileRounds.length),0);
    assert.equal(requests.filter(url=>/\/api\/member\/kifu\/(save|import)/.test(url)).length,0);
    await page.evaluate(()=>{activeRoomId='file-test';gid='file-test';latestState={finished:true};openKifuFiles();});
    for(const selector of ['#kifuFileSave','#kifuFileSaveAnonymous']) {
      const download=page.waitForEvent('download');
      await page.locator(selector).click();
      assert.match((await download).suggestedFilename(),/\.yaml$/);
    }
    assert.ok(requests.some(url=>url.includes('/games/file-test/kifu?anonymous=false')));
    assert.ok(requests.some(url=>url.includes('/games/file-test/kifu?anonymous=true')));
    assert.deepEqual(errors,[]);
    console.log('File dialog: guest preview, ten rounds, replay, mobile, invalid/stale files, no save requests, both downloads passed.');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
