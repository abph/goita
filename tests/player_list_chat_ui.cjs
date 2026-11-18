const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const people = [
  {name:'募集する人',location:'みんなでごいたA',role:'player',seat:'A',public_room_status:'recruiting'},
  {name:'観戦する人',location:'みんなでごいたA',role:'spectator',seat:''},
  {name:'ロビーの人',location:'トップページ',role:'lobby',seat:''},
  {name:'満席の人',location:'みんなでごいたB',role:'player',seat:'B',public_room_status:'full'},
  {name:'AI練習の人',location:'AIとごいたA',role:'player',seat:'C',tag:'ai_practice',public_room_status:'recruiting'},
  {name:'スコア挑戦',location:'スコアアタック',role:'player',seat:'A'},
  {name:'＊＊＊＊',location:'プライベートルーム',role:'player',seat:'D'},
];

(async()=>{
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const output=path.resolve('.server_deps_divergent/player-list-chat-ui');
  fs.mkdirSync(output,{recursive:true});
  try {
    const page=await browser.newPage({viewport:{width:1250,height:1000}});
    const errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.routeWebSocket('**/*',()=>{});
    await page.route('**/*',route=>{
      const url=new URL(route.request().url());
      if(url.hostname!=='goita.test') return route.abort();
      if(url.pathname==='/' || url.pathname.startsWith('/static/')) {
        const root=path.resolve(__dirname,'../frontend');
        const file=path.resolve(root,url.pathname==='/'?'index.html':url.pathname.slice(8));
        return file.startsWith(root+path.sep) && fs.existsSync(file)
          ?route.fulfill({path:file}):route.fulfill({status:404,body:''});
      }
      return route.fulfill({json:url.pathname==='/api/member/session'?{member:null}:{rooms:[],site_people:people,public_chat_messages:[]}});
    });
    await page.goto('http://goita.test/');
    await page.waitForFunction(()=>typeof personalSettings!=='undefined' && typeof chatPeopleShare!=='undefined');
    await page.evaluate(people=>{
      applyRoomChatPeopleSetting();renderLobbyPeople(people);
      toggleLobbyChatPanel(true,false);
    },people);
    const lobby=page.locator('#lobbyChatPanel');
    assert.match(await page.locator('#lobbyChatPeopleList').textContent(),/みんなでごいたA・在席中［募集中］/);
    assert.match(await page.locator('#lobbyChatPeopleList').textContent(),/みんなでごいたA・観戦中/);
    assert.match(await page.locator('#lobbyChatPeopleList').textContent(),/プライベートルーム・在席中/);
    assert.equal(await page.evaluate(()=>formatLobbyPersonLocation({location:'スコアアタック',seat:'A',role:'player'})),'スコアアタック');
    assert.equal(await page.locator('#lobbyChatPeopleList .lobby-recruiting-label').count(),2);
    const sizes=()=>lobby.evaluate(panel=>({
      total:panel.getBoundingClientRect().height,
      people:panel.querySelector('.room-chat-people').getBoundingClientRect().height,
      chat:panel.querySelector('.chat-content').getBoundingClientRect().height,
      form:panel.querySelector('.chat-form').getBoundingClientRect().toJSON(),
      bottom:panel.getBoundingClientRect().bottom,
    }));
    const before=await sizes();
    await lobby.locator('[data-people-adjust="up"]').click();
    const after=await sizes();
    assert.equal(before.total,after.total);
    assert.ok(after.people<before.people && after.chat>before.chat);
    assert.ok(after.form.bottom<=after.bottom);
    assert.equal(await page.evaluate(()=>localStorage.getItem(CHAT_PEOPLE_SHARE_KEY)),'40');
    await page.evaluate(()=>{for(let i=0;i<10;i++)adjustChatPeopleShare(-10);});
    assert.equal(await lobby.locator('[data-people-adjust="up"]').isDisabled(),true);
    assert.ok((await sizes()).people>0);
    await page.evaluate(()=>{for(let i=0;i<10;i++)adjustChatPeopleShare(10);});
    assert.equal(await lobby.locator('[data-people-adjust="down"]').isDisabled(),true);
    assert.ok((await sizes()).chat>0);
    await page.reload();
    await page.waitForFunction(()=>typeof chatPeopleShare!=='undefined' && chatPeopleShare===80);
    await page.evaluate(people=>{adjustChatPeopleShare(-30);renderLobbyPeople(people);toggleLobbyChatPanel(true,false);},people);
    for(const [language,status,recruiting] of [['en','Spectating','Open to players'],['zh','观战中','招募中']]) {
      await page.evaluate(({language,people})=>{goitaI18n.setLanguage(language,{dispatch:false});renderLobbyPeople(people);},{language,people});
      assert.match(await page.locator('#lobbyChatPeopleList').textContent(),new RegExp(status));
      assert.match(await page.locator('#lobbyChatPeopleList .lobby-recruiting-label').first().textContent(),new RegExp(recruiting));
      assert.equal(await lobby.locator('[data-people-adjust="up"]').getAttribute('title'),language==='en'?'Expand chat':'扩大聊天区域');
    }
    await page.evaluate(people=>{goitaI18n.setLanguage('ja',{dispatch:false});renderLobbyPeople(people);},people);
    await lobby.screenshot({path:path.join(output,'lobby-pc.png')});
    // The in-room panel uses the same divider and retains its overall height.
    await page.evaluate(people=>{
      toggleLobbyChatPanel(false);gid='main';mySeat=WATCH;
      document.getElementById('lobbyView').style.display='none';
      document.getElementById('gameView').style.display='block';
      renderRoomChatPeople(people);syncDesktopRoomChatHeight();
    },people);
    const room=page.locator('#chatPanel');
    const oldHeight=await room.evaluate(e=>e.getBoundingClientRect().height);
    await room.locator('[data-people-adjust="down"]').click();
    assert.equal(await room.evaluate(e=>e.getBoundingClientRect().height),oldHeight);
    await room.screenshot({path:path.join(output,'room-pc.png')});
    // Gathering uses its participant list above the same chat.
    await page.evaluate(()=>{
      document.body.classList.add('gathering-waiting-room');
      document.getElementById('gatheringRoomShell').hidden=false;
      document.getElementById('gatheringRoomPanel').hidden=false;
      document.getElementById('gatheringChatSlot').appendChild(document.getElementById('chatPanel'));
    });
    assert.equal(await room.locator('.chat-people-splitter').isVisible(),true);
    const gatheringBefore=await room.locator('.gathering-chat-players').evaluate(e=>e.getBoundingClientRect().height);
    await room.locator('[data-people-adjust="up"]').click();
    assert.ok(await room.locator('.gathering-chat-players').evaluate(e=>e.getBoundingClientRect().height)<gatheringBefore);
    await room.screenshot({path:path.join(output,'gathering-pc.png')});
    // Mobile keeps the existing left/right arrangement and no up/down divider.
    await page.evaluate(()=>{
      document.body.classList.remove('gathering-waiting-room');
      document.getElementById('gatheringRoomShell').hidden=true;
      document.querySelector('.play-layout').appendChild(document.getElementById('chatPanel'));
    });
    await page.setViewportSize({width:390,height:844});
    await page.evaluate(()=>toggleChatPanel(true,false));
    assert.equal(await room.locator('.chat-people-splitter').isVisible(),false);
    const mobile=await room.evaluate(panel=>({
      people:panel.querySelector('.room-chat-people').getBoundingClientRect().toJSON(),
      chat:panel.querySelector('.chat-content').getBoundingClientRect().toJSON(),
      panel:panel.getBoundingClientRect().toJSON(),
    }));
    assert.ok(mobile.people.right<=mobile.chat.left+1);
    assert.ok(mobile.panel.right<=391);
    await room.screenshot({path:path.join(output,'room-mobile.png')});
    assert.deepEqual(errors,[]);
    console.log('Player list and chat: role labels, recruiting badges, translations, height controls/limits/persistence, gathering and mobile layout passed');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
