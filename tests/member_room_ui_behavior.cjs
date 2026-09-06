// Run with Playwright installed and Microsoft Edge available.
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1100, height: 900}});
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    let member = {member_id: 'research-user', paid_active: true, paid_enabled: true, research_enabled: true,
      managed_room_id: 'room-gold-01', must_change_password: false, created_at: 1788624000};
    let room = {game_id: 'room-gold-01', owner_name: '研究用の部屋', is_private: true, ai_profile: 'current',
      ai_profiles: {current: '強化中AI', beginner_upper: '初級者（上）'}, show_log: false, show_legal_actions: false,
      managed_human_seats: [{seat: 'B', name: '研究仲間', occupancy_token: 'seat-token'}]};
    let failSave = false, pendingRead = null, holdRead = false, lastBody;
    await page.routeWebSocket('**/*', socket => socket.close());
    await page.route('**/*', async route => {
      const url = new URL(route.request().url());
      if (url.hostname !== 'goita.test') return route.abort();
      if (url.pathname === '/' || url.pathname.startsWith('/static/')) {
        const root = path.resolve(__dirname, '../frontend');
        const file = path.resolve(root, url.pathname === '/' ? 'index.html' : url.pathname.slice(8));
        return file.startsWith(root + path.sep) && fs.existsSync(file)
          ? route.fulfill({path: file}) : route.fulfill({status: 404, body: ''});
      }
      let data = {};
      if (url.pathname === '/games/list') data = {rooms: [], site_people: [], public_chat_messages: []};
      if (url.pathname === '/api/member/session') data = {member};
      if (url.pathname === '/api/member/logout') {member = null; data = {ok: true};}
      if (url.pathname === '/api/member/kifu/list') data = {records: []};
      if (url.pathname.startsWith('/api/member/room')) {
        assert.equal(route.request().headers()['x-goita-member'], '1');
        if (url.pathname === '/api/member/room' && holdRead) {
          pendingRead = route;
          return;
        }
        if (url.pathname.endsWith('/settings')) {
          lastBody = route.request().postDataJSON();
          if (failSave) return route.fulfill({status: 500, json: {detail: '保存テストエラー'}});
          room = {...room, owner_name: lastBody.new_owner_name, ai_profile: lastBody.ai_profile,
            show_log: lastBody.show_log, show_legal_actions: lastBody.show_legal_actions,
            is_private: lastBody.update_password ? !!lastBody.new_password : room.is_private};
        }
        if (url.pathname.endsWith('/vacate')) {
          assert.equal(route.request().postDataJSON().occupancy_token, 'seat-token');
          room = {...room, managed_human_seats: []};
        }
        data = {room};
      }
      return route.fulfill({json: data});
    });
    await page.goto('http://goita.test/');
    await page.waitForFunction(() => window.goitaMembers?.canUseAllStamps());
    await page.evaluate(() => {closeChatPanel('lobby'); openMemberPage();});
    const lobby = page.locator('#lobbyMemberPanel');
    await lobby.getByRole('tab', {name: 'ルーム管理', exact: true}).click();
    await lobby.locator('[name="new_owner_name"]').waitFor();
    assert.equal(await lobby.locator('[name="new_owner_name"]').inputValue(), room.owner_name);
    await page.evaluate(() => {window.roomEntries = []; tryJoinRoom = async (...args) => window.roomEntries.push(args);});
    await lobby.locator('[data-room-enter]').click();
    assert.deepEqual(await page.evaluate(() => window.roomEntries), [['room-gold-01', true]]);
    await page.evaluate(() => openMemberPage());
    await lobby.locator('[name="new_owner_name"]').waitFor();
    await lobby.locator('[name="new_owner_name"]').fill('変更後の研究室');
    await lobby.locator('[name="update_password"]').check();
    await lobby.locator('[data-room-settings] [name="new_password"]').fill('new-pass');
    await lobby.locator('[name="ai_profile"]').selectOption('beginner_upper');
    await lobby.locator('[name="show_log"]').check();
    await lobby.getByRole('button', {name: 'ルーム管理を保存', exact: true}).click();
    await lobby.getByText('ルーム設定を保存しました。', {exact: true}).waitFor();
    assert.equal(lastBody.game_id, 'room-gold-01');
    assert.equal(lastBody.new_owner_name, '変更後の研究室');
    assert.equal(lastBody.new_password, 'new-pass');
    assert.equal(lastBody.admin_password, undefined);
    assert.equal(await lobby.locator('[data-room-settings] [name="new_password"]').inputValue(), '');
    failSave = true;
    await lobby.locator('[name="new_owner_name"]').fill('未保存の入力');
    await lobby.getByRole('button', {name: 'ルーム管理を保存', exact: true}).click();
    await lobby.getByText('保存テストエラー', {exact: true}).waitFor();
    assert.equal(await lobby.locator('[name="new_owner_name"]').inputValue(), '未保存の入力');
    failSave = false;
    await lobby.locator('[data-room-refresh]').click();
    await page.waitForFunction(() => document.querySelector('#lobbyMemberPanel [name="new_owner_name"]').value === '変更後の研究室');
    await lobby.locator('.member-room-seats summary').click();
    page.once('dialog', dialog => dialog.accept());
    await lobby.getByRole('button', {name: '席を空ける', exact: true}).click();
    await lobby.getByText('席を空けました。', {exact: true}).waitFor();
    await page.setViewportSize({width: 390, height: 844});
    assert.equal(await lobby.evaluate(e => e.scrollWidth <= e.clientWidth), true);
    fs.mkdirSync(path.resolve(__dirname, '../results'), {recursive: true});
    await page.screenshot({path: path.resolve(__dirname, '../results/member-room-mobile.png')});
    // Reopening refreshes room settings, and moving to the room modal keeps the selected tab.
    await page.evaluate(() => closeLobbySettings());
    room.owner_name = '別端末での変更';
    await page.evaluate(() => openMemberPage());
    await page.waitForFunction(() => document.querySelector('#lobbyMemberPanel [name="new_owner_name"]')?.value === '別端末での変更');
    await page.evaluate(() => {closeLobbySettings(); document.getElementById('gameView').style.display = 'block'; gid = 'room-gold-01'; openMemberPage();});
    const inRoom = page.locator('#memberSettingsPanel');
    await inRoom.locator('[name="new_owner_name"]').waitFor();
    await inRoom.locator('[data-room-enter]').click();
    assert.equal(await page.evaluate(() => window.roomEntries.length), 1, 'entering the current room must not release the seat');
    await page.evaluate(() => openMemberPage());
    await inRoom.locator('[name="new_owner_name"]').waitFor();
    await inRoom.getByRole('tab', {name: '棋譜ライブラリ', exact: true}).click();
    assert.equal(await inRoom.locator('[data-member-room]').isVisible(), false);
    await inRoom.getByRole('tab', {name: 'ルーム管理', exact: true}).click();
    await inRoom.locator('[name="new_owner_name"]').waitFor();
    await inRoom.getByRole('tab', {name: 'ルーム管理', exact: true}).press('Home');
    assert.equal(await inRoom.locator('[data-member-account]').isVisible(), true);
    member = {...member, paid_active: false};
    await page.evaluate(() => goitaMembers.refresh());
    await inRoom.getByRole('tab', {name: 'ルーム管理', exact: true}).click();
    await inRoom.getByText('ルーム管理には有効な研究用プランが必要です。', {exact: true}).waitFor();
    member = {...member, paid_active: true, managed_room_id: ''};
    await page.evaluate(() => goitaMembers.refresh());
    await inRoom.getByText('管理する部屋が割り当てられていません。', {exact: true}).waitFor();
    member = {...member, managed_room_id: 'room-gold-01'};
    holdRead = true;
    await page.evaluate(() => goitaMembers.refresh());
    await inRoom.getByRole('tab', {name: 'アカウント', exact: true}).click();
    await inRoom.getByRole('button', {name: 'ログアウト', exact: true}).click();
    await inRoom.getByText('会員ログイン', {exact: true}).waitFor();
    assert.ok(pendingRead);
    await pendingRead.fulfill({json: {room}});
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    assert.equal(await inRoom.locator('[data-room-settings]').count(), 0);
    assert.equal(await inRoom.getByRole('tab', {name: 'ルーム管理', exact: true}).count(), 0);
    assert.deepEqual(errors, []);
    console.log('Research room UI: saving, failure recovery, seat management, mobile, reopen, both modals, keyboard, expiry and stale logout responses passed.');

    const admin = await browser.newPage({viewport: {width: 390, height: 844}});
    const adminErrors = [];
    admin.on('pageerror', e => adminErrors.push(e.message));
    let members = [{member_id: 'owner', enabled: true, paid_enabled: true, paid_active: true,
      research_enabled: true, managed_room_id: 'room-gold-01'},
      {member_id: 'normal', enabled: true, paid_enabled: true, paid_active: true}];
    const rooms = [{game_id: 'room-gold-01', name: '研究室A'}, {game_id: 'room-silver-02', name: '研究室B'}];
    let adminBody;
    await admin.route('**/*', async route => {
      const url = new URL(route.request().url());
      if (url.hostname !== 'goita.test') return route.abort();
      if (url.pathname === '/admin') {
        const html = fs.readFileSync(path.resolve(__dirname, '../frontend/admin.html'), 'utf8').replace(/<script\b[^>]*>[\s\S]*?<\/script>/g, '');
        return route.fulfill({contentType: 'text/html', body: html});
      }
      let data = {members, rooms, persistent: true};
      if (route.request().method() === 'POST') {
        adminBody = route.request().postDataJSON();
        members.push({...adminBody, enabled: true, paid_active: true});
        data = {member: adminBody, temporary_password: 'test-password', temporary_expires_at: 1800000000};
      }
      if (route.request().method() === 'PUT') {
        adminBody = route.request().postDataJSON();
        const id = decodeURIComponent(url.pathname.split('/').pop());
        members = members.map(m => m.member_id === id ? {...m, ...adminBody} : m);
        data = {member: members.find(m => m.member_id === id)};
      }
      return route.fulfill({json: data});
    });
    await admin.goto('http://goita.test/admin');
    await admin.addScriptTag({content: fs.readFileSync(path.resolve(__dirname, '../frontend/admin-members.js'), 'utf8')});
    await admin.evaluate(async () => {
      document.getElementById('authView').hidden = true;
      document.getElementById('adminView').hidden = false;
      document.getElementById('membersView').classList.add('active');
      await goitaMemberAdmin.load();
    });
    const create = admin.locator('#memberCreateForm');
    assert.equal(await create.locator('[name="managed_room_id"]').isDisabled(), true);
    await create.locator('[name="member_id"]').fill('new-user');
    await create.locator('[name="research_enabled"]').check();
    assert.equal(await create.locator('option[value="room-gold-01"]').isDisabled(), true);
    await create.locator('[name="managed_room_id"]').selectOption('room-silver-02');
    await create.getByRole('button', {name: '会員を発行', exact: true}).click();
    await admin.locator('#memberAdminStatus').getByText('会員を発行しました。', {exact: true}).waitFor();
    assert.equal(adminBody.research_enabled, true);
    assert.equal(adminBody.managed_room_id, 'room-silver-02');
    assert.equal(await create.locator('[name="managed_room_id"]').isDisabled(), true);
    let owner = admin.locator('form[data-id="owner"]');
    await owner.locator('[name="research_enabled"]').uncheck();
    await owner.getByRole('button', {name: '保存', exact: true}).click();
    await admin.locator('#memberAdminStatus').getByText('会員情報を保存しました。', {exact: true}).waitFor();
    assert.equal(adminBody.managed_room_id, '');
    const normal = admin.locator('form[data-id="normal"]');
    await normal.locator('[name="research_enabled"]').check();
    await normal.locator('[name="managed_room_id"]').selectOption('room-gold-01');
    await normal.getByRole('button', {name: '保存', exact: true}).click();
    await admin.waitForFunction(() => document.querySelector('form[data-id="normal"] [name="managed_room_id"]').value === 'room-gold-01' && !document.querySelector('form[data-id="normal"] button').disabled);
    assert.equal(adminBody.managed_room_id, 'room-gold-01');
    assert.equal(await create.evaluate(e => e.scrollWidth <= e.clientWidth), true);
    await create.screenshot({path: path.resolve(__dirname, '../results/member-room-admin-mobile.png')});
    assert.deepEqual(adminErrors, []);
    console.log('Admin UI: research plan creation, assigned-room exclusion, release/reassignment, persistence in forms and mobile width passed.');
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
