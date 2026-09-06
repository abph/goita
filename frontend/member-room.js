(() => {
  "use strict";
  let generation = 0;
  let active = null;
  const t = text => typeof uiText === "function" ? uiText(text) : text;
  const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
  const label = text => esc(t(text));

  function reset() {
    ++generation;
    active?.replaceChildren();
    active = null;
  }

  async function mount(root, member) {
    reset();
    active = root;
    const ticket = generation;
    if (!member.paid_active) { root.textContent = t("ルーム管理には有効な研究用プランが必要です。"); return; }
    if (!member.managed_room_id) { root.textContent = t("管理する部屋が割り当てられていません。"); return; }
    root.innerHTML = `<p class="member-help">${label("読み込み中...")}</p>`;
    let room = null;
    let busy = false;
    const current = () => ticket === generation && root === active;
    async function request(path = "", body) {
      const response = await fetch(`/api/member/room${path}`, {
        method: body === undefined ? "GET" : "POST", credentials: "same-origin", cache: "no-store",
        headers: {"Content-Type": "application/json", "X-Goita-Member": "1"},
        ...(body === undefined ? {} : {body: JSON.stringify(body)}),
      });
      const data = await response.json().catch(() => ({}));
      if (!current()) return null;
      if (!response.ok) {
        if ([401, 403].includes(response.status)) root.replaceChildren();
        throw new Error(data.detail || t("通信に失敗しました。"));
      }
      return data.room;
    }
    function status(message) {
      let node = root.querySelector('[role="status"]');
      if (!node) { node = document.createElement('p'); node.className = 'member-status'; node.setAttribute('role', 'status'); root.appendChild(node); }
      node.textContent = t(message);
    }
    function render() {
      root.innerHTML = `<div class="member-actions">
        <button type="button" data-room-enter>${label("この部屋に入る")}</button>
        <button type="button" data-room-refresh>${label("更新")}</button></div>
        <form data-room-settings>
          <label>${label("ルーム名（最大12文字）")}<input name="new_owner_name" maxlength="12" value="${esc(room.owner_name)}" required></label>
          <label class="member-room-check"><input type="checkbox" name="update_password">${label("入室用の合言葉を変更・解除する")}</label>
          <label data-room-password hidden>${label("新しい合言葉（空欄で解除）")}<input name="new_password" type="password" maxlength="128" autocomplete="new-password"></label>
          <label>${label("AIの選択")}<select name="ai_profile">${Object.entries(room.ai_profiles).map(([key, name]) => `<option value="${esc(key)}" ${room.ai_profile === key ? "selected" : ""}>${label(name)}</option>`).join("")}</select></label>
          <label class="member-room-check"><input type="checkbox" name="show_legal_actions" ${room.show_legal_actions ? "checked" : ""}>${label("（デバッグ用）合法手を表示する")}</label>
          <label class="member-room-check"><input type="checkbox" name="show_log" ${room.show_log ? "checked" : ""}>${label("ログを表示する")}</label>
          <div class="member-actions"><button type="submit" class="member-primary">${label("ルーム管理を保存")}</button></div>
        </form>
        <details class="member-room-seats"><summary>${label("着席者の管理")}</summary><div data-room-seats></div></details>
        <p class="member-status" role="status" aria-live="polite"></p>`;
      const seats = root.querySelector('[data-room-seats]');
      if (!room.managed_human_seats.length) seats.textContent = t("着席者はいません。");
      room.managed_human_seats.forEach(item => {
        const row = document.createElement('div'); row.className = 'member-room-seat';
        const name = document.createElement('span'); name.setAttribute('data-i18n-ignore', '');
        name.textContent = `${item.seat}: ${item.name || item.seat}`;
        const button = document.createElement('button'); button.type = 'button'; button.textContent = t("席を空ける");
        button.addEventListener('click', () => {
          if (window.confirm(`${item.seat}: ${item.name || item.seat}\n${t("この席を空けますか？")}`)) {
            work('/vacate', {game_id: room.game_id, seat: item.seat, occupancy_token: item.occupancy_token}, "席を空けました。");
          }
        });
        row.append(name, button); seats.appendChild(row);
      });
      const form = root.querySelector('form');
      form.elements.update_password.addEventListener('change', () => {
        root.querySelector('[data-room-password]').hidden = !form.elements.update_password.checked;
        if (!form.elements.update_password.checked) form.elements.new_password.value = '';
      });
      form.addEventListener('submit', event => {
        event.preventDefault(); event.stopPropagation();
        work('/settings', {
          game_id: room.game_id, new_owner_name: form.elements.new_owner_name.value,
          update_password: form.elements.update_password.checked,
          new_password: form.elements.update_password.checked ? form.elements.new_password.value : '',
          ai_profile: form.elements.ai_profile.value,
          show_legal_actions: form.elements.show_legal_actions.checked, show_log: form.elements.show_log.checked,
        }, "ルーム設定を保存しました。");
      });
      root.querySelector('[data-room-refresh]').addEventListener('click', () => work());
      root.querySelector('[data-room-enter]').addEventListener('click', () => {
        const {game_id, is_private} = room;
        const alreadyInRoom = typeof gid !== 'undefined' && gid === game_id
          && document.getElementById('gameView').style.display !== 'none';
        closeLobbySettings(); closeSettings();
        if (!alreadyInRoom) tryJoinRoom(game_id, is_private);
      });
    }
    async function work(path = '', body, message = '') {
      if (busy || !current()) return;
      busy = true;
      root.querySelectorAll('button, input, select').forEach(node => {node.disabled = true;});
      status("読み込み中...");
      try {
        const data = await request(path, body);
        if (!data || !current()) return;
        room = data; render(); status(message);
        if (body && typeof fetchRoomList === 'function') fetchRoomList();
      } catch (error) { if (current()) status(error.message); }
      finally {
        busy = false;
        if (current()) root.querySelectorAll('button, input, select').forEach(node => {node.disabled = false;});
      }
    }
    await work();
  }
  window.goitaMemberRoom = {mount, reset};
})();
