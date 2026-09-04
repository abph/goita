(() => {
  "use strict";
  const list = document.getElementById("memberList");
  const status = document.getElementById("memberAdminStatus");
  const credential = document.getElementById("memberCredential");
  let busy = false;
  let generation = 0;
  const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);

  async function request(path = "", method = "GET", body) {
    const started = generation;
    const response = await fetch(`/admin/api/members${path}`, {
      method, credentials: "same-origin", cache: "no-store",
      headers: {"Content-Type": "application/json", "X-Goita-Member": "1"},
      ...(body === undefined ? {} : {body: JSON.stringify(body)}),
    });
    const data = await response.json().catch(() => ({}));
    if (started !== generation) throw new Error("再度、会員管理を開いてください。");
    if (!response.ok) {
      if (response.status === 401) { clear(); showLogin(); }
      throw new Error(data.detail || "通信に失敗しました。");
    }
    return data;
  }

  function hideCredential() {
    credential.hidden = true;
    document.getElementById("memberCredentialText").value = "";
  }

  function showCredential(data) {
    credential.hidden = false;
    const id = data.member?.member_id || data.member_id;
    const expiry = new Date(data.temporary_expires_at * 1000).toLocaleString("ja-JP", {timeZone: "Asia/Tokyo"});
    document.getElementById("memberCredentialText").value =
      `会員ID: ${id}\n仮パスワード: ${data.temporary_password}\n有効期限: ${expiry} (JST)`;
  }

  async function load() {
    try {
      const data = await request();
      document.getElementById("memberPersistence").textContent = data.persistent
        ? "会員情報の永続保存：有効" : "会員情報はローカル保存です。本番環境では永続保存先を設定してください。";
      list.innerHTML = data.members.length ? data.members.map(member => `<form class="member-admin-row" data-id="${esc(member.member_id)}">
        <div class="member-admin-identity"><strong>${esc(member.member_id)}</strong>
          <span class="muted">${member.must_change_password ? "初回変更待ち" : "登録済み"} / ${member.paid_active ? "有料権限：有効" : "有料権限：無効・期限切れ"}</span></div>
        <label class="member-admin-check"><input type="checkbox" name="enabled" ${member.enabled ? "checked" : ""}>ログイン可</label>
        <label class="member-admin-check"><input type="checkbox" name="paid_enabled" ${member.paid_enabled ? "checked" : ""}>有料権限</label>
        <label>有効期限（JST）<input type="date" name="paid_until" value="${esc(member.paid_until)}" min="2000-01-01" max="9998-12-31"></label>
        <div class="member-admin-actions"><button class="button" type="submit">保存</button>
          <button class="button" type="button" data-reset>仮パスワード再発行</button></div>
      </form>`).join("") : '<p class="muted">会員はまだ登録されていません。</p>';
    } catch (error) { status.textContent = error.message; }
  }

  async function action(work) {
    if (busy) return;
    busy = true;
    status.textContent = "";
    hideCredential();
    document.querySelectorAll("#membersView button").forEach(button => { button.disabled = true; });
    try { await work(); }
    catch (error) { status.textContent = error.message; }
    finally {
      busy = false;
      document.querySelectorAll("#membersView button").forEach(button => { button.disabled = false; });
    }
  }

  document.getElementById("memberCreateForm").addEventListener("submit", event => {
    event.preventDefault();
    const form = event.target;
    action(async () => {
      const data = await request("", "POST", {
        member_id: form.elements.member_id.value,
        paid_enabled: form.elements.paid_enabled.checked,
        paid_until: form.elements.paid_until.value || null,
      });
      form.reset();
      await load();
      showCredential(data);
      status.textContent = "会員を発行しました。";
    });
  });
  list.addEventListener("submit", event => {
    event.preventDefault();
    const form = event.target;
    const enabled = form.elements.enabled.checked;
    if (!enabled && !window.confirm(`${form.dataset.id}のログインを停止し、すべてのログインを無効にしますか？`)) return;
    action(async () => {
      await request(`/${encodeURIComponent(form.dataset.id)}`, "PUT", {
        enabled, paid_enabled: form.elements.paid_enabled.checked, paid_until: form.elements.paid_until.value || null,
      });
      await load();
      status.textContent = "会員情報を保存しました。";
    });
  });
  list.addEventListener("click", event => {
    if (!event.target.closest("[data-reset]")) return;
    const id = event.target.closest("form").dataset.id;
    if (!window.confirm(`${id}の本人確認は済んでいますか？ 再発行すると、すべてのログインと現在のパスワードが無効になります。`)) return;
    action(async () => {
      const data = await request(`/${encodeURIComponent(id)}/reset-password`, "POST", {});
      await load();
      showCredential(data);
      status.textContent = "仮パスワードを再発行しました。";
    });
  });
  document.getElementById("memberCredentialClose").addEventListener("click", hideCredential);
  document.getElementById("memberRefresh").addEventListener("click", () => { hideCredential(); load(); });
  function clear() {
    ++generation;
    hideCredential();
    list.replaceChildren();
    document.getElementById("memberCreateForm").reset();
  }
  window.goitaMemberAdmin = {load, clear, hideCredential};
})();
