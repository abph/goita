(() => {
  "use strict";
  const roots = [...document.querySelectorAll("[data-member-panel]")];
  let member = null;
  let busy = false;
  let revision = 0;
  const t = value => typeof uiText === "function" ? uiText(value) : value;
  const escape = value => String(value ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
  const label = value => escape(t(value));

  async function request(path, body) {
    const response = await fetch(`/api/member/${path}`, {
      method: body === undefined ? "GET" : "POST", credentials: "same-origin", cache: "no-store",
      headers: {"Content-Type": "application/json", "X-Goita-Member": "1"},
      ...(body === undefined ? {} : {body: JSON.stringify(body)}),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const error = new Error(data.detail || t("通信に失敗しました。"));
      error.status = response.status;
      throw error;
    }
    return data;
  }

  function passwordForm(initial) {
    return `<form data-action="password">
      <input name="username" autocomplete="username" value="${escape(member.member_id)}" hidden>
      <label>${label(initial ? "仮パスワード" : "現在のパスワード")}
        <input name="current_password" type="password" autocomplete="current-password" maxlength="128" required></label>
      <label>${label("新しいパスワード（15〜128文字）")}
        <input name="new_password" type="password" autocomplete="new-password" minlength="15" maxlength="128" required></label>
      <label>${label("新しいパスワード（確認）")}
        <input name="confirm_password" type="password" autocomplete="new-password" minlength="15" maxlength="128" required></label>
      <div class="member-actions"><button class="member-primary" type="submit">${label("パスワードを変更")}</button></div>
    </form>`;
  }

  function render() {
    document.querySelectorAll("[data-member-entry]").forEach(button => {
      button.textContent = t(member ? "マイページ" : "ログイン");
    });
    roots.forEach(root => {
      let body;
      if (!member) {
        body = `<h4>${label("会員ログイン")}</h4><form data-action="login">
          <label>${label("会員ID")}<input name="member_id" autocomplete="username" autocapitalize="none" spellcheck="false" maxlength="32" required></label>
          <label>${label("パスワード")}<input name="password" type="password" autocomplete="current-password" maxlength="128" required></label>
          <div class="member-actions"><button class="member-primary" type="submit">${label("ログイン")}</button></div>
        </form><p class="member-help">${label("会員発行・パスワード再発行は運営へお問い合わせください。")}</p>`;
      } else if (member.must_change_password) {
        body = `<h4>${label("初回パスワード変更")}</h4><p>${label("会員ID")}：${escape(member.member_id)}</p>
          ${passwordForm(true)}<div class="member-actions"><button type="button" data-action="logout">${label("ログアウト")}</button></div>`;
      } else {
        const plan = member.paid_active ? "有料権限：有効" : member.paid_enabled ? "有料権限：期限切れ" : "有料権限：無効";
        body = `<h4>${label("マイページ")}</h4><dl class="member-info">
          <dt>${label("会員ID")}</dt><dd>${escape(member.member_id)}</dd>
          <dt>${label("プラン状態")}</dt><dd>${label(plan)}</dd>
          <dt>${label("有効期限")}</dt><dd>${escape(member.paid_until || t("期限なし"))}${member.paid_until ? " (JST)" : ""}</dd>
        </dl><details><summary>${label("パスワード変更")}</summary>${passwordForm(false)}</details>
        <div class="member-actions"><button type="button" data-action="logout">${label("ログアウト")}</button></div>`;
      }
      root.innerHTML = `${body}<div class="member-status" role="status" aria-live="polite"></div>`;
    });
  }

  function status(message) {
    roots.forEach(root => { root.querySelector(".member-status").textContent = message; });
  }

  async function refresh() {
    if (busy) return;
    const ticket = ++revision;
    try {
      const data = await request("session");
      if (ticket !== revision) return;
      member = data.member;
      render();
    } catch (_error) {
      if (ticket === revision) status(t("通信に失敗しました。"));
    }
  }

  async function perform(action, form) {
    if (busy) return;
    let body = {};
    if (form) {
      body = Object.fromEntries(new FormData(form));
      if (action === "password") {
        if (body.new_password !== body.confirm_password) {
          status(t("確認用パスワードが一致しません。"));
          return;
        }
        delete body.confirm_password;
        delete body.username;
      }
    }
    busy = true;
    ++revision;
    roots.forEach(root => root.querySelectorAll("button").forEach(button => { button.disabled = true; }));
    status("");
    try {
      const data = await request(action, body);
      member = data.member || null;
      render();
      if (action === "password") status(t("パスワードを変更しました。"));
    } catch (error) {
      if (error.status === 401 && action !== "login") { member = null; render(); }
      status(t(error.message));
    } finally {
      clearSecrets();
      busy = false;
      roots.forEach(root => root.querySelectorAll("button").forEach(button => { button.disabled = false; }));
    }
  }

  function clearSecrets() {
    roots.forEach(root => root.querySelectorAll('input[type="password"]').forEach(input => { input.value = ""; }));
  }
  roots.forEach(root => {
    root.addEventListener("submit", event => { event.preventDefault(); perform(event.target.dataset.action, event.target); });
    root.addEventListener("click", event => {
      if (event.target.closest('[data-action="logout"]')) perform("logout");
    });
  });
  window.goitaMembers = {refresh, clearSecrets};
  render();
  refresh();
})();
