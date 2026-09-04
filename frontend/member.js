(() => {
  "use strict";
  const roots = [...document.querySelectorAll("[data-member-panel]")];
  let member = null;
  let busy = false;
  let revision = 0;
  let libraryRoot = null;
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
      <label>${label("新しいパスワード（8〜128文字）")}
        <input name="new_password" type="password" autocomplete="new-password" minlength="8" maxlength="128" required></label>
      <label>${label("新しいパスワード（確認）")}
        <input name="confirm_password" type="password" autocomplete="new-password" minlength="8" maxlength="128" required></label>
      <div class="member-actions"><button class="member-primary" type="submit">${label("パスワードを変更")}</button></div>
    </form>`;
  }

  function render() {
    document.getElementById("memberKifuParking").appendChild(document.getElementById("researchKifuPanel"));
    if (typeof initializeChatStampPickers === "function") initializeChatStampPickers();
    document.querySelectorAll("[data-member-entry]").forEach(button => {
      button.textContent = t(member ? "マイページ" : "ログイン");
    });
    roots.forEach((root, index) => {
      const hasTabs = member && !member.must_change_password;
      const prefix = `member-section-${index}`;
      let body;
      if (!member) {
        body = `<h4>${label("会員ログイン")}</h4>
          <p class="member-help"><strong>${label("支援者向けの会員機能です。")}</strong><br>
          ${label("ログインすると、公開部屋でも全スタンプを使用でき、棋譜を自分専用のライブラリにサーバー保存できます。")}</p>
          <p class="member-help"><a href="https://vrcgoita.com/support/" target="_blank" rel="noopener noreferrer">${label("支援について")}</a></p>
          <form data-action="login">
          <label>${label("会員ID")}<input name="member_id" autocomplete="username" autocapitalize="none" spellcheck="false" maxlength="32" required></label>
          <label>${label("パスワード")}<input name="password" type="password" autocomplete="current-password" maxlength="128" required></label>
          <div class="member-actions"><button class="member-primary" type="submit">${label("ログイン")}</button></div>
        </form><p class="member-help">${label("会員発行・パスワード再発行は運営へお問い合わせください。")}</p>`;
      } else if (member.must_change_password) {
        body = `<h4>${label("初回パスワード変更")}</h4><p>${label("会員ID")}：${escape(member.member_id)}</p>
          ${passwordForm(true)}<div class="member-actions"><button type="button" data-action="logout">${label("ログアウト")}</button></div>`;
      } else {
        const plan = member.paid_active ? "有料権限：有効" : member.paid_enabled ? "有料権限：期限切れ" : "有料権限：無効";
        body = `<dl class="member-info">
          <dt>${label("会員ID")}</dt><dd>${escape(member.member_id)}</dd>
          <dt>${label("プラン状態")}</dt><dd>${label(plan)}</dd>
          <dt>${label("有効期限")}</dt><dd>${escape(member.paid_until || t("期限なし"))}${member.paid_until ? " (JST)" : ""}</dd>
        </dl>
        <details><summary>${label("パスワード変更")}</summary>${passwordForm(false)}</details>
        <div class="member-actions"><button type="button" data-action="logout">${label("ログアウト")}</button></div>`;
      }
      const tabs = hasTabs ? `<div class="member-tabs" role="tablist" aria-label="${label("マイページ")}">
        <button type="button" role="tab" id="${prefix}-account-tab" aria-controls="${prefix}-account" aria-selected="true" data-action="account">${label("アカウント")}</button>
        <button type="button" role="tab" id="${prefix}-library-tab" aria-controls="${prefix}-library" aria-selected="false" tabindex="-1" data-action="library">${label("棋譜ライブラリ")}</button>
      </div>` : "";
      root.innerHTML = `${tabs}<div data-member-account id="${prefix}-account" ${hasTabs ? `role="tabpanel" aria-labelledby="${prefix}-account-tab"` : ""}>${body}</div><div class="member-status" role="status" aria-live="polite"></div>
        <div data-member-library id="${prefix}-library" role="tabpanel" aria-labelledby="${prefix}-library-tab" hidden>
        <p class="member-help">${label("保存上限：100件。保存した棋譜は本人だけが閲覧できます。")}</p>
        ${member && !member.paid_active ? `<p class="member-help">${label("新規保存には有効な有料権限が必要です。")}</p>` : ""}
        <div data-member-library-slot></div></div>`;
    });
    if (libraryRoot && member && !member.must_change_password) showLibrary(libraryRoot, false);
  }

  function resetLibrary() {
    libraryRoot = null;
    resetMemberKifuLibrary();
  }

  function showLibrary(root, load = true) {
    if (!member || member.must_change_password) return;
    if (libraryRoot && libraryRoot !== root) selectTab(libraryRoot, "account");
    libraryRoot = root;
    selectTab(root, "library");
    mountMemberKifuLibrary(root.querySelector("[data-member-library-slot]"), canUseAllStamps());
    if (load) {
      resetMemberKifuLibrary();
      loadResearchKifuList();
    }
  }

  function selectTab(root, action) {
    root.querySelector("[data-member-account]").hidden = action !== "account";
    root.querySelector("[data-member-library]").hidden = action !== "library";
    root.querySelectorAll('[role="tab"]').forEach(tab => {
      const selected = tab.dataset.action === action;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
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
      if (member?.member_id !== data.member?.member_id || data.member?.must_change_password) resetLibrary();
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
      resetLibrary();
      member = data.member || null;
      render();
      if (action === "password") status(t("パスワードを変更しました。"));
    } catch (error) {
      if (error.status === 401 && action !== "login") { resetLibrary(); member = null; render(); }
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
      if (event.target.closest('[data-action="library"]') && libraryRoot !== root) showLibrary(root);
      if (event.target.closest('[data-action="account"]')) {
        libraryRoot = null;
        stopResearchKifuReplay({restoreFinal: true, clearStatus: true});
        selectTab(root, "account");
      }
    });
    root.addEventListener("keydown", event => {
      const tab = event.target.closest('[role="tab"]');
      if (!tab || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const tabs = [...root.querySelectorAll('[role="tab"]')];
      const index = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 :
        (tabs.indexOf(tab) + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
      tabs[index].focus();
      tabs[index].click();
    });
  });
  function canUseAllStamps() {
    if (!member || member.must_change_password || !member.paid_active) return false;
    return !member.paid_until || Date.now() < Date.parse(`${member.paid_until}T23:59:59.999+09:00`);
  }
  window.goitaMembers = {refresh, clearSecrets, canUseAllStamps};
  render();
  refresh();
})();
