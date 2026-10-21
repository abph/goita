/* VRChat A + browser C + AI B/D. Existing Goita board/actions remain unchanged. */
(() => {
    "use strict";
    let panel, status, joinButton, leaveButton, pending = false, polling = false;
    let lastMeta = null;

    function isLinkedRoom() {
        return typeof gid !== "undefined" && /^vrc-solo-[a-z0-9][a-z0-9-]{0,31}$/.test(gid);
    }

    function ensurePanel() {
        if (panel) return;
        const view = document.getElementById("gameView");
        if (!view) return;
        panel = document.createElement("section");
        panel.id = "vrcCoopPanel";
        panel.setAttribute("aria-label", "VRChat連携卓への参加");
        panel.style.cssText = "margin:10px 0;padding:14px;border:2px solid #386a62;border-radius:8px;background:#f4faf8;color:#173d35;";
        const title = document.createElement("strong");
        title.textContent = "VRChat連携卓：A＝VRChat ／ C＝サイト ／ B・D＝AI";
        status = document.createElement("p");
        status.setAttribute("role", "status");
        status.setAttribute("aria-live", "polite");
        joinButton = document.createElement("button");
        joinButton.type = "button";
        joinButton.textContent = "C席で参加・再接続";
        joinButton.addEventListener("click", async () => {
            if (pending || !lastMeta?.can_join) return;
            pending = true;
            joinButton.disabled = true;
            try { await selectSeat("C"); }
            finally { pending = false; await poll(); }
        });
        leaveButton = document.createElement("button");
        leaveButton.type = "button";
        leaveButton.textContent = "C席から退出";
        leaveButton.addEventListener("click", async () => {
            if (pending || !lastMeta?.owned) return;
            if (!window.confirm("C席から退出しますか？ VRChat側でAIに交代できます。")) return;
            pending = true;
            leaveButton.disabled = true;
            try { await leaveSeat("C"); }
            finally { pending = false; await poll(); }
        });
        panel.append(title, status, joinButton, leaveButton);
        view.prepend(panel);
    }

    function renderMeta(meta) {
        ensurePanel();
        if (!panel) return;
        panel.hidden = !isLinkedRoom();
        if (panel.hidden) return;
        lastMeta = meta;
        let message;
        if (meta?.ended) {
            message = "VRChat側の利用が終了したため、この対局は終了しました。続ける場合はVRChat側でSTARTを押してから、C席へ参加し直してください。";
        } else if (!meta?.active) {
            message = "VRChat側でSTARTを押すと参加できます。利用が終了した場合も、VRChat側で開始し直してください。";
        } else if (!meta.enabled) {
            message = "この卓は1人用です。サイトからは観戦できます。協力プレイにはUnity側の卓を更新してください。";
        } else if (meta.owned) {
            message = meta.waiting ? "C席に参加しました。VRChat側でもう一度STARTを押すと対局が始まります。" :
                "C席で対局中です。A席のVRChatプレイヤーと協力して遊べます。盤面はVRChat側へ数秒遅れて反映されます。";
        } else if (meta.can_join) {
            message = meta.joined ? "同じブラウザからC席へ再接続できます。" : "C席で参加できます。参加後、VRChat側のSTARTで対局を開始します。";
        } else if (meta.joined) {
            message = meta.online ? "C席は使用中です。このまま観戦できます。" :
                `C席の再接続待ちです（保持時間：あと${meta.reconnect_seconds}秒）。別の人は観戦のみです。`;
        } else {
            message = "C席はAIに交代しています。この対局は観戦できます。人間で参加するにはVRChat側で利用終了後、開始し直してください。";
        }
        status.textContent = message;
        joinButton.hidden = !meta?.enabled || !meta?.active || meta?.owned;
        joinButton.disabled = pending || !meta?.can_join;
        leaveButton.hidden = !meta?.owned;
        leaveButton.disabled = pending;
    }

    async function poll() {
        if (!isLinkedRoom()) {
            if (panel) panel.hidden = true;
            return;
        }
        if (polling) return;
        polling = true;
        const room = gid;
        try {
            const response = await fetch(`${API}/vrc/solo/${room.slice(9)}/web/heartbeat`, {
                method: "POST", headers: {"Content-Type": "application/json"},
                body: JSON.stringify({client_id: clientId}), cache: "no-store",
            });
            if (!response.ok) throw new Error("heartbeat failed");
            const meta = await response.json();
            if (gid !== room) return;
            renderMeta(meta);
            // Covers reconnects, initial room creation and a lost WebSocket update.
            await refresh();
        } catch (error) {
            if (gid === room) {
                ensurePanel();
                if (status) status.textContent = "通信を確認しています。再接続中はC席の操作をお待ちください。";
            }
        } finally { polling = false; }
    }

    window.goitaVrcCoop = {
        render(state) { if (isLinkedRoom()) renderMeta(state?.vrc_web); else if (panel) panel.hidden = true; },
        focus() { ensurePanel(); panel?.scrollIntoView({block: "nearest"}); joinButton?.focus(); },
    };
    document.addEventListener("DOMContentLoaded", () => {
        ensurePanel();
        if (panel) panel.hidden = true;
        poll();
        window.setInterval(poll, 6000);
    });
})();
