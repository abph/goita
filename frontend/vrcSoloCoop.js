/* VRChat A + up to three browser players; empty seats become AI at START. */
(() => {
    "use strict";
    let panel, status, title, pending = false, polling = false;
    let lastMeta = null;
    const seatRows = new Map();

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
        title = document.createElement("strong");
        title.textContent = "VRChat連携卓：A＝VRChat ／ B・C・D＝サイトから参加可能";
        status = document.createElement("p");
        status.setAttribute("role", "status");
        status.setAttribute("aria-live", "polite");
        panel.append(title, status);
        for (const seat of ["B", "C", "D"]) {
            const row = document.createElement("div");
            row.setAttribute("role", "group");
            row.setAttribute("aria-label", `${seat}席の参加`);
            row.style.cssText = "align-items:center;flex-wrap:wrap;gap:8px;margin-top:8px;";
            const label = document.createElement("span");
            const join = document.createElement("button");
            join.type = "button";
            join.textContent = `${seat}席で参加・再接続`;
            join.addEventListener("click", async () => {
                if (pending || !seatMeta(lastMeta)[seat]?.can_join) return;
                pending = true;
                renderMeta(lastMeta);
                try { await selectSeat(seat); }
                finally { pending = false; await poll(); }
            });
            const leave = document.createElement("button");
            leave.type = "button";
            leave.textContent = `${seat}席から退出`;
            leave.addEventListener("click", async () => {
                if (pending || !seatMeta(lastMeta)[seat]?.owned) return;
                if (!window.confirm(`${seat}席から退出しますか？ 開始前は席を空け、対局中はVRChat側でAIに交代できます。`)) return;
                pending = true;
                renderMeta(lastMeta);
                try { await leaveSeat(seat); }
                finally { pending = false; await poll(); }
            });
            row.append(label, join, leave);
            seatRows.set(seat, {row, label, join, leave});
            panel.append(row);
        }
        view.prepend(panel);
    }

    function seatMeta(meta) {
        return meta?.seats || (meta?.enabled ? {C: meta} : {});
    }

    function renderMeta(meta) {
        ensurePanel();
        if (!panel) return;
        panel.hidden = !isLinkedRoom();
        if (panel.hidden) return;
        lastMeta = meta;
        const seats = seatMeta(meta);
        const ownedSeat = Object.keys(seats).find(seat => seats[seat].owned);
        title.textContent = meta?.mixed ? "VRChat連携卓：VRChat最大3人＋サイトの参加者／空席はAI"
            : meta?.multi ? "VRChat連携卓：A＝VRChat ／ B・C・D＝サイトから参加可能（空席はAI）"
            : "VRChat連携卓：A＝VRChat ／ C＝サイト ／ B・D＝AI";
        let message;
        if (meta?.ended) {
            message = "VRChat側の利用が終了したため、この対局は終了しました。続ける場合はVRChat側でSTARTを押してから、サイトの席へ参加し直してください。";
        } else if (!meta?.active) {
            message = "VRChat側でSTARTを押すと参加できます。利用が終了した場合も、VRChat側で開始し直してください。";
        } else if (!meta.enabled) {
            message = "この卓は1人用です。サイトからは観戦できます。協力プレイにはUnity側の卓を更新してください。";
        } else if (ownedSeat) {
            message = meta.waiting ? `${ownedSeat}席に参加しました。参加者がそろったら、VRChat側でもう一度STARTを押すと対局が始まります。空席はAIになります。` :
                `${ownedSeat}席で対局中です。チームはA・C 対 B・Dです。盤面はVRChat側へ数秒遅れて反映されます。`;
        } else if (meta.multi) {
            message = meta.waiting ? "B・C・D席を選んで参加できます。VRChat側のSTARTで開始し、空席はAIになります。参加は1人1席です。" :
                "対局中のため新規参加はできません。使用中の席は保持され、同じブラウザから再接続できます。このまま観戦できます。";
        } else if (meta.can_join) {
            message = meta.joined ? "同じブラウザからC席へ再接続できます。" : "C席で参加できます。参加後、VRChat側のSTARTで対局を開始します。";
        } else if (meta.joined) {
            message = meta.online ? "C席は使用中です。このまま観戦できます。" :
                `C席の再接続待ちです（保持時間：あと${meta.reconnect_seconds}秒）。別の人は観戦のみです。`;
        } else {
            message = "C席はAIに交代しています。この対局は観戦できます。人間で参加するにはVRChat側で利用終了後、開始し直してください。";
        }
        status.textContent = message;
        for (const [seat, elements] of seatRows) {
            const info = seats[seat];
            elements.row.style.display = meta?.enabled && info ? "flex" : "none";
            if (!info) continue;
            const state = !meta.active ? "終了" : info.source === "vrchat" ? (info.online ? "VRChatから参加中" : "VRChatから退出・切断") : info.owned ? "あなたの席" : info.joined
                ? (info.online ? "使用中" : `再接続待ち（あと${info.reconnect_seconds}秒）`)
                : info.is_ai ? "AI" : "空席";
            elements.label.textContent = `${seat}席：${state}（${seat === "C" ? "Aの味方" : "B・Dチーム"}）`;
            elements.join.hidden = !meta.active || info.owned;
            elements.join.disabled = pending || !info.can_join || !!(ownedSeat && ownedSeat !== seat);
            elements.leave.hidden = !info.owned;
            elements.leave.disabled = pending;
        }
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
                if (status) status.textContent = "通信を確認しています。再接続中は操作をお待ちください。";
            }
        } finally { polling = false; }
    }

    window.goitaVrcCoop = {
        render(state) { if (isLinkedRoom()) renderMeta(state?.vrc_web); else if (panel) panel.hidden = true; },
        focus(seat = "C") { ensurePanel(); panel?.scrollIntoView({block: "nearest"}); seatRows.get(seat)?.join.focus(); },
    };
    document.addEventListener("DOMContentLoaded", () => {
        ensurePanel();
        if (panel) panel.hidden = true;
        poll();
        window.setInterval(poll, 6000);
    });
})();
