/* Debug-room human feedback. The snapshot is frozen before any form input. */
(() => {
  let dialog, snapshot, groups = [], selected, step = 1, loading = false, returnFocus;
  const $ = id => document.getElementById(id);
  const pieces = values => (values || []).map(p => GAME_LOG_PIECE_LABELS[p] || p).join(" ");
  const actionLabels = {receive: "受けの判断", attack: "攻めの判断", attack_after_block: "伏せ・攻めの判断", pass: "パスの判断"};
  const actionText = group => group.entries.map(line => removeGameLogSeatPrefix(
    localizeGameLogLine(stripGameLogTechnicalDetails(line), "ja"), group.seat
  )).join(" → ");

  function ensureDialog() {
    if(dialog) return;
    dialog = document.createElement("dialog");
    dialog.id = "aiReviewReportDialog";
    dialog.setAttribute("aria-labelledby", "aiReviewReportTitle");
    dialog.innerHTML = `
      <div class="ai-review-heading"><h2 id="aiReviewReportTitle">AIの判断についてレポートを作る</h2>
        <button type="button" id="aiReviewClose" aria-label="閉じる">×</button></div>
      <p id="aiReviewProgress" class="ai-review-muted"></p>
      <p id="aiReviewStatus" role="status"></p>
      <section id="aiReviewStep1">
        <p>局ごとの通し番号です。受けから攻めまでを1手番とし、パスも数えます。</p>
        <div id="aiReviewTurns" class="ai-review-turns" role="group" aria-label="対象の手番"></div>
        <label for="aiReviewDecision">この手番のどの判断についてですか？</label>
        <select id="aiReviewDecision"></select>
        <div id="aiReviewPosition"></div>
      </section>
      <section id="aiReviewStep2" hidden>
        <p id="aiReviewSelectedLabel" class="ai-review-selected"></p>
        <form id="aiReviewForm">
          <label for="aiReviewPreferred">どう打つのがよいと思いますか？ <span>必須</span></label>
          <input id="aiReviewPreferred" maxlength="1000" required placeholder="例：金ではなく、しを攻める">
          <label class="ai-review-check"><input id="aiReviewUnknown" type="checkbox">推奨手はまだ分からない</label>
          <label for="aiReviewReason">なぜそう考えますか？ <span>必須</span></label>
          <textarea id="aiReviewReason" rows="4" maxlength="5000" required placeholder="例：4しあるので、単発の金よりし攻めを継続したい。疑問点だけでも構いません。"></textarea>
          <label for="aiReviewPlan">その後、どのように進めたいですか？ <span>任意</span></label>
          <textarea id="aiReviewPlan" rows="2" maxlength="3000"></textarea>
          <label for="aiReviewExceptions">条件や例外、補足はありますか？ <span>任意</span></label>
          <textarea id="aiReviewExceptions" rows="2" maxlength="3000"></textarea>
        </form>
      </section>
      <section id="aiReviewStep3" hidden>
        <p>棋譜・配牌・ログを含むファイルを保存し、このチャットに添付してください。</p>
        <pre id="aiReviewPreview"></pre>
        <p class="ai-review-muted">対局時に残された判断を収録します。記録のない評価は「未記録」となります。</p>
        <div class="ai-review-exports"><button type="button" id="aiReviewSave">レポートを保存</button>
          <button type="button" id="aiReviewCopy">文章とデータをコピー</button></div>
      </section>
      <div class="ai-review-navigation"><button type="button" id="aiReviewBack">戻る</button>
        <button type="button" id="aiReviewNext">考えを入力する →</button></div>`;
    document.body.appendChild(dialog);
    $("aiReviewClose").onclick = () => dialog.close();
    dialog.addEventListener("close", () => returnFocus?.focus());
    $("aiReviewDecision").onchange = renderPosition;
    $("aiReviewUnknown").onchange = () => {
      $("aiReviewPreferred").disabled = $("aiReviewUnknown").checked;
      $("aiReviewPreferred").required = !$("aiReviewUnknown").checked;
    };
    $("aiReviewBack").onclick = () => showStep(step - 1);
    $("aiReviewNext").onclick = () => {
      if(step === 1 && !decision()) return;
      if(step === 2) {
        for(const id of ["aiReviewPreferred", "aiReviewReason"]) {
          const input = $(id);
          input.value = input.value.trim();
        }
        if(!$("aiReviewForm").reportValidity()) return;
      }
      showStep(step + 1);
    };
    $("aiReviewForm").onsubmit = event => {event.preventDefault(); $("aiReviewNext").click();};
    $("aiReviewSave").onclick = saveReport;
    $("aiReviewCopy").onclick = copyReport;
  }

  function decision() {
    const value = $("aiReviewDecision").value;
    return snapshot?.decisions.find(item => String(item.log_index) === value);
  }

  function selectGroup(group) {
    selected = group;
    $("aiReviewTurns").querySelectorAll("button").forEach(button => {
      const active = Number(button.dataset.logIndex) === group.logIndices[0];
      button.setAttribute("aria-pressed", String(active));
    });
    $("aiReviewDecision").replaceChildren();
    for(const item of snapshot.decisions.filter(item => group.logIndices.includes(item.log_index))) {
      const option = document.createElement("option");
      option.value = item.log_index;
      option.textContent = `${actionLabels[item.action[0]]}：${removeGameLogSeatPrefix(localizeGameLogLine(stripGameLogTechnicalDetails(item.log), "ja"), item.seat)}`;
      $("aiReviewDecision").appendChild(option);
    }
    renderPosition();
  }

  function selectedLabel() {
    return `第${snapshot.round_number}局・${selected.seat}手番${selected.turnNumber}\n${actionText(selected)}\n対象：${actionLabels[decision().action[0]]}`;
  }

  function renderPosition() {
    const item = decision();
    const root = $("aiReviewPosition");
    root.replaceChildren();
    if(!item) return;
    const heading = document.createElement("h3");
    heading.textContent = `${item.seat}手番${item.turn_number}・判断直前の局面`;
    const situation = document.createElement("p");
    situation.textContent = item.before.current_attack
      ? `${item.before.attacker}の${pieces([item.before.current_attack])}攻め／${item.seat}の${item.before.phase === "receive" ? "受け番" : "攻め番"}`
      : `${item.seat}の攻め番`;
    const table = document.createElement("table");
    const header = document.createElement("tr");
    for(const text of ["席", "手駒", "受け・伏せ", "攻め"]) {
      const cell = document.createElement("th"); cell.textContent = text; header.appendChild(cell);
    }
    table.appendChild(header);
    for(const seat of "ABCD") {
      const row = document.createElement("tr");
      if(seat === item.seat) row.className = "ai-review-actor";
      const board = item.before.board[seat];
      const receives = board.receive.map((piece, index) => board.receive_hidden[index] ? "伏せ（非公開）" : pieces(piece ? [piece] : [])).filter(Boolean).join(" / ");
      for(const text of [seat, pieces(item.before.hands[seat]), receives || "—", pieces(board.attack.filter(Boolean)) || "—"]) {
        const cell = document.createElement("td"); cell.textContent = text; row.appendChild(cell);
      }
      table.appendChild(row);
    }
    const note = document.createElement("p"); note.className = "ai-review-muted";
    note.textContent = "検証用に全員の手駒を表示しています。判断は本人の手駒と、その時点の公開情報をもとに検討します。";
    root.append(heading, situation, table, note);
    const thoughts = document.createElement("details");
    const summary = document.createElement("summary"); summary.textContent = "この判断のAI思考を見る";
    const body = document.createElement("p"); body.className = "ai-review-thought";
    const rows = gameLogThoughtRows(item.log, "ja");
    body.textContent = rows.length ? rows.map(row => `${row.label}：${row.value}`).join("\n") : "判断理由・候補評価は未記録です。";
    thoughts.append(summary, body); root.appendChild(thoughts);
  }

  function reviewData() {
    return {
      preferred_action: $("aiReviewUnknown").checked ? null : $("aiReviewPreferred").value.trim(),
      preferred_action_unknown: $("aiReviewUnknown").checked,
      reason: $("aiReviewReason").value.trim(),
      continuation: $("aiReviewPlan").value.trim(),
      conditions_and_exceptions: $("aiReviewExceptions").value.trim(),
      status: "proposed",
    };
  }

  function summaryText() {
    const review = reviewData();
    return `${selectedLabel()}\n\n推奨手：${review.preferred_action || "まだ分からない"}\n理由：${review.reason}\nその後の狙い：${review.continuation || "記入なし"}\n条件・例外・補足：${review.conditions_and_exceptions || "記入なし"}\n\n収録：初期配牌、棋譜、対象判断直前の局面、対局時のログ\n局面の取得日時：${snapshot.captured_at}`;
  }

  function reportData() {
    const item = decision();
    return {...snapshot, exported_at: new Date().toISOString(),
      target: {seat: item.seat, turn_number: item.turn_number, decision_log_index: item.log_index,
        turn_log_indices: selected.logIndices, action: item.action, label: selectedLabel()},
      human_review: reviewData(), summary: summaryText()};
  }

  function showStep(value) {
    step = value;
    dialog.querySelector(".ai-review-navigation").hidden = false;
    for(let i = 1; i <= 3; i++) $("aiReviewStep" + i).hidden = i !== step;
    $("aiReviewProgress").textContent = `${step} / 3　${["対象の判断を選ぶ", "人間の考えを入力する", "内容を確認して保存する"][step - 1]}`;
    $("aiReviewBack").hidden = step === 1;
    $("aiReviewNext").hidden = step === 3;
    $("aiReviewNext").disabled = !decision();
    $("aiReviewNext").textContent = step === 1 ? "考えを入力する →" : "内容を確認する →";
    $("aiReviewStatus").textContent = `第${snapshot.round_number}局の記録を保持しています。対局が進んでも、このレポートの局面は変わりません。`;
    if(step === 2) $("aiReviewSelectedLabel").textContent = selectedLabel();
    if(step === 3) $("aiReviewPreview").textContent = summaryText();
    if(step === 2) $("aiReviewPreferred").disabled ? $("aiReviewReason").focus() : $("aiReviewPreferred").focus();
    if(step === 3) $("aiReviewSave").focus();
  }

  function saveReport() {
    const data = reportData();
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: "application/json;charset=utf-8"}));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `goita_ai_review_${snapshot.captured_at.replace(/[^0-9]/g, "").slice(0, 14)}_round${snapshot.round_number}_${selected.seat}_turn${selected.turnNumber}.json`;
    document.body.appendChild(anchor); anchor.click(); anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    $("aiReviewStatus").textContent = "レポートを出力しました。保存したファイルをチャットに添付してください。";
  }

  async function copyReport() {
    try {
      await navigator.clipboard.writeText(`${summaryText()}\n\n${JSON.stringify(reportData(), null, 2)}`);
      $("aiReviewStatus").textContent = "文章と再現用データをコピーしました。";
    } catch(error) {
      $("aiReviewStatus").textContent = "コピーできませんでした。「レポートを保存」をご利用ください。";
    }
  }

  window.openAiReviewReport = async function(logIndex = null) {
    if(gid !== DEBUG_GID || !isCurrentClientHost() || loading) return;
    ensureDialog();
    returnFocus = document.activeElement;
    if(!dialog.open) dialog.showModal();
    if(logIndex === null && snapshot) return; // Resume the frozen draft, including across round changes.
    loading = true;
    snapshot = null;
    for(let i = 1; i <= 3; i++) $("aiReviewStep" + i).hidden = true;
    dialog.querySelector(".ai-review-navigation").hidden = true;
    $("aiReviewStatus").textContent = "局面とログを取得しています…";
    $("aiReviewNext").disabled = true;
    try {
      const query = new URLSearchParams({client_id: clientId, round_id: latestState?.review_round_id || ""});
      const response = await fetch(`${API}/games/${gid}/ai_review_snapshot?${query}`);
      const data = await response.json();
      if(!response.ok) throw new Error(data.detail || "レポート用の局面を取得できませんでした。");
      if(!data.decisions?.length) throw new Error("まだ手番のログがありません。1手進めてから作成してください。");
      groups = groupGameLogEntries(data.log).filter(group => group.seat);
      const target = logIndex === null ? groups[groups.length - 1] : groups.find(group => group.logIndices.includes(logIndex));
      if(!target) throw new Error("対象の手番が見つかりません。ログを確認してください。");
      snapshot = data;
      $("aiReviewTurns").replaceChildren();
      for(const group of groups) {
        const button = document.createElement("button");
        button.type = "button"; button.dataset.logIndex = group.logIndices[0];
        const title = document.createElement("strong"); title.textContent = `${group.seat}手番${group.turnNumber}`;
        const detail = document.createElement("span"); detail.textContent = actionText(group);
        button.append(title, detail); button.onclick = () => selectGroup(group);
        $("aiReviewTurns").appendChild(button);
      }
      $("aiReviewForm").reset();
      $("aiReviewPreferred").disabled = false; $("aiReviewPreferred").required = true;
      selectGroup(target);
      showStep(1);
      $("aiReviewStatus").textContent = `第${snapshot.round_number}局の記録を保持しました。対局が進んでも、このレポートの局面は変わりません。`;
      const active = $("aiReviewTurns").querySelector('[aria-pressed="true"]');
      active?.focus(); active?.scrollIntoView({block: "nearest"});
    } catch(error) {
      $("aiReviewStatus").textContent = error.message;
    } finally {loading = false;}
  };
})();
