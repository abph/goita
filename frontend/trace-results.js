let traceResultId = '';

function updateRoundResetButton(state, isHost, autoNextRoundPending) {
  const button = document.getElementById('btnNewGame');
  if (!button) return;
  button.style.display = isHost && state.finished && !autoNextRoundPending ? '' : 'none';
  button.disabled = nextRoundRequestInFlight || traceActionBusy;
  if (nextRoundRequestInFlight) {
    button.textContent = uiText('配牌中...');
  } else if (isScoreAttackRoom() && state.trace_mode) {
    button.textContent = 'リセット';
    button.onclick = resetScoreAttack;
  } else if (state.match_finished) {
    button.textContent = '新規ゲーム (スコアリセット)';
    button.onclick = () => startNewGame(false);
  } else if (state.finished) {
    button.textContent = '次の一局へ';
    button.onclick = () => startNewGame(true);
  }
}

async function resetScoreAttack() {
  if (!isScoreAttackRoom() || mySeat !== 'A' || traceActionBusy) return;
  traceActionBusy = true;
  document.getElementById('btnNewGame').disabled = true;
  closeTraceResult();
  try {
    if (isPersonalScoreRoom()) {
      await traceApi("score_reset", {requester:"A", client_id:clientId});
      pending = null;
    } else {
      await startNewGame(false);
    }
    await refresh();
    if (isPersonalScoreRoom()) openDebugTrace();
  } catch (error) {
    alert(error.message);
  } finally {
    traceActionBusy = false;
    document.getElementById('btnNewGame').disabled = false;
  }
}
let traceResultSeen = '';
let traceResultRequest = 0;
let traceOriginalPayload = null;
let traceOriginalFrames = [];
let traceOriginalStep = 0;
let traceOriginalTimer = null;
let traceOriginalRequest = 0;
let traceActionBusy = false;
let traceHistoryRequest = 0;
let traceHistoryOffset = 0;
const traceElement = id => document.getElementById(id);
const traceSigned = value => `${value === 0 ? '±' : value > 0 ? '+' : ''}${value}点`;
const traceDate = value => new Date(value * 1000).toLocaleDateString('ja-JP', {timeZone:'Asia/Tokyo',year:'numeric',month:'2-digit',day:'2-digit'});

async function traceApi(path, body) {
  if (!isScoreAttackRoom()) throw new Error('スコアアタックのルームで開いてください。');
  const response = await fetch(`${API}/games/${gid}/${path}`, {
    method: body ? 'POST' : 'GET', credentials: 'same-origin', cache: 'no-store',
    headers: {'Content-Type': 'application/json', 'X-Goita-Member': '1'},
    ...(body ? {body: JSON.stringify(body)} : {}),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '結果を読み込めませんでした。');
  return data;
}

function syncTraceResult(state) {
  if (!isScoreAttackRoom()) {
    closeTraceResult();
    closeTraceHistory();
    traceResultSeen = '';
    return;
  }
  if (state?.trace_mode && state.finished && state.trace_attempt_id && state.trace_attempt_id !== traceResultSeen) {
    traceResultSeen = state.trace_attempt_id;
    openTraceResult(state.trace_attempt_id);
  }
}

async function openTraceResult(attemptId = '') {
  if (!isScoreAttackRoom()) return;
  closeTraceOriginal();
  closeTraceHistory();
  const generation = ++traceResultRequest;
  const modal = traceElement('traceResultModal');
  modal.style.display = 'flex';
  modal.querySelector('.settings-modal-close').focus();
  traceElement('traceResultContent').hidden = true;
  traceElement('traceResultStatus').textContent = '結果を読み込んでいます。';
  try {
    const id = attemptId || (await traceApi('trace_results/latest')).attempt_id;
    if (!id) throw new Error('まだ終了した対戦の記録がありません。');
    const data = await traceApi(`trace_results/${encodeURIComponent(id)}?mode=all`);
    if (generation !== traceResultRequest || !isScoreAttackRoom()) return;
    traceResultId = id;
    const scoreLabel = score => score.BD > score.AC ? `BD：${score.BD}点` : `AC：${score.AC}点`;
    traceElement('traceActualScore').textContent = scoreLabel(data.actual);
    traceElement('traceOriginalScore').textContent = scoreLabel(data.original);
    traceElement('traceImprovement').textContent = `元の対局との差　${traceSigned(data.improvement)}`;
    const rows = traceElement('traceRankingRows');
    rows.replaceChildren();
    for (const item of data.ranking) {
      const row = document.createElement('tr');
      if (item.self) row.className = 'trace-ranking-self';
      const values = [`${item.rank}位`, `${item.name}（${item.attempt_no === 1 ? '初回' : `${item.attempt_no}回目`}）${item.guest ? '（ゲスト）' : ''}${item.self ? '（自分）' : ''}`,
        traceSigned(item.improvement), traceDate(item.finished_at)];
      for (const text of values) {
        const cell = document.createElement('td');
        cell.textContent = text;
        row.append(cell);
      }
      rows.append(row);
    }
    traceElement('traceResultContent').hidden = false;
    traceElement('traceResultStatus').textContent = '';
  } catch (error) {
    if (generation === traceResultRequest) traceElement('traceResultStatus').textContent = error.message;
  }
}

function closeTraceResult() {
  ++traceResultRequest;
  traceElement('traceResultModal').style.display = 'none';
  closeTraceOriginal();
}

async function openTraceHistory(offset = 0) {
  if (!isScoreAttackRoom()) return;
  closeTraceResult();
  const generation = ++traceHistoryRequest;
  traceElement('traceHistoryModal').style.display = 'flex';
  traceElement('traceHistoryModal').querySelector('.settings-modal-close').focus();
  traceElement('traceHistoryStatus').textContent = '挑戦履歴を読み込んでいます。';
  traceElement('traceHistoryRows').replaceChildren();
  traceElement('traceHistoryPrev').disabled = true;
  traceElement('traceHistoryNext').disabled = true;
  try {
    const data = await traceApi(`trace_results/history?offset=${offset}&limit=30`);
    if (generation !== traceHistoryRequest || !isScoreAttackRoom()) return;
    offset = data.offset;
    traceHistoryOffset = offset;
    for (const item of data.records) {
      const row = document.createElement('tr');
      for (const text of [traceDate(item.finished_at),item.challenge_label,
        `${traceSigned(item.improvement)}${item.is_best ? ' ★' : ''}`]) {
        const cell = document.createElement('td');
        cell.textContent = text;
        row.append(cell);
      }
      const cell = document.createElement('td');
      const button = document.createElement('button');
      button.type = 'button'; button.textContent = '結果を見る';
      button.setAttribute('aria-label', `${traceDate(item.finished_at)} ${item.challenge_label}の結果を見る`);
      button.onclick = () => openTraceResult(item.attempt_id);
      cell.append(button); row.append(cell);
      traceElement('traceHistoryRows').append(row);
    }
    traceElement('traceHistoryStatus').textContent = data.total
      ? `${data.total}件中 ${offset + 1}〜${offset + data.records.length}件（★は自己ベスト）`
      : 'まだ終了した挑戦の記録がありません。';
    traceElement('traceHistoryPrev').disabled = offset === 0;
    traceElement('traceHistoryNext').disabled = offset + data.records.length >= data.total;
    traceElement('traceHistoryModal').querySelector('.trace-history-scroll').scrollTo({left:0,top:0});
  } catch (error) {
    if (generation === traceHistoryRequest) traceElement('traceHistoryStatus').textContent = error.message;
  }
}

function closeTraceHistory() {
  ++traceHistoryRequest;
  traceElement('traceHistoryModal').style.display = 'none';
}

async function openTraceOriginal() {
  const generation = ++traceOriginalRequest;
  traceElement('traceResultStatus').textContent = '元の棋譜を読み込んでいます。';
  try {
    const data = await traceApi(`trace_results/${encodeURIComponent(traceResultId)}/original`);
    if (generation !== traceOriginalRequest || !isScoreAttackRoom()) return;
    traceOriginalPayload = data.payload;
    traceOriginalFrames = researchKifuReplayFrames(data.payload);
    traceOriginalStep = traceOriginalFrames.length;
    renderTraceOriginal();
    traceElement('traceOriginalModal').style.display = 'flex';
    traceElement('traceOriginalBack').focus();
    traceElement('traceResultStatus').textContent = '';
  } catch (error) {
    if (generation === traceOriginalRequest) traceElement('traceResultStatus').textContent = error.message;
  }
}

function renderTraceOriginal() {
  if (!traceOriginalPayload) return;
  const frame = traceOriginalFrames[traceOriginalStep - 1];
  renderResearchKifuBoard(traceOriginalPayload, {
    container: traceElement('traceOriginalBoard'), rows: frame?.rows || [],
    finished: traceOriginalStep === traceOriginalFrames.length, latestAction: frame?.latestAction,
  });
}

function stopTraceOriginal() {
  clearTimeout(traceOriginalTimer);
  traceOriginalTimer = null;
  traceElement('traceOriginalPlay').textContent = '棋譜再生';
}

function stepTraceOriginal(delta) {
  stopTraceOriginal();
  traceOriginalStep = Math.max(0, Math.min(traceOriginalFrames.length, traceOriginalStep + delta));
  renderTraceOriginal();
}

function playTraceOriginal() {
  if (traceOriginalTimer) { stopTraceOriginal(); return; }
  traceOriginalStep = 0;
  renderTraceOriginal();
  traceElement('traceOriginalPlay').textContent = '停止';
  const next = () => {
    traceOriginalStep++;
    renderTraceOriginal();
    if (traceOriginalStep >= traceOriginalFrames.length) { stopTraceOriginal(); return; }
    traceOriginalTimer = setTimeout(next, traceOriginalFrames[traceOriginalStep - 1]?.delay || 760);
  };
  if (traceOriginalFrames.length) traceOriginalTimer = setTimeout(next, 620);
}

function closeTraceOriginal() {
  const wasOpen = traceElement('traceOriginalModal').style.display === 'flex';
  ++traceOriginalRequest;
  stopTraceOriginal();
  traceOriginalPayload = null;
  traceOriginalFrames = [];
  traceElement('traceOriginalBoard').replaceChildren();
  traceElement('traceOriginalModal').style.display = 'none';
  if (wasOpen && traceElement('traceResultModal').style.display === 'flex') traceElement('traceViewOriginalButton').focus();
}

document.addEventListener('keydown', event => {
  const original = traceElement('traceOriginalModal');
  const result = traceElement('traceResultModal');
  const history = traceElement('traceHistoryModal');
  const modal = original.style.display === 'flex' ? original : result.style.display === 'flex' ? result : history.style.display === 'flex' ? history : null;
  if (!modal) return;
  if (event.key === 'Escape') {
    event.preventDefault();
    if (modal === original) { closeTraceOriginal(); traceElement('traceViewOriginalButton').focus(); }
    else if (modal === history) closeTraceHistory();
    else closeTraceResult();
  }
  if (event.key === 'Tab') {
    const controls = [...modal.querySelectorAll('button:not(:disabled),[href]')].filter(element => element.getClientRects().length);
    const index = controls.indexOf(document.activeElement);
    if ((event.shiftKey && index <= 0) || (!event.shiftKey && index === controls.length - 1)) {
      event.preventDefault(); controls[event.shiftKey ? controls.length - 1 : 0]?.focus();
    }
  }
});
