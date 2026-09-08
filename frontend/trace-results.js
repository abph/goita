let traceResultId = '';
let traceResultSeen = '';
let traceResultRequest = 0;
let traceOriginalPayload = null;
let traceOriginalFrames = [];
let traceOriginalStep = 0;
let traceOriginalTimer = null;
let traceOriginalRequest = 0;
let traceActionBusy = false;
let traceRankingMode = 'best';
let traceHistoryRequest = 0;
let traceHistoryOffset = 0;
const traceElement = id => document.getElementById(id);
const traceSigned = value => `${value > 0 ? '+' : ''}${value}点`;
const traceDate = value => new Date(value * 1000).toLocaleDateString('ja-JP', {timeZone:'Asia/Tokyo',year:'numeric',month:'2-digit',day:'2-digit'});

async function traceApi(path, body) {
  if (gid !== 'debug') throw new Error('デバッグルーム専用です。');
  const response = await fetch(`${API}/games/debug/${path}`, {
    method: body ? 'POST' : 'GET', credentials: 'same-origin', cache: 'no-store',
    headers: {'Content-Type': 'application/json', 'X-Goita-Member': '1'},
    ...(body ? {body: JSON.stringify(body)} : {}),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '結果を読み込めませんでした。');
  return data;
}

function syncTraceResult(state) {
  if (gid !== 'debug') {
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

async function openTraceResult(attemptId = '', mode = 'best') {
  if (gid !== 'debug') return;
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
    const data = await traceApi(`trace_results/${encodeURIComponent(id)}?mode=${mode}`);
    if (generation !== traceResultRequest || gid !== 'debug') return;
    traceResultId = id;
    traceRankingMode = mode;
    for (const value of ['best','first']) {
      traceElement(`traceRanking${value}`).setAttribute('aria-pressed', String(value === mode));
    }
    traceElement('traceChallengeLabel').textContent = data.challenge_label || '';
    for (const team of ['AC', 'BD']) {
      traceElement(`traceOriginal${team}`).textContent = `${data.original[team]}点`;
      traceElement(`traceActual${team}`).textContent = `${data.actual[team]}点`;
    }
    traceElement('traceImprovement').textContent = `元の棋譜との差：${traceSigned(data.improvement)}`;
    traceElement('traceRecordKind').textContent = `${data.attempt_no === 1 ? '初回' : `${data.attempt_no}回目`}の挑戦${data.is_best ? '・自己ベスト' : ''}`;
    traceElement('traceRetention').textContent = data.guest
      ? `ゲストの記録期限：${traceDate(data.expires_at)}` : '会員の成績として保存しました。';
    const rows = traceElement('traceRankingRows');
    rows.replaceChildren();
    for (const item of data.ranking) {
      const row = document.createElement('tr');
      if (item.self) row.className = 'trace-ranking-self';
      const values = [`${item.rank}位`, `${item.name}${item.guest ? '（ゲスト）' : ''}${item.self ? '（自分）' : ''}`,
        traceSigned(item.improvement), `${item.attempt_no}回目`, traceDate(item.finished_at)];
      for (const text of values) {
        const cell = document.createElement('td');
        cell.textContent = text;
        row.append(cell);
      }
      rows.append(row);
    }
    traceElement('traceRankingSummary').textContent = data.total
      ? `${data.total}人中${data.own_rank ? `・自分の${mode === 'best' ? 'ベスト' : '初回'}記録は${data.own_rank}位` : '・自分のランキング記録はありません'}（上位100人を表示）`
      : 'この棋譜のランキング記録はまだありません。';
    traceElement('traceRetryButton').disabled = mySeat !== 'A';
    traceElement('traceNextButton').disabled = mySeat !== 'A';
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
  if (gid !== 'debug') return;
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
    if (generation !== traceHistoryRequest || gid !== 'debug') return;
    offset = data.offset;
    traceHistoryOffset = offset;
    for (const item of data.records) {
      const row = document.createElement('tr');
      for (const text of [traceDate(item.finished_at),item.challenge_label,
        `${traceSigned(item.improvement)}${item.is_best ? ' ★' : ''}`,`${item.attempt_no}回目`]) {
        const cell = document.createElement('td');
        cell.textContent = text;
        row.append(cell);
      }
      const cell = document.createElement('td');
      const button = document.createElement('button');
      button.type = 'button'; button.textContent = '結果を見る';
      button.setAttribute('aria-label', `${item.challenge_label} ${item.attempt_no}回目の結果を見る`);
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
    if (generation !== traceOriginalRequest || gid !== 'debug') return;
    traceOriginalPayload = data.payload;
    traceOriginalFrames = researchKifuReplayFrames(data.payload);
    traceOriginalStep = traceOriginalFrames.length;
    renderTraceOriginal();
    traceElement('traceOriginalModal').style.display = 'flex';
    traceElement('traceOriginalModal').querySelector('.settings-modal-close').focus();
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
  traceElement('traceOriginalPosition').textContent = `${traceOriginalStep} / ${traceOriginalFrames.length} ${frame?.label || '配牌'}`;
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
  ++traceOriginalRequest;
  stopTraceOriginal();
  traceOriginalPayload = null;
  traceOriginalFrames = [];
  traceElement('traceOriginalBoard').replaceChildren();
  traceElement('traceOriginalModal').style.display = 'none';
}

async function continueTrace(retry) {
  if (traceActionBusy || gid !== 'debug' || mySeat !== 'A') return;
  traceActionBusy = true;
  traceElement('traceRetryButton').disabled = true;
  traceElement('traceNextButton').disabled = true;
  traceElement('traceResultStatus').textContent = '対戦を準備しています。';
  try {
    await traceApi(retry ? `trace_results/${encodeURIComponent(traceResultId)}/retry` : 'trace_random_start',
      {requester: 'A', client_id: clientId});
    closeTraceResult();
    await refresh();
  } catch (error) {
    traceElement('traceResultStatus').textContent = error.message;
  } finally {
    traceActionBusy = false;
    traceElement('traceRetryButton').disabled = mySeat !== 'A';
    traceElement('traceNextButton').disabled = mySeat !== 'A';
  }
}

async function startSameTrace() {
  if (traceActionBusy || gid !== 'debug') return;
  traceActionBusy = true;
  traceElement('debugTraceStatus').textContent = '同じ棋譜の対戦を準備しています。';
  try {
    await traceApi('trace_same_start', {requester:'A', client_id:clientId});
    closeDebugTrace();
    await refresh();
  } catch (error) {
    traceElement('debugTraceStatus').textContent = error.message;
  } finally {traceActionBusy = false;}
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
