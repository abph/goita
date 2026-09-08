let kifuFileRounds = [];
let kifuFileFrames = [];
let kifuFileFrame = 0;
let kifuFileTimer = null;
let kifuFileRequest = null;
let kifuFileOpener = null;
const kifuFileElement = id => document.getElementById(id);

function openKifuFiles() {
  kifuFileOpener = document.activeElement?.closest('[data-header-menu]')?.querySelector('[data-header-menu-toggle]') || document.activeElement;
  const canSave = !!activeRoomId && latestState?.finished === true;
  kifuFileElement('kifuFileSave').disabled = !canSave;
  kifuFileElement('kifuFileSaveAnonymous').disabled = !canSave;
  kifuFileElement('kifuFilesModal').style.display = 'flex';
  kifuFileElement('kifuFilesModal').querySelector('.settings-modal-close').focus();
}

function stopKifuFilePlayback() {
  clearTimeout(kifuFileTimer);
  kifuFileTimer = null;
  kifuFileElement('kifuFilePlay').textContent = '再生';
}

function closeKifuFiles() {
  stopKifuFilePlayback();
  kifuFileRequest?.abort();
  kifuFileRequest = null;
  kifuFileRounds = [];
  kifuFileFrames = [];
  kifuFileElement('kifuFileInput').value = '';
  kifuFileElement('kifuFileViewer').hidden = true;
  for (const id of ['kifuFileBoard','kifuFileRound','kifuFileStatus','kifuFileMove']) kifuFileElement(id).replaceChildren();
  kifuFileElement('kifuFilesModal').style.display = 'none';
  kifuFileOpener?.focus();
}

async function readKifuFile(input) {
  const file = input.files?.[0];
  if (!file) return;
  stopKifuFilePlayback();
  kifuFileRequest?.abort();
  const request = new AbortController();
  kifuFileRequest = request;
  const status = kifuFileElement('kifuFileStatus');
  const timeout = setTimeout(() => request.abort(), 15000);
  status.textContent = '棋譜を読み込んでいます。';
  try {
    if (file.size > 200000) throw new Error('棋譜ファイルは200KB以下にしてください。');
    const text = await file.text();
    if (request !== kifuFileRequest || request.signal.aborted) return;
    const response = await fetch('/kifu/preview', {
      method:'POST', headers:{'Content-Type':'application/json'}, cache:'no-store',
      credentials:'omit', signal:request.signal, body:JSON.stringify({kifu_text:text}),
    });
    const data = await response.json();
    if (request !== kifuFileRequest) return;
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '棋譜を読み込めませんでした。');
    if (!Array.isArray(data.rounds) || !data.rounds.length) throw new Error('棋譜に局がありません。');
    kifuFileRounds = data.rounds;
    const select = kifuFileElement('kifuFileRound');
    select.replaceChildren();
    data.rounds.forEach((round, index) => {
      const option = document.createElement('option');
      option.value = index;
      option.textContent = `第${round.round_index}局（${round.winner} ${round.gained_score}点）`;
      select.append(option);
    });
    kifuFileElement('kifuFileViewer').hidden = false;
    selectKifuFileRound();
    status.textContent = `${file.name} ／ ${data.rounds.length}局`;
  } catch (error) {
    if (request === kifuFileRequest) status.textContent = request.signal.aborted ? '読み込みがタイムアウトしました。もう一度お試しください。' : error.message;
  } finally {
    clearTimeout(timeout);
    if (request === kifuFileRequest) { kifuFileRequest = null; input.value = ''; }
  }
}

function selectKifuFileRound() {
  stopKifuFilePlayback();
  const payload = kifuFileRounds[Number(kifuFileElement('kifuFileRound').value)];
  kifuFileFrames = payload ? researchKifuReplayFrames(payload) : [];
  kifuFileFrame = 0;
  renderKifuFileFrame();
}

function renderKifuFileFrame() {
  const payload = kifuFileRounds[Number(kifuFileElement('kifuFileRound').value)];
  if (!payload) return;
  const frame = kifuFileFrames[kifuFileFrame - 1];
  renderResearchKifuBoard(payload, {
    container:kifuFileElement('kifuFileBoard'), rows:frame?.rows || [],
    finished:kifuFileFrame === kifuFileFrames.length, latestAction:frame?.latestAction,
  });
  kifuFileElement('kifuFileMove').textContent = `${kifuFileFrame} / ${kifuFileFrames.length}　${frame?.label || '最初の配牌'}`;
}

function moveKifuFileFrame(amount, absolute = false) {
  stopKifuFilePlayback();
  kifuFileFrame = Math.max(0, Math.min(kifuFileFrames.length, absolute ? amount : kifuFileFrame + amount));
  renderKifuFileFrame();
}

function playKifuFile() {
  if (kifuFileTimer !== null) { stopKifuFilePlayback(); return; }
  if (!kifuFileFrames.length) return;
  if (kifuFileFrame >= kifuFileFrames.length) kifuFileFrame = 0;
  kifuFileElement('kifuFilePlay').textContent = '停止';
  renderKifuFileFrame();
  const next = () => {
    kifuFileFrame++;
    renderKifuFileFrame();
    if (kifuFileFrame >= kifuFileFrames.length) stopKifuFilePlayback();
    else kifuFileTimer = setTimeout(next, kifuFileFrames[kifuFileFrame - 1]?.delay || 620);
  };
  kifuFileTimer = setTimeout(next, 620);
}

document.addEventListener('keydown', event => {
  const modal = kifuFileElement('kifuFilesModal');
  if (modal.style.display !== 'flex') return;
  if (event.key === 'Escape') { event.preventDefault(); closeKifuFiles(); }
  if (event.key === 'Tab') {
    const controls = [...modal.querySelectorAll('button:not(:disabled),select,input:not([hidden])')].filter(el => el.getClientRects().length);
    const index = controls.indexOf(document.activeElement);
    if ((event.shiftKey && index <= 0) || (!event.shiftKey && index === controls.length - 1)) {
      event.preventDefault(); controls[event.shiftKey ? controls.length - 1 : 0]?.focus();
    }
  }
});

let debugTraceText = '';
let debugTraceRounds = [];
let debugTraceRequest = null;

function syncDebugTraceMenu() {
  const item = document.getElementById('debugTraceMenuItem');
  if (item) item.style.display = gid === 'debug' ? '' : 'none';
}

function openDebugTrace() {
  if (typeof gid === 'undefined' || gid !== 'debug') return;
  debugTraceText = '';
  debugTraceRounds = [];
  debugTraceRequest?.abort();
  const input = document.getElementById('debugTraceInput');
  const round = document.getElementById('debugTraceRound');
  const button = document.getElementById('debugTraceStartButton');
  const randomButton = document.getElementById('debugTraceRandomButton');
  const status = document.getElementById('debugTraceStatus');
  if (input) input.value = '';
  if (round) { round.replaceChildren(); round.disabled = true; }
  if (button) button.disabled = true;
  if (randomButton) randomButton.disabled = mySeat !== 'A';
  if (status) status.textContent = '棋譜ファイルを選択してください。';
  const modal = document.getElementById('debugTraceModal');
  modal.style.display = 'flex';
  modal.querySelector('.settings-modal-close')?.focus();
}

function closeDebugTrace() {
  debugTraceRequest?.abort();
  debugTraceRequest = null;
  debugTraceText = '';
  debugTraceRounds = [];
  const modal = document.getElementById('debugTraceModal');
  modal.style.display = 'none';
  document.getElementById('debugTraceInput').value = '';
  document.getElementById('debugTraceRound').replaceChildren();
  document.getElementById('debugTraceRound').disabled = true;
  document.getElementById('debugTraceStartButton').disabled = true;
  document.getElementById('debugTraceRandomButton').disabled = false;
}

async function previewDebugTrace(input) {
  const file = input.files?.[0];
  if (!file) return;
  debugTraceRequest?.abort();
  const request = new AbortController();
  debugTraceRequest = request;
  const status = document.getElementById('debugTraceStatus');
  const round = document.getElementById('debugTraceRound');
  const start = document.getElementById('debugTraceStartButton');
  status.textContent = '棋譜を確認しています。';
  start.disabled = true;
  try {
    if (file.size > 200000) throw new Error('棋譜ファイルは200KB以下にしてください。');
    debugTraceText = await file.text();
    const response = await fetch('/kifu/preview', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, cache: 'no-store',
      signal: request.signal, body: JSON.stringify({kifu_text: debugTraceText}),
    });
    const data = await response.json();
    if (request !== debugTraceRequest) return;
    if (!response.ok) throw new Error(data.detail || '棋譜を読み込めませんでした。');
    debugTraceRounds = data.rounds || [];
    round.replaceChildren(...debugTraceRounds.map((item, index) => {
      const option = document.createElement('option'); option.value = index + 1;
      option.textContent = `第${item.round_index}局（${item.winner} ${item.gained_score}点）`; return option;
    }));
    round.disabled = !debugTraceRounds.length;
    start.disabled = !debugTraceRounds.length;
    syncDebugTraceRoundStatus();
  } catch (error) {
    if (request === debugTraceRequest && !request.signal.aborted) status.textContent = error.message;
  } finally {
    if (request === debugTraceRequest) debugTraceRequest = null;
  }
}

function syncDebugTraceRoundStatus() {
  const index = Number(document.getElementById('debugTraceRound').value || 1) - 1;
  const item = debugTraceRounds[index];
  const status = document.getElementById('debugTraceStatus');
  if (item && status) status.textContent = `親：${item.dealer}　開始点数 AC：${item.score_before?.AC || 0} ／ BD：${item.score_before?.BD || 0}`;
}

async function startDebugTraceFromFile() {
  if (gid !== 'debug' || mySeat !== 'A' || !debugTraceText) return;
  const button = document.getElementById('debugTraceStartButton');
  const randomButton = document.getElementById('debugTraceRandomButton');
  const status = document.getElementById('debugTraceStatus');
  const round = Number(document.getElementById('debugTraceRound').value || 1);
  button.disabled = true;
  randomButton.disabled = true;
  status.textContent = '棋譜トレース対戦を準備しています。';
  try {
    const response = await fetch(`${API}/games/debug/trace_start`, {
      method: 'POST', credentials: 'same-origin', headers: {'Content-Type': 'application/json', 'X-Goita-Member': '1'},
      body: JSON.stringify({kifu_text: debugTraceText, round_index: round, requester: 'A', client_id: clientId}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '対戦を開始できませんでした。');
    closeDebugTrace();
    await refresh();
  } catch (error) {
    status.textContent = error.message;
    button.disabled = false;
    randomButton.disabled = mySeat !== 'A';
  }
}

async function startRandomDebugTrace() {
  if (gid !== 'debug' || mySeat !== 'A') return;
  const button = document.getElementById('debugTraceRandomButton');
  const fileButton = document.getElementById('debugTraceStartButton');
  const status = document.getElementById('debugTraceStatus');
  button.disabled = true;
  fileButton.disabled = true;
  status.textContent = '50点以下の棋譜からランダムに選んでいます。';
  try {
      const response = await fetch(`${API}/games/debug/trace_random_start`, {
        method: 'POST', credentials: 'same-origin',
        headers: {'Content-Type': 'application/json', 'X-Goita-Member': '1'},
      body: JSON.stringify({requester: 'A', client_id: clientId}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'ランダム棋譜を選べませんでした。');
    const source = data.source || {};
    closeDebugTrace();
    await refresh();
      if (source.match_id) {
        setHint(`ランダム棋譜を開始しました（${source.match_id}・第${source.round_index}局）。`);
      } else {
        setHint('50点以下のランダム棋譜を開始しました。');
      }
  } catch (error) {
    status.textContent = error.message;
    button.disabled = false;
    fileButton.disabled = !debugTraceRounds.length;
  }
}

window.addEventListener('load', syncDebugTraceMenu);
