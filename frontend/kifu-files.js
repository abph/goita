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
  kifuFileElement('kifuBoardImageButton').disabled = !boardImageSourceAvailable();
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
    kifuFileElement('kifuBoardImageButton').disabled = false;
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

let debugTraceStarting = false;

function syncDebugTraceMenu() {
  const item = document.getElementById('debugTraceMenuItem');
  if (item) item.style.display = isScoreAttackRoom() ? '' : 'none';
}

function openDebugTrace() {
  if (typeof gid === 'undefined' || !isScoreAttackRoom()) return;
  syncScoreAttackMemberPrompt();
  document.getElementById('debugTraceRandomButton').disabled = mySeat !== 'A' || debugTraceStarting;
  document.getElementById('debugTraceStatus').textContent = debugTraceStarting
    ? '対戦を準備しています。' : mySeat === 'A' ? '' : 'A席に着席すると対戦を開始できます。';
  const modal = document.getElementById('debugTraceModal');
  modal.style.display = 'flex';
  modal.querySelector('.settings-modal-close')?.focus();
}

function closeDebugTrace() {
  document.getElementById('debugTraceModal').style.display = 'none';
}

function boardImageSourceAvailable() {
  const loadedViewer = kifuFileElement('kifuFileViewer');
  if (loadedViewer && !loadedViewer.hidden && kifuFileElement('kifuFileBoard')?.childElementCount) return true;
  if (document.body.classList.contains('shared-kifu-board-active') && kifuFileElement('sharedKifuBoard')?.childElementCount) return true;
  return !!activeRoomId && !!latestState && kifuFileElement('board')?.childElementCount > 0;
}

function cloneBoardImageElement(element) {
  const clone = element.cloneNode(true);
  clone.removeAttribute('id');
  clone.querySelectorAll('[id]').forEach(item => item.removeAttribute('id'));
  clone.querySelectorAll('.is-replay-new').forEach(item => item.classList.remove('is-replay-new'));
  clone.querySelectorAll('.name-thinking-spinner,.turn-countdown,.beginner-support-note').forEach(item => item.remove());
  clone.querySelectorAll('[aria-live]').forEach(item => item.removeAttribute('aria-live'));
  return clone;
}

function researchBoardImageSource(boardId) {
  const board = kifuFileElement(boardId);
  if (!board?.childElementCount) return null;
  const source = document.createElement('div');
  source.className = 'research-kifu-board-wrap';
  source.style.cssText = 'width:max-content;max-width:none;margin:0;padding:10px;background:#d1ab75;';
  const clone = cloneBoardImageElement(board);
  clone.style.setProperty('--research-cell', '56px');
  clone.style.setProperty('--research-gap', '5px');
  source.append(clone);
  return source;
}

function liveBoardImageSource() {
  const board = kifuFileElement('board');
  if (!board?.childElementCount || !activeRoomId || !latestState) return null;
  const source = document.createElement('div');
  source.style.cssText = [
    'display:flex', 'flex-direction:column', 'gap:14px', 'width:max-content',
    'max-width:none', 'padding:16px', 'background:#f4efdf', '--cell:64px', '--gap:6px',
  ].join(';');
  const boardWrap = document.createElement('div');
  boardWrap.className = 'board-wrap';
  boardWrap.style.cssText = 'width:max-content;max-width:none;margin:0;';
  boardWrap.append(cloneBoardImageElement(board));
  source.append(boardWrap);
  const hands = kifuFileElement('handsArea');
  if (hands?.childElementCount) {
    const handsClone = cloneBoardImageElement(hands);
    handsClone.style.cssText = 'display:flex;width:100%;max-width:none;margin:0;';
    source.append(handsClone);
  }
  return source;
}

function currentBoardImageSource() {
  const viewer = kifuFileElement('kifuFileViewer');
  if (viewer && !viewer.hidden) {
    const loaded = researchBoardImageSource('kifuFileBoard');
    if (loaded) return loaded;
  }
  if (document.body.classList.contains('shared-kifu-board-active')) {
    const shared = researchBoardImageSource('sharedKifuBoard');
    if (shared) return shared;
  }
  return liveBoardImageSource();
}

function inlineBoardImageStyles(source, target) {
  if (!(source instanceof Element) || !(target instanceof Element)) return;
  const computed = getComputedStyle(source);
  for (let index = 0; index < computed.length; index += 1) {
    const property = computed[index];
    target.style.setProperty(property, computed.getPropertyValue(property), computed.getPropertyPriority(property));
  }
  target.style.setProperty('animation', 'none', 'important');
  target.style.setProperty('transition', 'none', 'important');
  const sourceChildren = Array.from(source.children);
  const targetChildren = Array.from(target.children);
  sourceChildren.forEach((child, index) => inlineBoardImageStyles(child, targetChildren[index]));
}

function boardImageTimestamp() {
  const now = new Date();
  const two = value => String(value).padStart(2, '0');
  return `${now.getFullYear()}${two(now.getMonth() + 1)}${two(now.getDate())}_${two(now.getHours())}${two(now.getMinutes())}${two(now.getSeconds())}`;
}

function canvasToJpegBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error('JPEG conversion failed')), 'image/jpeg', 0.92);
  });
}

async function saveBoardImageSource(source, filename) {
  if (!source) throw new Error('表示できる盤面がありません。');
  if (document.fonts?.ready) await document.fonts.ready;
  const staging = document.createElement('div');
  staging.setAttribute('aria-hidden', 'true');
  staging.style.cssText = 'position:fixed;left:-100000px;top:0;z-index:-1;width:max-content;max-width:none;pointer-events:none;';
  staging.append(source);

  const viewClasses = ['board-view-3d', 'board-view-pixel', 'board-view-pixel-mono'];
  const removedClasses = viewClasses.filter(className => document.body.classList.contains(className));
  removedClasses.forEach(className => document.body.classList.remove(className));
  document.body.append(staging);
  let width;
  let height;
  let markup;
  try {
    const rect = source.getBoundingClientRect();
    width = Math.max(1, Math.ceil(source.scrollWidth || rect.width));
    height = Math.max(1, Math.ceil(source.scrollHeight || rect.height));
    const snapshot = source.cloneNode(true);
    inlineBoardImageStyles(source, snapshot);
    snapshot.setAttribute('xmlns', 'http://www.w3.org/1999/xhtml');
    markup = new XMLSerializer().serializeToString(snapshot);
  } finally {
    staging.remove();
    removedClasses.forEach(className => document.body.classList.add(className));
  }

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><foreignObject width="100%" height="100%">${markup}</foreignObject></svg>`;
  const svgUrl = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
  const image = new Image();
  image.decoding = 'async';
  await new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = () => reject(new Error('Board rendering failed'));
    image.src = svgUrl;
  });
  const scale = Math.min(2, 2400 / Math.max(width, height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  const context = canvas.getContext('2d');
  context.fillStyle = '#f4efdf';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  const jpeg = await canvasToJpegBlob(canvas);
  const jpegUrl = URL.createObjectURL(jpeg);
  const link = document.createElement('a');
  link.href = jpegUrl;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(jpegUrl), 1000);
}

async function runBoardImageDownload(button, source, statusId) {
  const status = kifuFileElement(statusId);
  const originalDisabled = button?.disabled === true;
  if (button) button.disabled = true;
  if (status) status.textContent = uiText('盤面画像を作成しています。');
  try {
    await saveBoardImageSource(source, `sorou_goita_board_${boardImageTimestamp()}.jpg`);
    if (status) status.textContent = uiText('盤面画像を保存しました。');
  } catch (error) {
    console.error('Board image export failed:', error);
    if (status) status.textContent = uiText(error?.message === '表示できる盤面がありません。' ? error.message : '盤面画像を保存できませんでした。');
  } finally {
    if (button) button.disabled = originalDisabled;
  }
}

function downloadCurrentBoardImage(button) {
  return runBoardImageDownload(button, currentBoardImageSource(), 'kifuFileStatus');
}

function downloadResearchKifuBoardImage(button) {
  return runBoardImageDownload(button, researchBoardImageSource('researchKifuBoard'), 'researchKifuDetailStatus');
}

function openScoreAttackHowTo() {
  closeDebugTrace();
  const modal = document.getElementById('scoreAttackHowToModal');
  modal.style.display = 'flex';
  modal.querySelector('.settings-modal-close')?.focus();
}

function closeScoreAttackHowTo() {
  document.getElementById('scoreAttackHowToModal').style.display = 'none';
  openDebugTrace();
}

async function startRandomDebugTrace() {
  if (!isScoreAttackRoom() || mySeat !== 'A' || debugTraceStarting) return;
  debugTraceStarting = true;
  const button = document.getElementById('debugTraceRandomButton');
  const status = document.getElementById('debugTraceStatus');
  button.disabled = true;
  status.textContent = 'スコアアタックの棋譜を選んでいます。';
  try {
    const response = await fetch(`${API}/games/${gid}/trace_random_start`, {
      method:'POST', credentials:'same-origin',
      headers:{'Content-Type':'application/json','X-Goita-Member':'1'},
      body:JSON.stringify({requester:'A',client_id:clientId}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'ランダム棋譜を選べませんでした。');
    closeDebugTrace();
    await refresh();
    setHint('スコアアタックを開始しました。');
  } catch (error) {
    status.textContent = error.message;
  } finally {
    debugTraceStarting = false;
    button.disabled = mySeat !== 'A';
  }
}

window.addEventListener('load', syncDebugTraceMenu);
