// Personal rooms use the same page as public rooms. Only the server owns the room identity.
function isPersonalScoreRoom(room = gid) {
  return String(room || '').startsWith('score-');
}
function isScoreAttackRoom() {
  return gid === 'debug' || isPersonalScoreRoom();
}
let scoreRoomEntering = false;
let scoreRoomLeaving = false;
let scoreActivitySentAt = 0;

async function enterScoreAttackRoom() {
  if (scoreRoomEntering) return false;
  scoreRoomEntering = true;
  const button = document.getElementById('scoreAttackEntry');
  button.disabled = true;
  try {
    const response = await fetch(`${API}/api/score-attack/enter`, {
      method: 'POST', credentials: 'same-origin',
      headers: {'Content-Type':'application/json','X-Goita-Member':'1'},
      body: JSON.stringify({client_id:clientId, name:personalSettings.playerName || ''}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'ルームを準備できませんでした。');
    await enterRoom(data.game_id, 'スコアアタック');
    sessionStorage.setItem('goitaScoreRoomOpen', '1');
    if (!latestState?.is_started) openDebugTrace();
    return true;
  } catch (error) {
    alert(error.message);
    return false;
  } finally {
    scoreRoomEntering = false;
    button.disabled = false;
  }
}

function syncScoreRoomUI() {
  const personal = isPersonalScoreRoom();
  document.body.classList.toggle('personal-score-room', personal);
  for (const seat of ['A','B','C','D']) {
    document.getElementById(`btnSeat${seat}`).disabled = personal;
  }
}

async function leaveExpiredScoreRoom() {
  if (scoreRoomLeaving || !isPersonalScoreRoom()) return;
  scoreRoomLeaving = true;
  try {
    await returnToLobby();
    alert('スコアアタックのルームは終了しました。トップページから入り直してください。');
  } finally {
    scoreRoomLeaving = false;
  }
}

// Add the private API header to the existing game's state/action requests.
// Restrict this wrapper to our own score room URLs; public room traffic is unchanged.
const scoreRoomFetch = window.fetch.bind(window);
window.fetch = async function(input, init) {
  const url = new URL(input instanceof Request ? input.url : String(input), location.href);
  if (url.origin !== location.origin || !/^\/games\/score-[^/]+\//.test(url.pathname)) {
    return scoreRoomFetch(input, init);
  }
  const headers = new Headers(init?.headers || (input instanceof Request ? input.headers : undefined));
  headers.set('X-Goita-Member', '1');
  const response = await scoreRoomFetch(input, {...init, headers, credentials:'same-origin', cache:'no-store'});
  if ([401,403,410].includes(response.status) && /\/state$/.test(url.pathname)
      && url.pathname.startsWith(`/games/${gid}/`)) {
    void leaveExpiredScoreRoom();
  }
  return response;
};

// Only trusted interaction, never a timer/AI/WebSocket refresh, renews inactivity.
function recordScoreActivity(event) {
  if (!event.isTrusted || !isPersonalScoreRoom() || document.getElementById('gameView').style.display === 'none') return;
  const now = Date.now();
  if (now - scoreActivitySentAt < 10000) return;
  scoreActivitySentAt = now;
  void fetch(`${API}/games/${gid}/score_activity`, {method:'POST'}).catch(() => {});
}
document.addEventListener('pointerdown', recordScoreActivity, {passive:true});
document.addEventListener('keydown', recordScoreActivity, {passive:true});
// A programmatic scroll also emits a trusted scroll event, so use user input instead.
document.addEventListener('wheel', recordScoreActivity, {passive:true});
document.addEventListener('touchmove', recordScoreActivity, {passive:true});
