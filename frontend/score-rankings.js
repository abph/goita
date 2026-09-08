let scoreRankingsRequest = 0;
let scoreRankingsReturnFocus = null;

async function openScoreRankings(period = 'daily') {
  const element = id => document.getElementById(id);
  const modal = element('scoreRankingsModal');
  if (modal.style.display !== 'flex') {
    scoreRankingsReturnFocus = document.activeElement;
    modal.style.display = 'flex';
    modal.querySelector('.settings-modal-close').focus();
  }
  const generation = ++scoreRankingsRequest;
  const weekly = period === 'weekly';
  for (const value of ['daily', 'weekly']) {
    element(`scorePeriod${value}`).setAttribute('aria-pressed', String(period === value));
  }
  element('scoreRankingsScoreHeader').textContent = weekly ? '合計点数' : '合計点差';
  element('scoreRankingsNote').textContent = weekly
    ? '日ごとの合計点差を、マイナスの日は0点として合計します。今日の分は対戦結果に応じて変わります。'
    : '今日終了した対戦の「元の対局との差」を合計します。';
  element('scoreRankingsPeriod').textContent = '';
  element('scoreRankingsStatus').textContent = 'ランキングを読み込んでいます。';
  element('scoreRankingsRows').replaceChildren();
  try {
    const response = await fetch(`${API}/api/score-attack/rankings?period=${period}`, {
      credentials:'same-origin', cache:'no-store', headers:{'X-Goita-Member':'1'},
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'ランキングを読み込めませんでした。');
    if (generation !== scoreRankingsRequest) return;
    element('scoreRankingsPeriod').textContent = (weekly
      ? `${data.start_date} 〜 ${data.end_date}` : data.start_date) + '（日本時間）';
    for (const item of data.ranking) {
      const row = document.createElement('tr');
      if (item.self) row.className = 'trace-ranking-self';
      for (const value of [`${item.rank}位`,
        `${item.name}${item.guest ? '（ゲスト）' : ''}${item.self ? '（自分）' : ''}`,
        weekly ? `${item.score}点` : traceSigned(item.score), `${item.games}局`]) {
        const cell = document.createElement('td');
        cell.textContent = value;
        row.append(cell);
      }
      element('scoreRankingsRows').append(row);
    }
    element('scoreRankingsStatus').textContent = !data.total ? 'この期間の記録はまだありません。'
      : data.total > 100 ? '上位100人を表示しています。' : '';
  } catch (error) {
    if (generation === scoreRankingsRequest) element('scoreRankingsStatus').textContent = error.message;
  }
}

function closeScoreRankings() {
  ++scoreRankingsRequest;
  document.getElementById('scoreRankingsModal').style.display = 'none';
  scoreRankingsReturnFocus?.focus();
}

document.addEventListener('keydown', event => {
  const modal = document.getElementById('scoreRankingsModal');
  if (modal.style.display !== 'flex') return;
  if (event.key === 'Escape') {
    event.preventDefault(); closeScoreRankings();
  } else if (event.key === 'Tab') {
    const buttons = [...modal.querySelectorAll('button')];
    const index = buttons.indexOf(document.activeElement);
    if ((event.shiftKey && index <= 0) || (!event.shiftKey && index === buttons.length - 1)) {
      event.preventDefault(); buttons[event.shiftKey ? buttons.length - 1 : 0].focus();
    }
  }
});
