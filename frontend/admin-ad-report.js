(() => {
  "use strict";
  const element = id => document.getElementById(id);
  const escape = text => String(text ?? "").replace(/[&<>"']/g, c => ({
    "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;",
  })[c]);
  const number = value => Number(value || 0).toLocaleString("ja-JP");
  const rate = value => value == null ? "—" : `${Number(value).toFixed(2)}%`;
  const surfaceLabels = {public_room:"公開部屋", score_attack:"スコアアタック"};
  const sortLabels = {title:"広告名", status:"状態", impressions:"表示回数", clicks:"クリック数", click_rate:"クリック率"};
  let period = "30", report = null, selected = "", requestNumber = 0;
  let sortKey = "impressions", descending = true;

  function setPanel(settings) {
    element("regionalAdSettingsPanel").hidden = !settings;
    element("regionalAdReportPanel").hidden = settings;
    element("regionalAdShowSettings").setAttribute("aria-pressed", String(settings));
    element("regionalAdShowReport").setAttribute("aria-pressed", String(!settings));
  }

  function summary() {
    const values = report?.summary;
    element("regionalAdReportSummary").innerHTML = [
      ["表示回数", values ? `${number(values.impressions)}回` : "—"],
      ["クリック数", values ? `${number(values.clicks)}回` : "—"],
      ["クリック率", values ? rate(values.click_rate) : "—"],
    ].map(([label, value]) => `<div class="metric"><div class="metric-label">${label}</div><div class="metric-value">${value}</div></div>`).join("");
  }

  function sortedItems() {
    return [...(report?.items || [])].sort((a, b) => {
      const left = a[sortKey], right = b[sortKey];
      // Undefined CTRs stay last in both directions.
      if (left == null && right != null) return 1;
      if (right == null && left != null) return -1;
      const difference = typeof left === "number" ? left - right
        : String(left ?? "").localeCompare(String(right ?? ""), "ja");
      return difference ? difference * (descending ? -1 : 1)
        : String(a.id).localeCompare(String(b.id));
    });
  }

  function renderRows() {
    document.querySelectorAll("[data-ad-sort]").forEach(header => {
      const key = header.dataset.adSort;
      const active = key === sortKey;
      header.setAttribute("aria-sort", active ? (descending ? "descending" : "ascending") : "none");
      header.querySelector("button").textContent = sortLabels[key] + (active ? (descending ? " ▼" : " ▲") : "");
    });
    const body = element("regionalAdReportRows");
    body.replaceChildren();
    for (const item of sortedItems()) {
      const row = document.createElement("tr");
      row.classList.toggle("is-selected", selected === item.id);
      const cell = document.createElement("td");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "ad-report-open";
      button.textContent = item.title;
      button.setAttribute("aria-expanded", String(selected === item.id));
      button.addEventListener("click", () => {
        selected = item.id;
        renderRows();
        detail();
        element("regionalAdReportDetailTitle").focus({preventScroll:true});
      });
      cell.append(button); row.append(cell);
      for (const value of [item.status, number(item.impressions), number(item.clicks), rate(item.click_rate)]) {
        const next = document.createElement("td"); next.textContent = value; row.append(next);
      }
      body.append(row);
    }
    if (!report?.items.length) body.innerHTML = '<tr><td colspan="5">広告はまだありません。「広告の設定」から追加できます。</td></tr>';
  }

  function dailyRows(item) {
    const rows = new Map(item.daily.map(row => [row.date, row]));
    const start = report.period_start || report.period_end;
    const result = [];
    // UTC arithmetic is only for calendar labels, which are already JST dates.
    for (let day = Date.parse(`${start}T00:00:00Z`), end = Date.parse(`${report.period_end}T00:00:00Z`);
         day <= end; day += 86400000) {
      const date = new Date(day).toISOString().slice(0, 10);
      result.push(rows.get(date) || {date, impressions:0, clicks:0});
    }
    return result;
  }

  function chart(rows, field, label, color) {
    const max = Math.max(1, ...rows.map(row => row[field]));
    const width = 440, height = 190, left = 48, right = 12, top = 18, bottom = 33;
    const x = index => left + (rows.length <= 1 ? (width - left - right) / 2
      : index * (width - left - right) / (rows.length - 1));
    const y = value => height - bottom - value * (height - top - bottom) / max;
    const points = rows.map((row, i) => `${x(i).toFixed(1)},${y(row[field]).toFixed(1)}`).join(" ");
    const ticks = max <= 2 ? [0,max] : [0,Math.round(max / 2),max];
    const grid = ticks.map(value => `<line x1="${left}" y1="${y(value)}" x2="${width - right}" y2="${y(value)}" stroke="#d8dde3"/><text x="${left - 7}" y="${y(value) + 4}" text-anchor="end" fill="#67717d" font-size="12">${number(value)}</text>`).join("");
    const indices = [...new Set([0, Math.floor((rows.length - 1) / 2), rows.length - 1])];
    const dates = indices.map(i => `<text x="${x(i)}" y="${height - 10}" text-anchor="middle" fill="#67717d" font-size="12">${Number(rows[i].date.slice(5,7))}/${Number(rows[i].date.slice(8,10))}</text>`).join("");
    return `<figure class="ad-report-chart"><figcaption>${label}</figcaption><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${label}の推移。正確な数値は日別データの表で確認できます。"><title>${label}</title>${grid}<polyline points="${points}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>${rows.length === 1 ? `<circle cx="${x(0)}" cy="${y(rows[0][field])}" r="3" fill="${color}"/>` : ""}${dates}</svg></figure>`;
  }

  function breakdown(rows, label, key) {
    if (!rows.length) return '<p class="muted">実績はありません。</p>';
    return `<table><thead><tr><th>${label}</th><th>表示回数</th><th>クリック数</th><th>クリック率</th></tr></thead><tbody>${rows.map(row => `<tr><td>${escape(key === "surface" ? surfaceLabels[row[key]] || row[key] : row[key])}</td><td>${number(row.impressions)}</td><td>${number(row.clicks)}</td><td>${rate(row.click_rate)}</td></tr>`).join("")}</tbody></table>`;
  }

  function detail() {
    const item = report?.items.find(item => item.id === selected);
    element("regionalAdReportDetail").hidden = !item;
    if (!item) return;
    const title = element("regionalAdReportDetailTitle");
    title.textContent = item.title;
    title.tabIndex = -1;
    element("regionalAdReportBrowsers").textContent = period === "all"
      ? `閲覧ブラウザー数（概算）：${number(item.viewers)}。計測を開始してからの初回表示で集計しています。`
      : `期間内の初回表示ブラウザー（概算）：${number(item.viewers)}。この広告を初めて表示した記録の数です。期間中の全閲覧者数ではありません。`;
    const days = dailyRows(item);
    element("regionalAdReportCharts").innerHTML = item.daily.length
      ? chart(days, "impressions", "日別の表示回数", "#176b57") + chart(days, "clicks", "日別のクリック数", "#346dab")
      : '<p class="muted">選択した期間の実績はありません。</p>';
    element("regionalAdReportDaily").innerHTML = `<table><thead><tr><th>日付（日本時間）</th><th>表示回数</th><th>クリック数</th></tr></thead><tbody>${days.map(row => `<tr><td>${escape(row.date)}</td><td>${number(row.impressions)}</td><td>${number(row.clicks)}</td></tr>`).join("")}</tbody></table>`;
    element("regionalAdReportSurfaces").innerHTML = breakdown(item.surfaces, "表示場所", "surface");
    element("regionalAdReportRegions").innerHTML = breakdown(item.regions, "地方", "region");
  }

  async function load() {
    const request = ++requestNumber;
    report = null;
    summary();
    element("regionalAdReportRange").textContent = "";
    element("regionalAdReportDetail").hidden = true;
    element("regionalAdReportRows").innerHTML = '<tr><td colspan="5">読み込んでいます...</td></tr>';
    const status = element("regionalAdReportStatus");
    status.textContent = "実績を読み込んでいます...";
    try {
      const data = await api(`/admin/api/regional-ads/report?period=${period}`);
      if (request !== requestNumber) return;
      report = data;
      if (!report.items.some(item => item.id === selected)) selected = "";
      const range = report.period_start ? `${report.period_start} ～ ${report.period_end}（日本時間）` : "集計データはまだありません。";
      element("regionalAdReportRange").textContent = `${period === "all" ? "全期間：" : "集計期間："}${range}`;
      summary(); renderRows(); detail();
      status.textContent = `${report.items.length}件の広告`;
    } catch (error) {
      if (request !== requestNumber) return;
      status.textContent = error.message;
      element("regionalAdReportRows").innerHTML = '<tr><td colspan="5">実績を取得できませんでした。「実績を更新」で再試行できます。</td></tr>';
      if (error.status === 401) showLogin();
    }
  }

  element("regionalAdShowReport").addEventListener("click", () => {setPanel(false); load();});
  element("regionalAdShowSettings").addEventListener("click", () => setPanel(true));
  element("regionalAdReportRefresh").addEventListener("click", load);
  document.querySelectorAll("[data-ad-period]").forEach(button => button.addEventListener("click", () => {
    period = button.dataset.adPeriod;
    document.querySelectorAll("[data-ad-period]").forEach(other => other.setAttribute("aria-pressed", String(other === button)));
    load();
  }));
  document.querySelectorAll("[data-ad-sort]").forEach(header => header.querySelector("button").addEventListener("click", () => {
    const next = header.dataset.adSort;
    descending = next === sortKey ? !descending : !["title", "status"].includes(next);
    sortKey = next;
    if (report) renderRows();
  }));
  window.goitaAdReport = Object.freeze({open() {setPanel(false); return load();}});
  summary();
})();
