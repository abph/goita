(() => {
  "use strict";
  let lastContext = "";
  let lastFetch = 0;
  let requestNumber = 0;
  const DAILY_LIMIT = 3;
  const EXPOSURE_STORAGE_KEY = "goitaRegionalAdDailyExposure";
  const VIEWER_STORAGE_PREFIX = "goitaRegionalAdViewer";
  const surfaces = new Set(["public_room", "score_attack"]);
  const element = id => document.getElementById(id);

  function hide() { if (element("regionalAd")) element("regionalAd").hidden = true; }

  function recordMetric(adId, event, surface, useBeacon = false, details = {}) {
    if (!adId) return;
    const body = JSON.stringify({ad_id:adId, event, surface, ...details});
    if (useBeacon && navigator.sendBeacon) {
      navigator.sendBeacon(
        "/api/regional-ad/metric",
        new Blob([body], {type:"application/json"}),
      );
      return;
    }
    fetch("/api/regional-ad/metric", {
      method:"POST",
      credentials:"same-origin",
      headers:{"Content-Type":"application/json"},
      body,
      keepalive:useBeacon,
    }).catch(() => {});
  }

  function japanDate() {
    return new Date(Date.now() + 9 * 60 * 60 * 1000).toISOString().slice(0, 10);
  }

  function reserveExposure(adId, surface) {
    try {
      const date = japanDate();
      const saved = JSON.parse(localStorage.getItem(EXPOSURE_STORAGE_KEY) || "null");
      const state = saved?.date === date && saved.counts && typeof saved.counts === "object"
        ? saved
        : {date, counts:{}};
      const count = Number(state.counts[adId] || 0);
      if (count >= DAILY_LIMIT) return {allowed:false, viewerFirst:false, surfaceViewerFirst:false};
      state.counts[adId] = count + 1;
      localStorage.setItem(EXPOSURE_STORAGE_KEY, JSON.stringify(state));

      const viewerKey = `${VIEWER_STORAGE_PREFIX}:${adId}`;
      const surfaceViewerKey = `${VIEWER_STORAGE_PREFIX}:${adId}:${surface}`;
      const viewerFirst = !localStorage.getItem(viewerKey);
      const surfaceViewerFirst = !localStorage.getItem(surfaceViewerKey);
      if (viewerFirst) localStorage.setItem(viewerKey, "1");
      if (surfaceViewerFirst) localStorage.setItem(surfaceViewerKey, "1");
      return {allowed:true, viewerFirst, surfaceViewerFirst};
    } catch (_error) {
      return {allowed:true, viewerFirst:false, surfaceViewerFirst:false};
    }
  }

  async function sync(surface, roomId, exposureId = "") {
    if (!surfaces.has(surface)) { requestNumber++; lastContext = ""; hide(); return; }
    if (!element("regionalAd")) return;
    const context = `${surface}:${roomId}:${exposureId || "room"}`;
    const now = Date.now();
    if (lastContext === context && now - lastFetch < 60000) return;
    const contextChanged = lastContext !== context;
    lastContext = context;
    lastFetch = now;
    if (contextChanged) hide();
    const current = ++requestNumber;
    try {
      const response = await fetch(`/api/regional-ad?surface=${encodeURIComponent(surface)}`, {credentials:"same-origin", cache:"no-store"});
      if (!response.ok) throw new Error("regional ad unavailable");
      const data = await response.json();
      if (current !== requestNumber || lastContext !== context) return;
      if (!data.ad) { hide(); return; }
      const ad = data.ad;
      if (sessionStorage.getItem(`goitaRegionalAdDismissed:${ad.id}`)) return;
      const renderKey = `${ad.id}:${context}`;
      const adElement = element("regionalAd");
      if (!adElement.hidden && adElement.dataset.renderKey === renderKey) return;
      const exposure = reserveExposure(ad.id, surface);
      if (!exposure.allowed) { hide(); return; }
      element("regionalAdTitle").textContent = ad.title;
      element("regionalAdMessage").textContent = ad.message;
      const link = element("regionalAdLink");
      if (ad.url) link.href = ad.url;
      else link.removeAttribute("href");
      adElement.dataset.adId = ad.id;
      adElement.dataset.surface = surface;
      adElement.dataset.renderKey = renderKey;
      adElement.hidden = false;
      recordMetric(ad.id, "impression", surface, false, {
        viewer_measurement: true,
        viewer_first: exposure.viewerFirst,
        surface_viewer_first: exposure.surfaceViewerFirst,
      });
      window.goitaI18n?.refresh?.();
    } catch (_error) { if (current === requestNumber) hide(); }
  }

  document.addEventListener("DOMContentLoaded", () => {
    element("regionalAdClose")?.addEventListener("click", () => {
      const id = element("regionalAd")?.dataset.adId;
      if (id) sessionStorage.setItem(`goitaRegionalAdDismissed:${id}`, "1");
      hide();
    });
    element("regionalAdLink")?.addEventListener("click", () => {
      const ad = element("regionalAd");
      const link = element("regionalAdLink");
      if (ad?.dataset.adId && link?.hasAttribute("href")) {
        recordMetric(ad.dataset.adId, "click", ad.dataset.surface || "public_room", true);
      }
    });
  });
  window.goitaRegionalAds = Object.freeze({sync});
})();
