(() => {
  "use strict";
  let lastContext = "";
  let lastFetch = 0;
  let requestNumber = 0;
  const surfaces = new Set(["public_room", "score_attack"]);
  const element = id => document.getElementById(id);

  function hide() { if (element("regionalAd")) element("regionalAd").hidden = true; }

  function recordMetric(adId, event, surface, useBeacon = false) {
    if (!adId) return;
    const body = JSON.stringify({ad_id:adId, event, surface});
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
      element("regionalAdTitle").textContent = ad.title;
      element("regionalAdMessage").textContent = ad.message;
      const link = element("regionalAdLink");
      if (ad.url) link.href = ad.url;
      else link.removeAttribute("href");
      element("regionalAd").dataset.adId = ad.id;
      element("regionalAd").dataset.surface = surface;
      element("regionalAd").hidden = false;
      const impressionKey = `goitaRegionalAdImpression:${ad.id}:${context}`;
      if (surface !== "score_attack" || !sessionStorage.getItem(impressionKey)) {
        if (surface === "score_attack") sessionStorage.setItem(impressionKey, "1");
        recordMetric(ad.id, "impression", surface);
      }
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
