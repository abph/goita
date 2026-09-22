(() => {
  "use strict";
  let lastRoom = "";
  let lastFetch = 0;
  let requestNumber = 0;
  const element = id => document.getElementById(id);

  function hide() { if (element("regionalAd")) element("regionalAd").hidden = true; }

  async function sync(isPublicRoom, roomId) {
    if (!isPublicRoom) { requestNumber++; lastRoom = ""; hide(); return; }
    if (!element("regionalAd")) return;
    const now = Date.now();
    if (lastRoom === roomId && now - lastFetch < 60000) return;
    lastRoom = roomId;
    lastFetch = now;
    hide();
    const current = ++requestNumber;
    try {
      const response = await fetch("/api/regional-ad", {credentials:"same-origin", cache:"no-store"});
      if (!response.ok) throw new Error("regional ad unavailable");
      const data = await response.json();
      if (current !== requestNumber || lastRoom !== roomId || !data.ad) return;
      const ad = data.ad;
      if (sessionStorage.getItem(`goitaRegionalAdDismissed:${ad.id}`)) return;
      element("regionalAdTitle").textContent = ad.title;
      element("regionalAdMessage").textContent = ad.message;
      const link = element("regionalAdLink");
      if (ad.url) link.href = ad.url;
      else link.removeAttribute("href");
      element("regionalAd").dataset.adId = ad.id;
      element("regionalAd").hidden = false;
      window.goitaI18n?.refresh?.();
    } catch (_error) { if (current === requestNumber) hide(); }
  }

  document.addEventListener("DOMContentLoaded", () => {
    element("regionalAdClose")?.addEventListener("click", () => {
      const id = element("regionalAd")?.dataset.adId;
      if (id) sessionStorage.setItem(`goitaRegionalAdDismissed:${id}`, "1");
      hide();
    });
  });
  window.goitaRegionalAds = Object.freeze({sync});
})();
