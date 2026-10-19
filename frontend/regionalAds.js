(() => {
  "use strict";

  const AUTO_SLIDE_MS = 60 * 1000;
  const DAILY_LIMIT = 3;
  const EXPOSURE_STORAGE_KEY = "goitaRegionalAdDailyExposure";
  const VIEWER_STORAGE_PREFIX = "goitaRegionalAdViewer";
  const ROTATION_STORAGE_PREFIX = "goitaRegionalAdRotation";
  const backgroundThemes = new Set(["standard", "blue", "green", "orange", "pink", "gray"]);
  const surfaces = new Set(["public_room", "score_attack"]);
  const element = id => document.getElementById(id);

  let lastContext = "";
  let lastFetch = 0;
  let requestNumber = 0;
  let carouselAds = [];
  let currentIndex = 0;
  let currentSurface = "public_room";
  let renderedKey = "";
  let rotationTimer = 0;
  let interactionPaused = false;
  let touchStartX = null;
  let suppressClickUntil = 0;
  let shownInContext = new Set();
  let currentImpressionEligible = false;
  let preferredPosition = "top";
  let placementFrame = 0;

  function stopRotation() {
    if (rotationTimer) window.clearTimeout(rotationTimer);
    rotationTimer = 0;
  }

  function hide() {
    stopRotation();
    const ad = element("regionalAd");
    if (ad) ad.hidden = true;
    carouselAds = [];
    currentIndex = 0;
    renderedKey = "";
    shownInContext = new Set();
    currentImpressionEligible = false;
    element("regionalAdTrack")?.replaceChildren();
    element("regionalAdDots")?.replaceChildren();
    updatePlacement();
  }

  function applyPosition(position) {
    preferredPosition = position === "bottom" ? "bottom" : "top";
    updatePlacement();
  }

  function updatePlacement() {
    const slot = element("regionalAdSlot");
    if (!slot) return;
    const game = element("gameView");
    const chat = element("chatPanel");
    const layout = document.querySelector(".play-layout");
    const main = layout?.querySelector(".play-main-column");
    const log = element("logCard");
    let sideWidth = 0;
    if (!element("regionalAd")?.hidden && window.innerWidth > 980
        && game && main && chat && chat.parentElement === layout) {
      const mainWidth = main.getBoundingClientRect().width;
      const chatWidth = chat.getBoundingClientRect().width;
      const gap = Number.parseFloat(window.getComputedStyle(layout).columnGap) || 16;
      const logWidth = game.classList.contains("desktop-log-visible") && window.innerWidth >= 1150
        ? log?.getBoundingClientRect().width || 0 : 0;
      const requiredWidth = Math.ceil(mainWidth + chatWidth + gap * 2 + 200 + (logWidth ? logWidth + 16 : 0));
      // Reserve the full board, chat, and any desktop log before adding the ad.
      if (mainWidth > 0 && chatWidth > 0 && requiredWidth <= document.documentElement.clientWidth - 40) {
        sideWidth = requiredWidth;
      }
    }
    game?.classList.toggle("regional-ad-side-visible", sideWidth > 0);
    if (sideWidth) {
      const width = `${sideWidth}px`;
      if (game.style.getPropertyValue("--regional-ad-layout-width") !== width) {
        game.style.setProperty("--regional-ad-layout-width", width);
      }
      if (slot.previousElementSibling !== chat) chat.after(slot);
      slot.dataset.position = "side";
    } else {
      game?.style.removeProperty("--regional-ad-layout-width");
      if (slot.dataset.position !== preferredPosition) {
        if (preferredPosition === "bottom") element("regionalAdBottomAnchor")?.before(slot);
        else element("regionalAdTopAnchor")?.after(slot);
        slot.dataset.position = preferredPosition;
      }
    }
  }

  function schedulePlacement() {
    if (placementFrame) return;
    placementFrame = window.requestAnimationFrame(() => {
      placementFrame = 0;
      updatePlacement();
    });
  }

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

  function dailyExposureState() {
    const date = japanDate();
    try {
      const saved = JSON.parse(localStorage.getItem(EXPOSURE_STORAGE_KEY) || "null");
      return saved?.date === date && saved.counts && typeof saved.counts === "object"
        ? saved
        : {date, counts:{}};
    } catch (_error) {
      return {date, counts:{}};
    }
  }

  function hasExposureCapacity(adId) {
    return Number(dailyExposureState().counts[adId] || 0) < DAILY_LIMIT;
  }

  function reserveExposure(adId, surface) {
    try {
      const state = dailyExposureState();
      const count = Number(state.counts[adId] || 0);
      if (count >= DAILY_LIMIT) {
        return {allowed:false, viewerFirst:false, surfaceViewerFirst:false};
      }
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

  function rotateStartingAd(ads, surface) {
    if (ads.length < 2) return ads;
    const adSet = ads.map(ad => ad.id).join(".");
    const key = `${ROTATION_STORAGE_PREFIX}:${surface}:${adSet}`;
    let start = 0;
    try {
      start = Math.abs(Number.parseInt(localStorage.getItem(key) || "0", 10)) % ads.length;
      localStorage.setItem(key, String((start + 1) % ads.length));
    } catch (_error) {}
    return [...ads.slice(start), ...ads.slice(0, start)];
  }

  function buildCarousel(ads) {
    const track = element("regionalAdTrack");
    const dots = element("regionalAdDots");
    if (!track || !dots) return;
    track.replaceChildren();
    dots.replaceChildren();

    ads.forEach((ad, index) => {
      const slide = document.createElement("article");
      slide.className = "regional-ad-slide";
      slide.dataset.adId = ad.id;
      slide.dataset.theme = backgroundThemes.has(ad.background_theme) ? ad.background_theme : "standard";
      slide.setAttribute("aria-hidden", String(index !== 0));

      const content = document.createElement(ad.url ? "a" : "div");
      content.className = "regional-ad-content";
      content.dataset.adId = ad.id;
      if (ad.url) {
        content.href = ad.url;
        content.target = "_blank";
        content.rel = "noopener noreferrer";
        content.tabIndex = index === 0 ? 0 : -1;
      }

      const title = document.createElement("strong");
      title.dataset.i18nIgnore = "";
      title.textContent = ad.title;
      const message = document.createElement("p");
      message.dataset.i18nIgnore = "";
      message.textContent = ad.message;
      const openMark = document.createElement("span");
      openMark.className = "regional-ad-open";
      openMark.setAttribute("aria-hidden", "true");
      openMark.textContent = "›";
      content.append(title, message, openMark);
      slide.appendChild(content);
      track.appendChild(slide);

      const dot = document.createElement("span");
      dot.className = `regional-ad-dot${index === 0 ? " active" : ""}`;
      dots.appendChild(dot);
    });
  }

  function recordVisibleAd() {
    const ad = carouselAds[currentIndex];
    if (!currentImpressionEligible || !ad || document.hidden === true || shownInContext.has(ad.id)) return;
    const exposure = reserveExposure(ad.id, currentSurface);
    shownInContext.add(ad.id);
    if (!exposure.allowed) return;
    recordMetric(ad.id, "impression", currentSurface, false, {
      viewer_measurement: true,
      viewer_first: exposure.viewerFirst,
      surface_viewer_first: exposure.surfaceViewerFirst,
    });
  }

  function showIndex(index, recordImpression = false) {
    if (!carouselAds.length) return;
    currentIndex = (index + carouselAds.length) % carouselAds.length;
    currentImpressionEligible = recordImpression;
    const track = element("regionalAdTrack");
    if (track) track.style.transform = `translateX(-${currentIndex * 100}%)`;
    track?.querySelectorAll(".regional-ad-slide").forEach((slide, slideIndex) => {
      const active = slideIndex === currentIndex;
      slide.setAttribute("aria-hidden", String(!active));
      const link = slide.querySelector("a");
      if (link) link.tabIndex = active ? 0 : -1;
    });
    element("regionalAdDots")?.querySelectorAll(".regional-ad-dot").forEach((dot, dotIndex) => {
      dot.classList.toggle("active", dotIndex === currentIndex);
    });
    recordVisibleAd();
  }

  function startRotation() {
    stopRotation();
    const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches === true;
    if (carouselAds.length < 2 || interactionPaused || document.hidden === true || reducedMotion) return;
    rotationTimer = window.setTimeout(() => {
      showIndex(currentIndex + 1, false);
      startRotation();
    }, AUTO_SLIDE_MS);
  }

  async function sync(surface, roomId, exposureId = "") {
    if (!surfaces.has(surface)) {
      requestNumber++;
      lastContext = "";
      hide();
      return;
    }
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
      const response = await fetch(
        `/api/regional-ad?surface=${encodeURIComponent(surface)}`,
        {credentials:"same-origin", cache:"no-store"},
      );
      if (!response.ok) throw new Error("regional ad unavailable");
      const data = await response.json();
      if (current !== requestNumber || lastContext !== context) return;
      applyPosition(data.position);
      const received = Array.isArray(data.ads) ? data.ads : (data.ad ? [data.ad] : []);
      const ads = received
        .filter(ad => ad && ad.id && ad.title && ad.message)
        .filter(ad => hasExposureCapacity(ad.id))
        .slice(0, 3);
      if (!ads.length || sessionStorage.getItem(`goitaRegionalAdDismissed:${surface}`)) {
        hide();
        return;
      }
      const nextKey = `${context}:${ads.map(ad => ad.id).join(",")}`;
      const adElement = element("regionalAd");
      if (!adElement.hidden && renderedKey === nextKey) return;

      stopRotation();
      currentSurface = surface;
      renderedKey = nextKey;
      shownInContext = new Set();
      carouselAds = rotateStartingAd(ads, surface);
      currentIndex = 0;
      buildCarousel(carouselAds);
      adElement.hidden = false;
      updatePlacement();
      showIndex(0, true);
      startRotation();
      window.goitaI18n?.refresh?.();
    } catch (_error) {
      if (current === requestNumber) hide();
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const adElement = element("regionalAd");
    const viewport = element("regionalAdViewport");
    const track = element("regionalAdTrack");
    window.addEventListener("resize", schedulePlacement);
    if (typeof ResizeObserver === "function") {
      const placementObserver = new ResizeObserver(schedulePlacement);
      [element("gameView"), element("chatPanel"), element("logCard"),
        document.querySelector(".play-main-column")].filter(Boolean).forEach(node => placementObserver.observe(node));
    }

    element("regionalAdClose")?.addEventListener("click", () => {
      sessionStorage.setItem(`goitaRegionalAdDismissed:${currentSurface}`, "1");
      hide();
    });
    track?.addEventListener("click", event => {
      if (Date.now() < suppressClickUntil) {
        event.preventDefault();
        return;
      }
      const link = event.target.closest("a[data-ad-id]");
      if (!link?.dataset.adId) return;
      recordMetric(link.dataset.adId, "click", currentSurface, true);
    });

    adElement?.addEventListener("mouseenter", () => {
      interactionPaused = true;
      stopRotation();
    });
    adElement?.addEventListener("mouseleave", () => {
      interactionPaused = false;
      startRotation();
    });
    adElement?.addEventListener("focusin", () => {
      interactionPaused = true;
      stopRotation();
    });
    adElement?.addEventListener("focusout", () => {
      window.setTimeout(() => {
        if (!adElement.contains(document.activeElement)) {
          interactionPaused = false;
          startRotation();
        }
      }, 0);
    });

    viewport?.addEventListener("touchstart", event => {
      touchStartX = event.changedTouches[0]?.clientX ?? null;
      interactionPaused = true;
      stopRotation();
    }, {passive:true});
    viewport?.addEventListener("touchend", event => {
      const endX = event.changedTouches[0]?.clientX ?? touchStartX;
      const distance = touchStartX == null ? 0 : endX - touchStartX;
      touchStartX = null;
      if (Math.abs(distance) >= 36 && carouselAds.length > 1) {
        showIndex(currentIndex + (distance < 0 ? 1 : -1), true);
        suppressClickUntil = Date.now() + 450;
      }
      interactionPaused = false;
      startRotation();
    }, {passive:true});

    document.addEventListener("visibilitychange", () => {
      if (document.hidden) stopRotation();
      else {
        recordVisibleAd();
        startRotation();
      }
    });
  });

  window.goitaRegionalAds = Object.freeze({sync});
})();
