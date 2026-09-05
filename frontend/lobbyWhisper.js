(() => {
  "use strict";

  const PUBLIC_MESSAGES = Object.freeze([
    "こんにちは！",
    "最近の広告は姑息すぎて、ほんと嫌ですね。",
  ]);
  const MESSAGE_HOLD_MS = Object.freeze([1150]);
  const SURPRISE_MESSAGE = "なにもありませんよ笑";
  const MILESTONE_MESSAGE = "100回目おめでとう。そんな暇なあなたには、プライベートルームを一つ、1年間、授けます。希望するなら、連絡をください。";
  const EXIT_ANIMATION_MS = 220;
  let sequenceTimer = null;
  let exitTimer = null;
  let surpriseClickCount = 0;
  let activeMessages = PUBLIC_MESSAGES;
  let activeUrl = "";
  let activeContextKey = "";
  let activePublicMessage = false;
  const dismissedContexts = new Set();

  function messageElement() {
    return document.getElementById("lobbyWhisperMessage");
  }

  function refreshTranslation() {
    window.goitaI18n?.refresh?.();
  }

  function clearTimers() {
    window.clearTimeout(sequenceTimer);
    window.clearTimeout(exitTimer);
    sequenceTimer = null;
    exitTimer = null;
  }

  function showMessage(index) {
    const message = messageElement();
    const whisper = document.getElementById("lobbyWhisper");
    if (!message || !whisper || whisper.hidden) return;

    message.classList.remove("is-entering", "is-leaving");
    message.textContent = activeMessages[index];
    refreshTranslation();
    void message.offsetWidth;
    message.classList.add("is-entering");

    if (index >= activeMessages.length - 1) return;

    sequenceTimer = window.setTimeout(() => {
      message.classList.remove("is-entering");
      message.classList.add("is-leaving");
      exitTimer = window.setTimeout(() => showMessage(index + 1), EXIT_ANIMATION_MS);
    }, MESSAGE_HOLD_MS[index]);
  }

  function startSequence() {
    clearTimers();
    showMessage(0);
  }

  function surpriseMessage() {
    if (surpriseClickCount >= 100) return MILESTONE_MESSAGE;
    if (surpriseClickCount >= 10) {
      return `なにもありませんよ（笑）${surpriseClickCount}回目`;
    }
    return SURPRISE_MESSAGE;
  }

  function showSurpriseMessage() {
    const message = messageElement();
    const whisper = document.getElementById("lobbyWhisper");
    if (!message || !whisper || whisper.hidden || !activePublicMessage) return;

    clearTimers();
    surpriseClickCount = Math.min(surpriseClickCount + 1, 100);
    message.classList.remove("is-entering", "is-leaving");
    void message.offsetWidth;
    message.classList.add("is-leaving");
    exitTimer = window.setTimeout(() => {
      message.classList.remove("is-leaving");
      message.textContent = surpriseMessage();
      refreshTranslation();
      void message.offsetWidth;
      message.classList.add("is-entering");
    }, EXIT_ANIMATION_MS);
  }

  function dismiss() {
    const whisper = document.getElementById("lobbyWhisper");
    if (!whisper) return;
    if (activeContextKey) dismissedContexts.add(activeContextKey);
    clearTimers();
    whisper.hidden = true;
  }

  function setRoomContext(isPublicRoom, privateAd = null, publicAd = null) {
    const whisper = document.getElementById("lobbyWhisper");
    const message = messageElement();
    const label = document.querySelector("#lobbyWhisper .lobby-whisper-label");
    if (!whisper || !message || !label) return;

    const ad = isPublicRoom ? publicAd : privateAd;
    const specialPublic = isPublicRoom && (publicAd === null || (publicAd.enabled === true && publicAd.mode === "whisper"));
    const customMessage = String(ad?.message || "").trim();
    const customLabel = String(ad?.label || "お知らせ").trim() || "お知らせ";
    const customEnabled = !specialPublic && ad?.enabled === true && customMessage;
    const contextPrefix = isPublicRoom ? `public:${publicAd?.room_id || ""}` : "private";
    const nextContextKey = specialPublic
      ? `${contextPrefix}:whisper`
      : (customEnabled ? `${contextPrefix}:${customLabel}:${customMessage}:${String(ad?.url || "")}` : "");
    if (!nextContextKey) {
      clearTimers();
      whisper.hidden = true;
      activeContextKey = "";
      return;
    }

    if (nextContextKey === activeContextKey && !whisper.hidden) return;
    activeContextKey = nextContextKey;
    activePublicMessage = specialPublic;
    activeMessages = specialPublic ? PUBLIC_MESSAGES : [customMessage];
    activeUrl = specialPublic ? "" : String(ad?.url || "").trim();
    label.textContent = specialPublic ? "1222のつぶやき" : customLabel;
    whisper.setAttribute("aria-label", label.textContent);
    whisper.classList.toggle("has-link", Boolean(activeUrl));
    whisper.classList.toggle("is-interactive-message", activePublicMessage);
    if (specialPublic) {
      label.removeAttribute("data-i18n-ignore");
      message.removeAttribute("data-i18n-ignore");
    } else {
      label.setAttribute("data-i18n-ignore", "");
      message.setAttribute("data-i18n-ignore", "");
    }
    refreshTranslation();

    if (dismissedContexts.has(activeContextKey)) {
      clearTimers();
      whisper.hidden = true;
      return;
    }
    whisper.hidden = false;
    startSequence();
  }

  function setRoomVisibility(isPublicRoom) {
    // Wait for the room's settings so a disabled/custom notice never flashes the special message.
    setRoomContext(isPublicRoom, null, {enabled: false});
  }

  function activate() {
    if (activeUrl) {
      window.open(activeUrl, "_blank", "noopener,noreferrer");
      return;
    }
    showSurpriseMessage();
  }

  function initialize() {
    const whisper = document.getElementById("lobbyWhisper");
    const closeButton = document.getElementById("lobbyWhisperClose");
    const message = messageElement();
    if (!whisper || !closeButton || !message) return;

    whisper.hidden = true;
    whisper.addEventListener("click", activate);
    closeButton.addEventListener("click", (event) => {
      event.stopPropagation();
      dismiss();
    });
  }

  window.goitaLobbyWhisper = Object.freeze({
    dismiss,
    setRoomContext,
    setRoomVisibility,
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
