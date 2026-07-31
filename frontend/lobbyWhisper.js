(() => {
  "use strict";

  const MESSAGES = Object.freeze([
    "こんにちは！",
    "最近の広告は姑息すぎて",
    "その類の広告は絶滅すべきだと思います！",
  ]);
  const MESSAGE_HOLD_MS = Object.freeze([1150, 1650]);
  const SURPRISE_MESSAGE = "なにもありませんよ笑";
  const MILESTONE_MESSAGE = "100回目おめでとう。そんな暇なあなたには、プライベートルームを一つ、1年間、授けます。希望するなら、連絡をください。";
  const EXIT_ANIMATION_MS = 220;
  let sequenceTimer = null;
  let exitTimer = null;
  let dismissed = false;
  let surpriseClickCount = 0;

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
    message.textContent = MESSAGES[index];
    refreshTranslation();
    void message.offsetWidth;
    message.classList.add("is-entering");

    if (index >= MESSAGES.length - 1) return;

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
    if (!message || !whisper || whisper.hidden || dismissed) return;

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
    dismissed = true;
    clearTimers();
    whisper.hidden = true;
  }

  function setRoomVisibility(isPublicRoom) {
    const whisper = document.getElementById("lobbyWhisper");
    if (!whisper) return;
    if (!isPublicRoom || dismissed) {
      clearTimers();
      whisper.hidden = true;
      return;
    }
    whisper.hidden = false;
    startSequence();
  }

  function initialize() {
    const whisper = document.getElementById("lobbyWhisper");
    const closeButton = document.getElementById("lobbyWhisperClose");
    const message = messageElement();
    if (!whisper || !closeButton || !message) return;

    whisper.hidden = true;
    whisper.addEventListener("click", showSurpriseMessage);
    closeButton.addEventListener("click", (event) => {
      event.stopPropagation();
      dismiss();
    });
  }

  window.goitaLobbyWhisper = Object.freeze({
    dismiss,
    setRoomVisibility,
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
