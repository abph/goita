(() => {
  "use strict";

  const SHOWN_DAY_KEY = "goita-solo-discord-prompt-day";
  const ROOM_TYPES = new Set(["public_human", "public_ai", "private"]);
  const SEATS = new Set(["A", "B", "C", "D"]);
  let shownDay = "";
  let activeKey = "";
  let latest = null;

  const element = id => document.getElementById(id);
  const japanDate = () => new Date(Date.now() + 9 * 60 * 60 * 1000).toISOString().slice(0, 10);

  function hide() {
    const panel = element("soloDiscordPrompt");
    if (panel) panel.hidden = true;
    activeKey = "";
  }

  function reset() {
    latest = null;
    hide();
  }

  function alreadyShown(day) {
    if (shownDay === day) return true;
    try { return localStorage.getItem(SHOWN_DAY_KEY) === day; }
    catch (_error) { return false; }
  }

  function sync(state, context = {}) {
    latest = {state, context};
    const panel = element("soloDiscordPrompt");
    if (!panel) return;
    const humans = new Set(Array.isArray(state?.human_seats) ? state.human_seats : []);
    const ais = new Set(Array.isArray(state?.ai_seats) ? state.ai_seats : []);
    const owned = Array.isArray(state?.owned_human_seats) ? state.owned_human_seats : [];
    const round = Number(state?.round_count || 0);
    const qualifies = context.normalRoom === true && ROOM_TYPES.has(context.roomType)
      && context.roomId && SEATS.has(context.seat) && owned.includes(context.seat)
      && state?.is_started === true && state.finished === true && Number.isFinite(round) && round >= 2
      && !state.trace_mode && !state.tutorial_mode && !state.practice_replay_active
      && humans.size === 1 && humans.has(context.seat) && ais.size === 3
      && [...ais].every(seat => SEATS.has(seat) && !humans.has(seat));
    if (!qualifies) { hide(); return; }
    const key = `${context.roomId}:${round}:${context.seat}`;
    if (!panel.hidden && activeKey === key) return;
    hide();
    const day = japanDate();
    if (document.hidden === true || alreadyShown(day)) return;
    shownDay = day;
    try { localStorage.setItem(SHOWN_DAY_KEY, day); } catch (_error) {}
    activeKey = key;
    panel.hidden = false;
    window.trackAnalytics?.("discord_solo_prompt_shown", {}, context.roomType);
  }

  document.addEventListener("DOMContentLoaded", () => {
    element("soloDiscordDismiss")?.addEventListener("click", hide);
    element("soloDiscordOpen")?.addEventListener("click", () => {
      if (element("soloDiscordPrompt")?.hidden !== false || !latest) return;
      const roomType = latest.context.roomType;
      window.trackAnalytics?.("discord_solo_prompt_clicked", {}, roomType);
      window.trackAnalytics?.("discord_info_opened", {}, roomType);
      window.openSiteInfo?.("discord");
      hide();
    });
    document.addEventListener("visibilitychange", () => {
      if (document.hidden !== true && latest) sync(latest.state, latest.context);
    });
    window.addEventListener("storage", event => {
      if (event.key === SHOWN_DAY_KEY && event.newValue === japanDate()) hide();
    });
  });

  window.goitaSoloDiscord = Object.freeze({sync, reset});
})();
