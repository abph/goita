(() => {
  "use strict";

  const STORAGE_KEY = "goita_survey_response_key_v1";
  const DONE_KEY = "goita_survey_completed_v1";
  const quickQuestions = [
    ["satisfaction", "そろうごいたを遊んでみて、どう感じましたか？", [["very_satisfied","とても満足"],["satisfied","満足"],["neutral","どちらともいえない"],["dissatisfied","少し不満"],["very_dissatisfied","不満"]]],
    ["primary_use", "主にどのように遊んでいますか？", [["public","公開部屋で人と対局"],["ai","AIと対局"],["private","友人とプライベートルームで対局"],["research","手駒や場面を指定して研究・練習"],["score_attack","スコアアタック"],["spectate","観戦"],["new","まだほとんど遊んでいない"]]],
    ["improvement", "最も改善してほしいところはどこですか？", [["ai","AIの強さや打ち方"],["matching","対局相手の見つけやすさ"],["usability","画面や操作の分かりやすさ"],["research","研究・練習機能"],["kifu","棋譜ライブラリ"],["score_attack","スコアアタック"],["stability","通信の安定性や動作速度"],["beginner","初心者向けの説明"],["none","特にない"],["other","その他"]]],
  ];
  const detailedSingles = [
    ["frequency", "そろうごいたを、どのくらい利用していますか？", [["daily","ほぼ毎日"],["weekly","週に数回"],["monthly","月に数回"],["few","これまでに数回"],["first","今日が初めて"]]],
    ["experience", "そろうごいた以外での、ごいた経験を教えてください", [["first","そろうごいたで初めて知った"],["few","数回遊んだことがある"],["sometimes","ときどき遊んでいる"],["regular","日常的に遊んでいる"],["tournament","大会などに参加したことがある"]]],
  ];
  const playStyles = [["public","公開部屋で人と対局する"],["ai","AIと対局する"],["private","友人とプライベートルームで遊ぶ"],["research","配碑や手札を指定して研究・練習する"],["score_attack","スコアアタックで遊ぶ"],["spectate","対局を観戦する"],["kifu","棋譜を保存して振り返る"],["new","まだほとんど遊んでいない"]];
  const features = [
    ["library","棋譜ライブラリ","保存した棋譜を見たり、整理したりできる機能"],
    ["auto_save","棋譜の自動保存","参加した局の棋譜を自動的に保存する機能"],
    ["statistics","保存棋譜の統計","保存した棋譜から勝率や得失点などを確認する機能"],
    ["deal_style","手駒の配り方を選ぶ","「よくある手駒」など、配碑の傾向を指定する機能"],
    ["balanced_deal","極端な手駒を除いて配る","強すぎる手駒と弱すぎる手駒を除く配碑設定"],
    ["hand_spec","手札を指定する","「きっちり指定」と「大雑把に指定」で手札を作る機能"],
    ["replay_practice","同じ局や特定の場面から練習する","「もう一度練習する」と「場面を指定して練習する」機能"],
  ];
  const sets = {
    confusing_areas: [["top","トップページ"],["room_entry","対局ルームへの入り方"],["game","対局中の操作"],["settings","対局設定"],["kifu","棋譜ライブラリ"],["score_attack","スコアアタック"],["account","会員登録・ログイン"],["mobile","スマートフォンでの表示"],["other","その他"]],
    ai_concerns: [["pass","不自然なパスがある"],["receive","不自然な受け方がある"],["attack","攻め方が不自然"],["repeats","同じような手を繰り返す"],["slow","思考時間が長い"],["shallow","思考時間が短く、十分に考えていないように見える"],["none","特に気になることはない"],["other","その他"]],
    score_good: [["comparison","元の棋譜と結果を比較できる"],["same_hand","同じ手駒で挑戦できる"],["ranking","ランキングで競える"],["rewards","報酬がある"],["solo","1人で遊べる"],["history","棋譜履歴を残せる"],["none","特にない"],["other","その他"]],
    score_improvements: [["rules","遊び方が分かりにくい"],["difference","元の棋譜との違いが分かりにくい"],["ai","AIの動きが不自然"],["tempo","対局のテンポが悪い"],["ranking","ランキングが分かりにくい"],["rewards","報酬を増やしてほしい"],["choose_record","挑戦する棋譜を選びたい"],["none","特にない"],["other","その他"]],
    future_features: [["kifu_analysis","棋譜の詳しい分析"],["ai_advice","AIによる対局後の助言"],["beginner","初心者向けの説明や練習"],["stronger_ai","AIのさらなる強化"],["matching","対人相手を見つけやすくする機能"],["events","大会・イベント機能"],["score_content","スコアアタックの種類や報酬の追加"],["profile","成績・称号・プロフィール機能"],["research","研究・練習機能の追加"],["mobile","スマートフォン画面の改善"],["other","その他"]],
    problems: [["matching","対局相手が見つからない"],["room_entry","ルームへの入り方が分からない"],["settings","対局設定が分からない"],["game","対局中の操作が分からない"],["find_features","機能がどこにあるか分からない"],["account","会員登録やログインが分からない"],["kifu","棋譜の保存・確認方法が分からない"],["connection","通信が切れたり、動作が遅くなったりする"],["mobile","スマートフォンで操作しにくい"],["none","特に困ったことはない"],["other","その他"]],
  };

  const t = source => window.goitaI18n?.translate?.(source) || source;
  const esc = value => String(value).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"})[c]);
  const optionHtml = (name, values, type="radio", required=false) => `<div class="survey-options">${values.map(([value,label], index) => `<label><input type="${type}" name="${esc(name)}" value="${esc(value)}" ${required && index === 0 ? "required" : ""}><span>${esc(t(label))}</span></label>`).join("")}</div>`;
  const fieldset = (name, legend, values, type="radio", required=false) => `<fieldset class="survey-question"><legend>${esc(t(legend))}</legend>${optionHtml(name, values, type, required)}</fieldset>`;

  function responseKey() {
    let value = localStorage.getItem(STORAGE_KEY) || "";
    if (!/^[A-Za-z0-9_-]{16,80}$/.test(value)) {
      value = typeof crypto.randomUUID === "function"
        ? crypto.randomUUID().replaceAll("-", "")
        : Array.from(crypto.getRandomValues(new Uint8Array(20)), byte => byte.toString(16).padStart(2,"0")).join("");
      localStorage.setItem(STORAGE_KEY, value);
    }
    return value;
  }
  function device() {
    const width = Math.min(screen.width || innerWidth, innerWidth || screen.width);
    return width <= 700 ? "mobile" : (width <= 1000 ? "tablet" : "desktop");
  }
  function language() {
    const value = window.goitaI18n?.getLanguage?.() || "ja";
    return ["ja","zh","en"].includes(value) ? value : "other";
  }
  function dialog() { return document.getElementById("surveyDialog"); }
  function body() { return document.getElementById("surveyBody"); }

  function ensureDialog() {
    if (dialog()) return;
    const element = document.createElement("dialog");
    element.id = "surveyDialog";
    element.className = "survey-dialog";
    element.innerHTML = `<div class="survey-shell"><header class="survey-header"><h2>${esc(t("そろうごいた改善アンケート"))}</h2><button class="survey-close" type="button" aria-label="${esc(t("閉じる"))}">×</button></header><div id="surveyBody"></div></div>`;
    document.body.appendChild(element);
    element.querySelector(".survey-close").addEventListener("click", () => element.close());
    element.addEventListener("click", event => { if (event.target === element) element.close(); });
  }

  function showChooser() {
    body().innerHTML = `<p class="survey-lead">${esc(t("回答時間を選べます。名前や会員IDは回答内容に保存しません。"))}</p><div class="survey-kind-grid"><button class="survey-kind" type="button" data-kind="quick"><strong>${esc(t("30秒で回答する"))}</strong><span>${esc(t("3問のかんたんなアンケート"))}</span></button><button class="survey-kind" type="button" data-kind="detailed"><strong>${esc(t("詳しく回答する"))}</strong><span>${esc(t("約3分のアンケート"))}</span></button></div>`;
    body().querySelectorAll("[data-kind]").forEach(button => button.addEventListener("click", () => renderForm(button.dataset.kind)));
  }

  function renderQuick() {
    return `<form id="surveyForm" class="survey-form" data-kind="quick"><section class="survey-section"><h3>${esc(t("かんたんアンケート"))}</h3>${quickQuestions.map(q => fieldset(q[0],q[1],q[2],"radio",true)).join("")}<div id="quickOther" class="survey-conditional" hidden><label>${esc(t("その他（任意）"))}<input class="survey-short-text" name="improvement_other" maxlength="200"></label></div></section>${actions()}</form>`;
  }

  function featureHtml() {
    const choices = [["unknown","知らなかった"],["known","知っているが、使ったことはない"],["used","使ったことがある"]];
    return features.map(([key,title,description]) => `<div class="survey-feature"><strong>${esc(t(title))}</strong><small>${esc(t(description))}</small>${optionHtml(`feature_${key}`,choices,"radio",true)}</div>`).join("");
  }
  function actions() {
    return `<div class="survey-status" role="status" aria-live="polite"></div><div class="survey-actions"><button class="survey-back" type="button">${esc(t("種類を選び直す"))}</button><button class="survey-submit" type="submit">${esc(t("回答を送信する"))}</button></div>`;
  }
  function renderDetailed() {
    const clarity = [["very_clear","とても分かりやすい"],["clear","分かりやすい"],["neutral","どちらともいえない"],["unclear","少し分かりにくい"],["very_unclear","とても分かりにくい"]];
    const aiStrength = [["too_strong","強すぎる"],["strong","少し強い"],["right","ちょうどよい"],["weak","少し弱い"],["too_weak","弱すぎる"],["unknown","判断できない"]];
    const natural = [["very_natural","とても自然"],["natural","自然"],["neutral","どちらともいえない"],["unnatural","少し不自然"],["very_unnatural","とても不自然"]];
    const scoreUsage = [["regular","定期的に遊んでいる"],["sometimes","ときどき遊んでいる"],["once","1回だけ遊んだ"],["known_unused","知っているが、遊んだことはない"],["unknown","知らなかった"]];
    const enjoyment = [["very_fun","とても楽しめた"],["fun","楽しめた"],["neutral","どちらともいえない"],["not_fun","あまり楽しめなかった"],["very_not_fun","楽しめなかった"]];
    const membership = [["understand","内容まで知っている"],["name_only","無料会員があることだけ知っている"],["unknown","知らなかった"],["registered","すでに登録している"]];
    return `<form id="surveyForm" class="survey-form" data-kind="detailed">
      <section class="survey-section"><h3>${esc(t("1. 利用状況"))}</h3>${detailedSingles.map(q=>fieldset(q[0],q[1],q[2],"radio",true)).join("")}${fieldset("play_styles","普段は、どのように遊んでいますか？",playStyles,"checkbox")}</section>
      <section class="survey-section"><h3>${esc(t("2. 棋譜・研究機能"))}</h3>${featureHtml()}${fieldset("membership_awareness","無料会員の機能を知っていますか？",membership,"radio",true)}<p class="survey-limit">${esc(t("無料会員は、棋譜を最大20局まで自分専用のライブラリに保存できます。"))}</p></section>
      <section class="survey-section"><h3>${esc(t("3. 画面・AI・スコアアタック"))}</h3>${fieldset("ui_clarity","画面や操作は分かりやすいですか？",clarity,"radio",true)}<div id="clarityMore" class="survey-conditional" hidden>${fieldset("confusing_areas","どこが分かりにくかったですか？",sets.confusing_areas,"checkbox")}</div>${fieldset("ai_used","AIと対局したことがありますか？",[["yes","ある"],["no","ない"]],"radio",true)}<div id="aiMore" class="survey-conditional" hidden>${fieldset("ai_strength","AIの強さをどう感じますか？",aiStrength,"radio")}${fieldset("ai_naturalness","AIの打ち方は、人間らしく自然だと思いますか？",natural,"radio")}${fieldset("ai_concerns","AIについて気になることはありますか？",sets.ai_concerns,"checkbox")}<label>${esc(t("AIについての補足（任意）"))}<input class="survey-short-text" name="ai_other" maxlength="200"></label></div>${fieldset("score_usage","スコアアタックについて教えてください",scoreUsage,"radio",true)}<div id="scoreMore" class="survey-conditional" hidden>${fieldset("score_enjoyment","スコアアタックを楽しめましたか？",enjoyment,"radio")}${fieldset("score_good","良いと思ったところを教えてください",sets.score_good,"checkbox")}${fieldset("score_improvements","改善してほしいところを教えてください",sets.score_improvements,"checkbox")}</div></section>
      <section class="survey-section"><h3>${esc(t("4. 今後の改善"))}</h3>${fieldset("future_features","今後、特にほしいものを選んでください（3つまで）",sets.future_features,"checkbox")}${fieldset("problems","利用中に困ったことはありますか？",sets.problems,"checkbox")}<label><strong>${esc(t("ご意見やご要望があれば、自由にお書きください"))}</strong><textarea class="survey-text" name="free_text" maxlength="2000" placeholder="${esc(t("気に入っているところ、直してほしいところ、不自然だと感じたAIの手など、どのような内容でも構いません。"))}"></textarea></label></section>${actions()}</form>`;
  }

  function renderForm(kind) {
    body().innerHTML = kind === "quick" ? renderQuick() : renderDetailed();
    const form = document.getElementById("surveyForm");
    form.querySelector(".survey-back").addEventListener("click", showChooser);
    form.addEventListener("change", updateConditional);
    form.addEventListener("submit", submit);
    updateConditional();
  }
  function selected(form, name) { return form.querySelector(`[name="${name}"]:checked`)?.value || ""; }
  function updateConditional() {
    const form = document.getElementById("surveyForm"); if (!form) return;
    const quickOther = document.getElementById("quickOther");
    if (quickOther) quickOther.hidden = selected(form,"improvement") !== "other";
    const clarityMore = document.getElementById("clarityMore");
    if (clarityMore) clarityMore.hidden = !["unclear","very_unclear"].includes(selected(form,"ui_clarity"));
    const aiMore = document.getElementById("aiMore");
    if (aiMore) aiMore.hidden = selected(form,"ai_used") !== "yes";
    const scoreMore = document.getElementById("scoreMore");
    if (scoreMore) scoreMore.hidden = !["regular","sometimes","once"].includes(selected(form,"score_usage"));
    const future = [...form.querySelectorAll('[name="future_features"]:checked')];
    form.querySelectorAll('[name="future_features"]:not(:checked)').forEach(input => { input.disabled = future.length >= 3; });
  }
  function values(form, name) { return [...form.querySelectorAll(`[name="${name}"]:checked`)].map(input => input.value); }
  function answers(form) {
    if (form.dataset.kind === "quick") return {
      satisfaction:selected(form,"satisfaction"), primary_use:selected(form,"primary_use"), improvement:selected(form,"improvement"), improvement_other:form.elements.improvement_other?.value || "",
    };
    const feature_awareness = Object.fromEntries(features.map(([key]) => [key,selected(form,`feature_${key}`)]));
    return {
      frequency:selected(form,"frequency"), experience:selected(form,"experience"), play_styles:values(form,"play_styles"), feature_awareness,
      membership_awareness:selected(form,"membership_awareness"), ui_clarity:selected(form,"ui_clarity"), confusing_areas:values(form,"confusing_areas"),
      ai_used:selected(form,"ai_used"), ai_strength:selected(form,"ai_strength"), ai_naturalness:selected(form,"ai_naturalness"), ai_concerns:values(form,"ai_concerns"), ai_other:form.elements.ai_other?.value || "",
      score_usage:selected(form,"score_usage"), score_enjoyment:selected(form,"score_enjoyment"), score_good:values(form,"score_good"), score_improvements:values(form,"score_improvements"),
      future_features:values(form,"future_features"), problems:values(form,"problems"), free_text:form.elements.free_text?.value || "",
    };
  }
  async function submit(event) {
    event.preventDefault(); const form=event.currentTarget; const status=form.querySelector(".survey-status"); const button=form.querySelector(".survey-submit");
    const data=answers(form);
    if (form.dataset.kind === "detailed" && !data.play_styles.length) { status.textContent=t("普段の遊び方を1つ以上選んでください。"); return; }
    button.disabled=true; status.textContent=t("送信しています...");
    try {
      const response=await fetch("/api/survey/responses",{method:"POST",credentials:"same-origin",headers:{"Content-Type":"application/json","X-Goita-Member":"1"},body:JSON.stringify({response_key:responseKey(),kind:form.dataset.kind,device:device(),language:language(),answers:data})});
      const result=await response.json().catch(()=>({})); if(!response.ok) throw new Error(result.detail||t("回答を送信できませんでした。"));
      localStorage.setItem(DONE_KEY,result.kind||form.dataset.kind);
      body().innerHTML=`<div class="survey-thanks"><h3>${esc(t("回答ありがとうございました。"))}</h3><p>${esc(t("いただいた回答は、今後の改善に利用します。"))}</p>${form.dataset.kind==="quick"?`<button class="survey-submit" type="button" data-detailed>${esc(t("詳しいアンケートにも回答する"))}</button>`:""}</div>`;
      body().querySelector("[data-detailed]")?.addEventListener("click",()=>renderForm("detailed"));
    } catch(error) { status.textContent=error.message; button.disabled=false; }
  }
  function open() { ensureDialog(); showChooser(); if (!dialog().open) dialog().showModal(); }
  window.goitaSurvey=Object.freeze({open,completed:()=>localStorage.getItem(DONE_KEY)||""});
})();
