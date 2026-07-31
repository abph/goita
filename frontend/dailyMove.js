(() => {
  "use strict";

  const CHALLENGE = Object.freeze({
    id: "0206-r6-t12",
    kifuId: "0206",
    round: 6,
    turn: 12,
    seat: "D",
    scores: { AC: 70, BD: 70 },
    hand: ["し", "し", "香"],
    board: {
      A: [
        { receive: "し", attack: "馬", faceDown: true, turn: 1 },
        { receive: "飛", attack: "馬", turn: 8 },
        { receive: "し", attack: "金", faceDown: true, turn: 9 },
      ],
      B: [
        { receive: "馬", attack: "銀", turn: 2 },
        { receive: "王", attack: "角", turn: 10 },
        { receive: "し", attack: "銀", faceDown: true, turn: 11 },
      ],
      C: [
        { receive: "銀", attack: "馬", turn: 3 },
        { receive: "し", attack: "金", faceDown: true, turn: 4 },
        { receive: "し", attack: "金", faceDown: true, turn: 5 },
      ],
      D: [
        { receive: "金", attack: "角", turn: 6 },
        { receive: "し", attack: "飛", faceDown: true, turn: 7 },
        { receive: "王", attack: "?", turn: 12, question: true },
      ],
    },
    correctChoice: "pawn",
  });

  const BOARD_SLOTS = Object.freeze({
    A: {
      receive: [[3, 7], [4, 7], [5, 7], [6, 7]],
      attack: [[3, 8], [4, 8], [5, 8], [6, 8]],
    },
    B: {
      receive: [[7, 6], [7, 5], [7, 4], [7, 3]],
      attack: [[8, 6], [8, 5], [8, 4], [8, 3]],
    },
    C: {
      receive: [[6, 2], [5, 2], [4, 2], [3, 2]],
      attack: [[6, 1], [5, 1], [4, 1], [3, 1]],
    },
    D: {
      receive: [[2, 3], [2, 4], [2, 5], [2, 6]],
      attack: [[1, 3], [1, 4], [1, 5], [1, 6]],
    },
  });

  const SEAT_LABELS = Object.freeze({
    A: { column: "4 / 6", row: "6", rotate: "0deg" },
    B: { column: "6", row: "4 / 6", rotate: "-90deg" },
    C: { column: "4 / 6", row: "3", rotate: "180deg" },
    D: { column: "3", row: "4 / 6", rotate: "90deg" },
  });

  const STORAGE_KEY = `goita-daily-move:${CHALLENGE.id}`;
  const PIECE_ENGLISH = Object.freeze({
    "し": "Pawn",
    "香": "Lance",
    "馬": "Knight",
    "銀": "Silver",
    "金": "Gold",
    "角": "Bishop",
    "飛": "Rook",
    "王": "King",
    "玉": "King",
  });

  const COPY = Object.freeze({
    ja: {
      eyebrow: "今日の一手",
      title: "この場面、あなたならどうする？",
      summary: "実戦で勝負を分けた一手です。Dの立場で考えてみましょう。",
      meta: "棋譜ID：0206　第6局・12ターン目",
      open: "挑戦する",
      review: "結果と解説を見る",
      modalTitle: "今日の一手",
      prompt: "Bが3つ目の攻めに銀を出し、Cはパスしました。Dは王で受けます。「？」には何を出しますか？",
      score: "開始時点",
      turnStatus: "Dの12手目",
      boardLabel: "12手目の盤面",
      answerLabel: "回答",
      centerTurn: "12手目",
      hand: "王で受けた後のDの手駒",
      question: "攻めの「？」に出す駒は？",
      pawn: "しを出す",
      lance: "香を出す",
      correctTitle: "実戦と同じ一手です！",
      otherTitle: "実戦では別の一手が選ばれました",
      actual: "実戦の一手",
      actualMove: "Dは王で銀を受け、しを出しました。",
      continuation: "続いてBがしで受け、香で20点上がり。BDは70対70の局面から90点へ進みました。",
      correctReason: "王で受けるのは、パスするとAが銀を受けて香で上がるためです。その後に香を出すと、Aが香で受けて銀で上がります。Aが受けられないしだけが、味方Bの上がりにつながります。",
      lanceReason: "香を出すと、Aが香を受けて銀で上がれます。実戦では、Aが受けられないしを選びました。",
      hiddenPiece: "伏せ駒",
      questionPiece: "攻める駒",
      retry: "別の手を試す",
      close: "閉じる",
      tried: "挑戦済み",
    },
    zh: {
      eyebrow: "今日一手",
      title: "这个局面，你会怎么走？",
      summary: "这是实战中决定胜负的一手。请从D的位置思考。",
      meta: "棋谱ID：0206　第6局・第12回合",
      open: "开始挑战",
      review: "查看结果与解说",
      modalTitle: "今日一手",
      prompt: "B第三次打出银，C选择跳过。D用王防守后，“？”处应该打出什么？",
      score: "当前比分",
      turnStatus: "D的第12手",
      boardLabel: "第12手的棋盘",
      answerLabel: "回答选项",
      centerTurn: "第12手",
      hand: "D用王防守后的手牌",
      question: "“？”处应该打出哪枚棋子？",
      pawn: "打出し",
      lance: "打出香",
      correctTitle: "与实战选择相同！",
      otherTitle: "实战选择了另一手",
      actual: "实战的一手",
      actualMove: "D用王防守银，然后打出し。",
      continuation: "随后B用し防守，并以香获得20分。BD从70比70前进到90分。",
      correctReason: "D用王防守，是因为跳过后A可以用银防守并以香结束。之后如果打出香，A会用香防守并以银结束。只有A无法防守的し，才能连接到队友B的胜利。",
      lanceReason: "如果打出香，A可以用香防守并以银结束。实战选择了A无法防守的し。",
      hiddenPiece: "暗置棋子",
      questionPiece: "进攻棋子",
      retry: "尝试其他选择",
      close: "关闭",
      tried: "已挑战",
    },
    en: {
      eyebrow: "Move of the Day",
      title: "What would you play here?",
      summary: "This move decided a real game. Make the decision from seat D.",
      meta: "Record 0206 · Round 6 · Turn 12",
      open: "Take the challenge",
      review: "View result and notes",
      modalTitle: "Move of the Day",
      prompt: "B made a third attack with Silver and C passed. D defends with King. What belongs on “?”?",
      score: "Current score",
      turnStatus: "D's move 12",
      boardLabel: "Position for move 12",
      answerLabel: "Answer choices",
      centerTurn: "Move 12",
      hand: "D's hand after defending with King",
      question: "Which piece should replace “?”?",
      pawn: "Play Pawn",
      lance: "Play Lance",
      correctTitle: "You found the move played in the game!",
      otherTitle: "The game used a different move",
      actual: "Move played",
      actualMove: "D defended the Silver with King, then attacked with Pawn.",
      continuation: "B then defended with Pawn and went out on Lance for 20 points. BD moved from 70-70 to 90 points.",
      correctReason: "D uses King because passing lets A defend with Silver and go out on Lance. After that defense, playing Lance lets A defend and go out on Silver. Pawn is the only attack A cannot stop, and it connects to B's win.",
      lanceReason: "Lance lets A defend with Lance and go out on Silver. The game chose Pawn, which A could not defend.",
      hiddenPiece: "Face-down piece",
      questionPiece: "Attack piece",
      retry: "Try another move",
      close: "Close",
      tried: "Completed",
    },
  });

  let selectedChoice = "";
  let previousFocus = null;

  function language() {
    const value = window.goitaI18n?.getLanguage?.() || document.documentElement.lang || "ja";
    return COPY[value] ? value : "ja";
  }

  function copy() {
    return COPY[language()];
  }

  function storedChoice() {
    try {
      const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
      if (!value || typeof value.choice !== "string") return "";
      if (value.choice === "king-pawn") return "pawn";
      if (value.choice === "king-lance") return "lance";
      return ["pawn", "lance"].includes(value.choice) ? value.choice : "";
    } catch (_error) {
      return "";
    }
  }

  function saveChoice(choice) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        choice,
        answeredAt: new Date().toISOString(),
      }));
    } catch (_error) {
      // The challenge remains playable when local storage is unavailable.
    }
  }

  function setText(id, value) {
    const element = document.getElementById(id);
    if (element) element.textContent = value;
  }

  function pieceElement(piece, options = {}) {
    const element = document.createElement("span");
    element.className = "daily-move-piece";
    element.classList.add(`phys-${options.seat || "A"}`);

    if (options.faceDown) {
      element.classList.add("is-hidden");
      element.setAttribute("aria-label", copy().hiddenPiece);
    } else if (piece === "?") {
      element.classList.add("is-question");
      element.textContent = piece;
      element.setAttribute("aria-label", copy().questionPiece);
    } else if (PIECE_ENGLISH[piece]) {
      element.textContent = piece;
      element.dataset.pieceEn = PIECE_ENGLISH[piece];
    } else {
      element.textContent = piece;
    }
    return element;
  }

  function placeBoardItem(board, element, position, className = "") {
    const cell = document.createElement("div");
    cell.className = `daily-move-cell ${className}`.trim();
    cell.style.gridColumn = String(position[0]);
    cell.style.gridRow = String(position[1]);
    cell.appendChild(element);
    board.appendChild(cell);
    return cell;
  }

  function renderBoard() {
    const board = document.getElementById("dailyMoveBoard");
    if (!board || board.dataset.rendered === "true") return;

    Object.entries(SEAT_LABELS).forEach(([seat, position]) => {
      const label = document.createElement("div");
      label.className = "daily-move-seat-label";
      label.style.gridColumn = position.column;
      label.style.gridRow = position.row;

      const inner = document.createElement("span");
      inner.textContent = seat;
      inner.style.transform = `rotate(${position.rotate})`;
      label.appendChild(inner);
      board.appendChild(label);
    });

    Object.entries(CHALLENGE.board).forEach(([seat, actions]) => {
      actions.forEach((action, index) => {
        const receivePosition = BOARD_SLOTS[seat].receive[index];
        const attackPosition = BOARD_SLOTS[seat].attack[index];
        placeBoardItem(
          board,
          pieceElement(action.receive, { seat, faceDown: action.faceDown }),
          receivePosition
        );
        const attackCell = placeBoardItem(
          board,
          pieceElement(action.attack, { seat }),
          attackPosition,
          action.question ? "is-question-cell" : ""
        );

        const number = document.createElement("span");
        number.className = "daily-move-turn-number";
        number.textContent = String(action.turn);
        attackCell.appendChild(number);
      });
    });

    const center = document.createElement("section");
    center.className = "daily-move-center-score";
    center.style.gridColumn = "4 / 6";
    center.style.gridRow = "4 / 6";
    const turn = document.createElement("strong");
    turn.id = "dailyMoveCenterTurn";
    const scoreAc = document.createElement("span");
    scoreAc.textContent = `AC ${CHALLENGE.scores.AC}`;
    const scoreBd = document.createElement("span");
    scoreBd.textContent = `BD ${CHALLENGE.scores.BD}`;
    center.append(turn, scoreAc, scoreBd);
    board.appendChild(center);

    const hand = document.getElementById("dailyMoveHand");
    CHALLENGE.hand.forEach((piece) => hand?.appendChild(pieceElement(piece)));
    board.dataset.rendered = "true";
  }

  function renderCopy() {
    const t = copy();
    setText("dailyMoveEyebrow", t.eyebrow);
    setText("dailyMoveCardTitle", t.title);
    setText("dailyMoveSummary", t.summary);
    setText("dailyMoveMeta", t.meta);
    setText("dailyMoveModalTitle", t.modalTitle);
    setText("dailyMovePrompt", t.prompt);
    setText("dailyMoveScore", `${t.score}: AC ${CHALLENGE.scores.AC} / BD ${CHALLENGE.scores.BD}`);
    setText("dailyMoveTurn", t.turnStatus);
    setText("dailyMoveCenterTurn", t.centerTurn);
    setText("dailyMoveHandLabel", t.hand);
    setText("dailyMoveQuestionTitle", t.question);
    setText("dailyMoveChoicePawn", t.pawn);
    setText("dailyMoveChoiceLance", t.lance);
    setText("dailyMoveRetryButton", t.retry);

    const closeButton = document.getElementById("dailyMoveCloseButton");
    if (closeButton) {
      closeButton.setAttribute("aria-label", t.close);
      closeButton.title = t.close;
    }
    document.getElementById("dailyMoveBoard")?.setAttribute("aria-label", t.boardLabel);
    document.getElementById("dailyMoveChoices")?.setAttribute("aria-label", t.answerLabel);
    document.querySelectorAll(".daily-move-piece.is-hidden").forEach((piece) => {
      piece.setAttribute("aria-label", t.hiddenPiece);
    });
    document.querySelectorAll(".daily-move-piece.is-question").forEach((piece) => {
      piece.setAttribute("aria-label", t.questionPiece);
    });

    const completed = Boolean(storedChoice());
    setText("dailyMoveOpenButton", completed ? t.review : t.open);
    const card = document.getElementById("dailyMoveCard");
    card?.classList.toggle("is-completed", completed);
    card?.setAttribute("aria-label", completed ? `${t.eyebrow}: ${t.tried}` : t.eyebrow);

    if (selectedChoice) renderResult(selectedChoice, false);
  }

  function resultReason(choice, t) {
    if (choice === "lance") return t.lanceReason;
    return t.correctReason;
  }

  function renderResult(choice, shouldFocus = true) {
    const result = document.getElementById("dailyMoveResult");
    if (!result) return;
    const resultActions = document.getElementById("dailyMoveResultActions");
    const t = copy();
    const correct = choice === CHALLENGE.correctChoice;
    result.hidden = false;
    if (resultActions) resultActions.hidden = false;
    result.classList.toggle("is-correct", correct);
    result.innerHTML = "";

    const title = document.createElement("div");
    title.className = "daily-move-result-title";
    title.textContent = correct ? t.correctTitle : t.otherTitle;
    result.appendChild(title);

    const actual = document.createElement("p");
    const actualLabel = document.createElement("strong");
    actualLabel.textContent = `${t.actual}: `;
    actual.append(actualLabel, t.actualMove);
    result.appendChild(actual);

    const continuation = document.createElement("p");
    continuation.textContent = t.continuation;
    result.appendChild(continuation);

    const reason = document.createElement("p");
    reason.textContent = resultReason(choice, t);
    result.appendChild(reason);

    document.querySelectorAll("[data-daily-move-choice]").forEach((button) => {
      button.classList.toggle("is-selected", button.dataset.dailyMoveChoice === choice);
    });

    if (shouldFocus) {
      result.setAttribute("tabindex", "-1");
      result.focus({ preventScroll: false });
    }
  }

  function answer(choice) {
    if (!["pawn", "lance"].includes(choice)) return;
    selectedChoice = choice;
    saveChoice(choice);
    renderResult(choice);
    renderCopy();
  }

  function open() {
    const modal = document.getElementById("dailyMoveModal");
    if (!modal) return;
    previousFocus = document.activeElement;
    selectedChoice = storedChoice();
    renderBoard();
    renderCopy();
    if (selectedChoice) renderResult(selectedChoice, false);
    else {
      const result = document.getElementById("dailyMoveResult");
      if (result) result.hidden = true;
      const resultActions = document.getElementById("dailyMoveResultActions");
      if (resultActions) resultActions.hidden = true;
    }
    modal.style.display = "flex";
    document.body.style.overflow = "hidden";
    document.getElementById("dailyMoveCloseButton")?.focus();
  }

  function close(event) {
    if (event && event.target !== event.currentTarget) return;
    const modal = document.getElementById("dailyMoveModal");
    if (!modal) return;
    modal.style.display = "none";
    document.body.style.removeProperty("overflow");
    previousFocus?.focus?.();
  }

  function retry() {
    selectedChoice = "";
    const result = document.getElementById("dailyMoveResult");
    if (result) result.hidden = true;
    const resultActions = document.getElementById("dailyMoveResultActions");
    if (resultActions) resultActions.hidden = true;
    document.querySelectorAll("[data-daily-move-choice]").forEach((button) => {
      button.classList.remove("is-selected");
    });
    document.getElementById("dailyMoveChoicePawn")?.focus();
  }

  function initialize() {
    renderBoard();
    renderCopy();
    document.getElementById("dailyMoveOpenButton")?.addEventListener("click", open);
    document.getElementById("dailyMoveCloseButton")?.addEventListener("click", () => close());
    document.getElementById("dailyMoveModal")?.addEventListener("click", close);
    document.getElementById("dailyMoveModalContent")?.addEventListener("click", (event) => event.stopPropagation());
    document.getElementById("dailyMoveRetryButton")?.addEventListener("click", retry);
    document.querySelectorAll("[data-daily-move-choice]").forEach((button) => {
      button.addEventListener("click", () => answer(button.dataset.dailyMoveChoice));
    });
    window.addEventListener("goita-language-change", renderCopy);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && document.getElementById("dailyMoveModal")?.style.display === "flex") {
        close();
      }
    });
  }

  window.goitaDailyMove = Object.freeze({
    open,
    close,
    answer,
    retry,
    challenge: CHALLENGE,
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
