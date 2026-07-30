const canvas = document.getElementById("boardPixelCanvas");
const container = document.getElementById("boardPixel");
const context = canvas?.getContext("2d");

const SIZE = 320;
const GRID_ORIGIN = 12;
const CELL = 37;
const PIECE_WIDTH = 30;
const PIECE_HEIGHT = 34;
const PIECE_ROTATION = {
  A: 0,
  B: -Math.PI / 2,
  C: Math.PI,
  D: Math.PI / 2,
};

let visible = false;
let latestSnapshot = null;

const palettes = {
  color: {
    outer: "#263d35",
    frameDark: "#71462a",
    frame: "#b87543",
    board: "#efc57d",
    boardLight: "#f7d99d",
    boardDark: "#e7b86f",
    center: "#fff1b5",
    centerBorder: "#8b653a",
    text: "#231a13",
    aiText: "#a1181c",
    humanPiece: "#ffe19a",
    aiPiece: "#ffc9a3",
    revealedPiece: "#dde9df",
    hiddenPiece: "#8a5a3b",
    pieceEdge: "#7a4d2e",
    turn: "#c52327",
    grain: "#fff0c5",
    grid: "#f9dfad",
    acText: "#265e58",
    bdText: "#7f2e2d",
  },
  mono: {
    outer: "#000000",
    frameDark: "#000000",
    frame: "#ffffff",
    board: "#ffffff",
    boardLight: "#ffffff",
    boardDark: "#ffffff",
    center: "#ffffff",
    centerBorder: "#000000",
    text: "#000000",
    aiText: "#000000",
    humanPiece: "#ffffff",
    aiPiece: "#ffffff",
    revealedPiece: "#ffffff",
    hiddenPiece: "#ffffff",
    pieceEdge: "#000000",
    turn: "#000000",
    grain: "#ffffff",
    grid: "#ffffff",
    acText: "#000000",
    bdText: "#000000",
  },
};
let palette = palettes.color;

function snap(value) {
  return Math.round(value);
}

function trimName(name, maxLength = 7) {
  const chars = Array.from(String(name || ""));
  return chars.length > maxLength ? `${chars.slice(0, maxLength - 1).join("")}…` : chars.join("");
}

function drawPixelText(text, x, y, options = {}) {
  const {
    size = 12,
    color = palette.text,
    align = "center",
    baseline = "middle",
    weight = "700",
  } = options;
  context.save();
  context.imageSmoothingEnabled = false;
  context.fillStyle = color;
  context.textAlign = align;
  context.textBaseline = baseline;
  context.font = `${weight} ${size}px "MS Gothic", "Yu Gothic", sans-serif`;
  context.fillText(String(text), snap(x), snap(y));
  context.restore();
}

function drawWoodBoard() {
  context.fillStyle = palette.outer;
  context.fillRect(0, 0, SIZE, SIZE);

  context.fillStyle = palette.frameDark;
  context.fillRect(4, 4, 312, 312);
  context.fillStyle = palette.frame;
  context.fillRect(8, 8, 304, 304);
  context.fillStyle = palette.frameDark;
  context.fillRect(11, 11, 298, 298);
  context.fillStyle = palette.board;
  context.fillRect(GRID_ORIGIN, GRID_ORIGIN, CELL * 8, CELL * 8);

  for (let y = GRID_ORIGIN; y < GRID_ORIGIN + CELL * 8; y += 4) {
    context.fillStyle = ((y / 4) % 3 === 0) ? palette.boardLight : palette.boardDark;
    context.globalAlpha = 0.12;
    context.fillRect(GRID_ORIGIN, y, CELL * 8, 1);
  }
  context.globalAlpha = 1;

  for (let x = GRID_ORIGIN + 18; x < GRID_ORIGIN + CELL * 8; x += 37) {
    context.fillStyle = palette.grain;
    context.globalAlpha = 0.08;
    context.fillRect(x, GRID_ORIGIN, 2, CELL * 8);
  }
  context.globalAlpha = 1;

  context.strokeStyle = palette.grid;
  context.globalAlpha = 0.16;
  context.lineWidth = 1;
  for (let i = 1; i < 8; i += 1) {
    const line = GRID_ORIGIN + CELL * i;
    context.beginPath();
    context.moveTo(line, GRID_ORIGIN);
    context.lineTo(line, GRID_ORIGIN + CELL * 8);
    context.stroke();
    context.beginPath();
    context.moveTo(GRID_ORIGIN, line);
    context.lineTo(GRID_ORIGIN + CELL * 8, line);
    context.stroke();
  }
  context.globalAlpha = 1;
}

function drawPieceShape(fill, stroke) {
  const halfWidth = PIECE_WIDTH / 2;
  const halfHeight = PIECE_HEIGHT / 2;
  context.beginPath();
  context.moveTo(0, -halfHeight);
  context.lineTo(halfWidth - 1, -halfHeight + 5);
  context.lineTo(halfWidth, halfHeight);
  context.lineTo(-halfWidth, halfHeight);
  context.lineTo(-halfWidth + 1, -halfHeight + 5);
  context.closePath();
  context.fillStyle = fill;
  context.fill();
  context.strokeStyle = stroke;
  context.lineWidth = 2;
  context.stroke();
}

function drawPiece(piece) {
  const x = GRID_ORIGIN + (piece.col - 0.5) * CELL;
  const y = GRID_ORIGIN + (piece.row - 0.5) * CELL;
  const rotation = PIECE_ROTATION[piece.phys] || 0;
  const fill = piece.hidden
    ? palette.hiddenPiece
    : piece.revealedHand
      ? palette.revealedPiece
      : piece.ai
        ? palette.aiPiece
        : palette.humanPiece;

  context.save();
  context.translate(snap(x), snap(y));
  context.rotate(rotation);
  context.globalAlpha = piece.revealedHidden ? 0.56 : 1;

  drawPieceShape(fill, palette.pieceEdge);

  if (!piece.hidden && piece.label) {
    drawPixelText(piece.label, 0, 2, {
      size: 19,
      color: piece.ai ? palette.aiText : palette.text,
      weight: "900",
    });
  }
  context.restore();
}

function drawCenterPanel(snapshot) {
  context.fillStyle = palette.center;
  context.fillRect(103, 123, 112, 72);
  context.strokeStyle = palette.centerBorder;
  context.lineWidth = 2;
  context.strokeRect(103, 123, 112, 72);

  drawPixelText(`第${snapshot.round}局`, 159, 139, {size: 12});
  drawPixelText(`AC ${snapshot.scores.AC}点`, 159, 159, {size: 12, color: palette.acText});
  drawPixelText(`BD ${snapshot.scores.BD}点`, 159, 178, {size: 12, color: palette.bdText});
}

function drawSeatLabel(snapshot, phys, x, y, align = "center") {
  const seat = snapshot.physToSeat?.[phys] || phys;
  const name = trimName(snapshot.names?.[seat]);
  const thinking = snapshot.aiThinkingSeat === seat ? " ■" : "";
  const speaking = snapshot.voiceSpeakingSeats?.includes(seat) ? " MIC" : "";
  const label = `${seat}${name ? `:${name}` : ""}${thinking}${speaking}`;
  const isTurn = snapshot.isStarted && !snapshot.finished && snapshot.turn === seat;
  drawPixelText(label, x, y, {
    size: name ? 11 : 13,
    color: isTurn ? palette.turn : palette.text,
    align,
    weight: isTurn ? "900" : "700",
  });
}

function drawSeatLabels(snapshot) {
  drawSeatLabel(snapshot, "C", 160, 108);
  drawSeatLabel(snapshot, "A", 160, 211);
  drawSeatLabel(snapshot, "D", 94, 160, "right");
  drawSeatLabel(snapshot, "B", 224, 160, "left");
}

function drawScene() {
  if (!context || !latestSnapshot) return;
  context.imageSmoothingEnabled = false;
  context.clearRect(0, 0, SIZE, SIZE);
  drawWoodBoard();

  latestSnapshot.pieces.forEach(drawPiece);

  drawCenterPanel(latestSnapshot);
  drawSeatLabels(latestSnapshot);
}

function render(snapshot) {
  if (!context || !snapshot) return;
  latestSnapshot = snapshot;
  drawScene();
}

function setVisible(nextVisible) {
  visible = !!nextVisible;
  if (visible && latestSnapshot) drawScene();
}

function setTheme(theme) {
  palette = theme === "mono" ? palettes.mono : palettes.color;
  if (visible && latestSnapshot) drawScene();
}

if (canvas && context) {
  canvas.width = SIZE;
  canvas.height = SIZE;
  context.imageSmoothingEnabled = false;
}

window.goitaBoardPixel = {
  isAvailable: () => !!(canvas && container && context),
  setVisible,
  setTheme,
  render,
  getCanvas: () => canvas,
};

window.dispatchEvent(new CustomEvent("goita-board-pixel-ready"));
