import * as THREE from "./three.module.min.js";

const canvas = document.getElementById("board3dCanvas");
const container = document.getElementById("board3d");

let renderer = null;
let scene = null;
let camera = null;
let pieceLayer = null;
let labelLayer = null;
let resizeObserver = null;
let available = null;
let visible = false;
let hasRenderedState = false;
let previousPieceKeys = new Set();
let animationFrame = 0;
let activeAnimations = [];
let cameraYaw = 0;
let cameraElevation = 0.72;
let pointerState = null;

const pieceGeometry = createPieceGeometry();
const highlightGeometry = new THREE.RingGeometry(0.43, 0.52, 32);
const pieceMaterials = new Map();
const pieceTextMaterials = new Map();
const highlightMaterials = new Map();
const labelDisposables = [];
const PHYS_ROTATION = {
  A: 0,
  B: Math.PI / 2,
  C: Math.PI,
  D: -Math.PI / 2,
};

function createPieceGeometry() {
  const shape = new THREE.Shape();
  shape.moveTo(-0.38, -0.52);
  shape.lineTo(0.38, -0.52);
  shape.lineTo(0.34, 0.28);
  shape.lineTo(0, 0.54);
  shape.lineTo(-0.34, 0.28);
  shape.closePath();
  const geometry = new THREE.ExtrudeGeometry(shape, {
    depth: 0.16,
    bevelEnabled: true,
    bevelSegments: 2,
    bevelSize: 0.025,
    bevelThickness: 0.025,
  });
  geometry.computeVertexNormals();
  return geometry;
}

function roundedRect(context, x, y, width, height, radius) {
  context.beginPath();
  context.roundRect(x, y, width, height, radius);
  context.closePath();
}

function createWoodTexture() {
  const textureCanvas = document.createElement("canvas");
  textureCanvas.width = 1024;
  textureCanvas.height = 1024;
  const context = textureCanvas.getContext("2d");
  const gradient = context.createLinearGradient(0, 0, 1024, 1024);
  gradient.addColorStop(0, "#d5ad74");
  gradient.addColorStop(0.48, "#c4935a");
  gradient.addColorStop(1, "#b57b43");
  context.fillStyle = gradient;
  context.fillRect(0, 0, 1024, 1024);

  context.globalAlpha = 0.13;
  for (let index = 0; index < 46; index += 1) {
    const y = 12 + index * 22;
    context.strokeStyle = index % 3 === 0 ? "#6f421f" : "#f4d7a5";
    context.lineWidth = index % 4 === 0 ? 3 : 1.4;
    context.beginPath();
    for (let x = -40; x <= 1064; x += 20) {
      const wave = Math.sin((x + index * 31) / 82) * 5;
      if (x === -40) context.moveTo(x, y + wave);
      else context.lineTo(x, y + wave);
    }
    context.stroke();
  }
  context.globalAlpha = 1;

  const texture = new THREE.CanvasTexture(textureCanvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return texture;
}

function createScene() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xcda56f);

  camera = new THREE.PerspectiveCamera(34, 1, 0.1, 60);
  updateCamera();

  const ambient = new THREE.HemisphereLight(0xfff4dc, 0x65462d, 2.1);
  scene.add(ambient);

  const keyLight = new THREE.DirectionalLight(0xffead0, 3.6);
  keyLight.position.set(-5, 11, 7);
  keyLight.castShadow = !isSmallScreen();
  keyLight.shadow.mapSize.set(1024, 1024);
  keyLight.shadow.camera.left = -7;
  keyLight.shadow.camera.right = 7;
  keyLight.shadow.camera.top = 7;
  keyLight.shadow.camera.bottom = -7;
  scene.add(keyLight);

  const board = new THREE.Mesh(
    new THREE.BoxGeometry(9.05, 0.34, 9.05),
    new THREE.MeshStandardMaterial({
      map: createWoodTexture(),
      roughness: 0.76,
      metalness: 0,
    })
  );
  board.position.y = 0;
  board.receiveShadow = true;
  scene.add(board);

  const frameMaterial = new THREE.MeshStandardMaterial({
    color: 0x74421f,
    roughness: 0.68,
  });
  [
    [0, 0.16, -4.62, 9.5, 0.32, 0.2],
    [0, 0.16, 4.62, 9.5, 0.32, 0.2],
    [-4.62, 0.16, 0, 0.2, 0.32, 9.5],
    [4.62, 0.16, 0, 0.2, 0.32, 9.5],
  ].forEach(([x, y, z, width, height, depth]) => {
    const frame = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), frameMaterial);
    frame.position.set(x, y, z);
    frame.castShadow = true;
    scene.add(frame);
  });

  const centerMat = new THREE.MeshStandardMaterial({
    color: 0xe8d4a8,
    roughness: 0.86,
    transparent: true,
    opacity: 0.72,
  });
  const center = new THREE.Mesh(new THREE.BoxGeometry(3.5, 0.025, 3.5), centerMat);
  center.position.y = 0.19;
  center.receiveShadow = true;
  scene.add(center);

  pieceLayer = new THREE.Group();
  labelLayer = new THREE.Group();
  scene.add(pieceLayer, labelLayer);
}

function isSmallScreen() {
  return window.matchMedia("(max-width: 701px)").matches;
}

function ensureRenderer() {
  if (available !== null) return available;
  if (!canvas || !container) {
    available = false;
    return false;
  }

  try {
    renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: !isSmallScreen(),
      alpha: false,
      powerPreference: "high-performance",
    });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, isSmallScreen() ? 1.25 : 1.75));
    renderer.shadowMap.enabled = !isSmallScreen();
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    createScene();
    attachInteraction();
    resizeObserver = new ResizeObserver(() => resizeAndRender());
    resizeObserver.observe(container);
    available = true;
  } catch (error) {
    console.warn("3D board is unavailable:", error);
    available = false;
  }
  return available;
}

function resizeAndRender() {
  if (!renderer || !camera || !container) return;
  const width = Math.max(1, Math.floor(container.clientWidth));
  const height = Math.max(1, Math.floor(container.clientHeight));
  const size = renderer.getSize(new THREE.Vector2());
  if (size.x !== width || size.y !== height) {
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
  renderFrame();
}

function updateCamera() {
  if (!camera) return;
  const radius = 13.7;
  const horizontal = Math.cos(cameraElevation) * radius;
  camera.position.set(
    Math.sin(cameraYaw) * horizontal,
    Math.sin(cameraElevation) * radius,
    Math.cos(cameraYaw) * horizontal
  );
  camera.lookAt(0, 0.05, 0);
}

function attachInteraction() {
  canvas.addEventListener("pointerdown", (event) => {
    pointerState = { id: event.pointerId, x: event.clientX, y: event.clientY };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!pointerState || event.pointerId !== pointerState.id) return;
    const dx = event.clientX - pointerState.x;
    const dy = event.clientY - pointerState.y;
    pointerState.x = event.clientX;
    pointerState.y = event.clientY;
    cameraYaw = THREE.MathUtils.clamp(cameraYaw - dx * 0.004, -0.52, 0.52);
    cameraElevation = THREE.MathUtils.clamp(cameraElevation + dy * 0.003, 0.55, 0.98);
    updateCamera();
    renderFrame();
  });
  const releasePointer = (event) => {
    if (pointerState?.id === event.pointerId) pointerState = null;
  };
  canvas.addEventListener("pointerup", releasePointer);
  canvas.addEventListener("pointercancel", releasePointer);
  canvas.addEventListener("dblclick", () => {
    cameraYaw = 0;
    cameraElevation = 0.72;
    updateCamera();
    renderFrame();
  });
}

function pieceBaseMaterial(kind) {
  if (pieceMaterials.has(kind)) return pieceMaterials.get(kind);
  const colors = {
    human: 0xe8c778,
    ai: 0xe0a16f,
    revealed: 0xbcbcbc,
    hidden: 0x8c5b34,
  };
  const material = new THREE.MeshStandardMaterial({
    color: colors[kind] || colors.human,
    roughness: 0.62,
    metalness: 0,
    transparent: kind === "revealed",
    opacity: kind === "revealed" ? 0.82 : 1,
  });
  pieceMaterials.set(kind, material);
  return material;
}

function pieceTextMaterial(label, color, opacity) {
  const cacheKey = `${label}-${color}-${opacity}`;
  if (pieceTextMaterials.has(cacheKey)) return pieceTextMaterials.get(cacheKey);

  const textureCanvas = document.createElement("canvas");
  textureCanvas.width = 256;
  textureCanvas.height = 320;
  const context = textureCanvas.getContext("2d");
  context.clearRect(0, 0, 256, 320);
  context.fillStyle = color;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.font = '900 184px "Yu Kyokasho", "Yu Mincho", serif';
  context.fillText(label, 128, 170);

  const texture = new THREE.CanvasTexture(textureCanvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    opacity,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -4,
  });
  pieceTextMaterials.set(cacheKey, material);
  return material;
}

function createPieceObject(piece) {
  const root = new THREE.Group();
  root.position.set((piece.col - 4.5) * 1.04, 0.2, (piece.row - 4.5) * 1.04);
  root.rotation.y = PHYS_ROTATION[piece.phys] || 0;

  const flat = new THREE.Group();
  flat.rotation.x = -Math.PI / 2;
  root.add(flat);

  const materialKind = piece.hidden
    ? "hidden"
    : piece.revealedHand
      ? "revealed"
      : piece.ai
        ? "ai"
        : "human";
  const mesh = new THREE.Mesh(pieceGeometry, pieceBaseMaterial(materialKind));
  mesh.castShadow = !isSmallScreen();
  mesh.receiveShadow = true;
  flat.add(mesh);

  if (!piece.hidden && piece.label) {
    const textPlane = new THREE.Mesh(
      new THREE.PlaneGeometry(0.58, 0.72),
      pieceTextMaterial(piece.label, piece.ai ? "#a31313" : "#17130f", piece.revealedHidden ? 0.55 : 1)
    );
    textPlane.position.set(0, -0.01, 0.195);
    flat.add(textPlane);
  }

  if (piece.current || piece.pending) {
    const highlightKind = piece.pending ? "pending" : "current";
    if (!highlightMaterials.has(highlightKind)) {
      highlightMaterials.set(highlightKind, new THREE.MeshBasicMaterial({
        color: piece.pending ? 0x276fae : 0xd84b16,
        transparent: true,
        opacity: 0.82,
        side: THREE.DoubleSide,
      }));
    }
    const ring = new THREE.Mesh(
      highlightGeometry,
      highlightMaterials.get(highlightKind)
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.015;
    root.add(ring);
  }
  return root;
}

function clearLayer(layer, disposeMaterials) {
  while (layer.children.length) {
    const child = layer.children[0];
    layer.remove(child);
    child.traverse((object) => {
      if (disposeMaterials && object.material) {
        if (object.material.map) object.material.map.dispose();
        object.material.dispose();
      }
      if (disposeMaterials && object.geometry) object.geometry.dispose();
    });
  }
}

function createTextSprite(text, options = {}) {
  const width = options.width || 512;
  const height = options.height || 144;
  const textureCanvas = document.createElement("canvas");
  textureCanvas.width = width;
  textureCanvas.height = height;
  const context = textureCanvas.getContext("2d");
  context.clearRect(0, 0, width, height);

  if (options.background) {
    context.fillStyle = options.background;
    roundedRect(context, 8, 8, width - 16, height - 16, 20);
    context.fill();
  }

  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillStyle = options.color || "#2b2016";
  context.font = `800 ${options.fontSize || 58}px "Yu Kyokasho", "Yu Mincho", serif`;
  const lines = String(text).split("\n");
  const lineHeight = options.lineHeight || 62;
  const startY = height / 2 - ((lines.length - 1) * lineHeight) / 2;
  lines.forEach((line, index) => context.fillText(line, width / 2, startY + index * lineHeight));

  const texture = new THREE.CanvasTexture(textureCanvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
  });
  const sprite = new THREE.Sprite(material);
  sprite.renderOrder = 20;
  labelDisposables.push(sprite);
  return sprite;
}

function rebuildLabels(snapshot) {
  clearLayer(labelLayer, true);
  labelDisposables.length = 0;

  const namePositions = {
    A: [0, 0.66, 1.72],
    B: [1.72, 0.66, 0],
    C: [0, 0.66, -1.72],
    D: [-1.72, 0.66, 0],
  };

  Object.entries(namePositions).forEach(([phys, position]) => {
    const seat = snapshot.physToSeat[phys];
    const rawName = String(snapshot.names?.[seat] || "").slice(0, 9);
    const thinking = snapshot.aiThinkingSeat === seat ? "  ●" : "";
    const label = `${seat}${rawName ? `: ${rawName}` : ""}${thinking}`;
    const isTurn = snapshot.turn === seat && snapshot.isStarted && !snapshot.finished;
    const sprite = createTextSprite(label, {
      color: isTurn ? "#b00000" : "#302014",
      fontSize: rawName.length > 6 ? 44 : 53,
    });
    sprite.position.set(...position);
    sprite.scale.set(2.35, 0.66, 1);
    labelLayer.add(sprite);
  });

  if (snapshot.isStarted || snapshot.finished) {
    const score = createTextSprite(
      `第 ${snapshot.round} 局\nAC: ${snapshot.scores.AC}点   BD: ${snapshot.scores.BD}点`,
      {
        width: 640,
        height: 260,
        fontSize: 54,
        lineHeight: 78,
        color: "#4d321d",
        background: "rgba(255, 250, 232, 0.82)",
      }
    );
    score.position.set(0, 0.73, 0);
    score.scale.set(2.65, 1.08, 1);
    labelLayer.add(score);
  }
}

function animatePieces(now) {
  activeAnimations = activeAnimations.filter((animation) => {
    const progress = Math.min(1, (now - animation.startedAt) / 300);
    const eased = 1 - Math.pow(1 - progress, 3);
    animation.object.position.y = THREE.MathUtils.lerp(animation.fromY, animation.toY, eased);
    return progress < 1;
  });
  renderFrame();
  if (activeAnimations.length) {
    animationFrame = requestAnimationFrame(animatePieces);
  } else {
    animationFrame = 0;
  }
}

function renderFrame() {
  if (!renderer || !scene || !camera || !visible) return;
  renderer.render(scene, camera);
}

function render(snapshot) {
  if (!ensureRenderer() || !snapshot) return false;
  resizeAndRender();
  clearLayer(pieceLayer, false);
  activeAnimations = [];
  if (animationFrame) {
    cancelAnimationFrame(animationFrame);
    animationFrame = 0;
  }

  const nextKeys = new Set();
  snapshot.pieces.forEach((piece) => {
    nextKeys.add(piece.key);
    const object = createPieceObject(piece);
    const targetY = object.position.y;
    if (hasRenderedState && !previousPieceKeys.has(piece.key)) {
      object.position.y = targetY + 1.15;
      activeAnimations.push({
        object,
        fromY: object.position.y,
        toY: targetY,
        startedAt: performance.now(),
      });
    }
    pieceLayer.add(object);
  });
  rebuildLabels(snapshot);
  previousPieceKeys = nextKeys;
  hasRenderedState = true;
  renderFrame();
  if (activeAnimations.length) animationFrame = requestAnimationFrame(animatePieces);
  return true;
}

function setVisible(nextVisible) {
  visible = !!nextVisible;
  if (visible && ensureRenderer()) resizeAndRender();
}

window.goitaBoard3D = {
  isAvailable: ensureRenderer,
  render,
  setVisible,
  getCanvas: () => canvas,
};

window.dispatchEvent(new CustomEvent("goita-board3d-ready"));
