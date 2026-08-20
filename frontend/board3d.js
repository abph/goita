import * as THREE from "./three.module.min.js";

const canvas = document.getElementById("board3dCanvas");
const container = document.getElementById("board3d");
const zoomInButton = document.getElementById("board3dZoomIn");
const zoomOutButton = document.getElementById("board3dZoomOut");
const panUpButton = document.getElementById("board3dPanUp");
const panDownButton = document.getElementById("board3dPanDown");
const panLeftButton = document.getElementById("board3dPanLeft");
const panRightButton = document.getElementById("board3dPanRight");

let renderer = null;
let scene = null;
let camera = null;
let standardFloor = null;
let meetingRoomGroup = null;
let meetingPublicTablesGroup = null;
let playAreaGroup = null;
let standardBoardFurniture = null;
let pieceLayer = null;
let labelLayer = null;
let passLayer = null;
let scoreLayer = null;
let resizeObserver = null;
let available = null;
let visible = false;
let environmentMode = "board";
let hasRenderedState = false;
let latestSnapshot = null;
let previousPieceKeys = new Set();
let animationFrame = 0;
let activeAnimations = [];
let cameraYaw = 0;
let cameraElevation = 0.72;
let cameraRadius = 23.2;
let cameraPanX = 0;
let cameraPanZ = 0;
let pointerState = null;
let pinchDistance = null;
let pointerTravel = 0;
let suppressNextClick = false;
const activePointers = new Map();
const meetingPublicTables = new Map();
const meetingPublicTableTargets = [];

const DEFAULT_CAMERA_RADIUS = 23.2;
const MIN_CAMERA_RADIUS = 15.8;
const MAX_CAMERA_RADIUS = 31.5;
const MEETING_ZOOM_STOPS = [2.9, 3.5, 4.05, 4.6, 6.4, 8.2, 10, 11.8];
const CAMERA_PROFILES = {
  board: {
    yaw: 0,
    elevation: 0.72,
    radius: DEFAULT_CAMERA_RADIUS,
    minRadius: MIN_CAMERA_RADIUS,
    maxRadius: MAX_CAMERA_RADIUS,
    minYaw: -0.52,
    maxYaw: 0.52,
    minElevation: 0.55,
    maxElevation: 0.98,
    orbitCenter: [0, 0, 0],
    lookTarget: [0, -2.02, 0],
  },
  "meeting-room": {
    yaw: 0.03,
    elevation: 0.55,
    radius: 8.2,
    minRadius: 2.9,
    maxRadius: 11.8,
    minYaw: -0.22,
    maxYaw: 0.28,
    minElevation: 0.28,
    maxElevation: 1.05,
    orbitCenter: [5.1, -2.18, -14.6],
    lookTarget: [5.1, -1.1, -14.6],
  },
};
const BOARD_TOP_Y = 0.17;
const BOARD_BODY_HEIGHT = 3.44;
const BOARD_BODY_TOP_Y = -0.37;
const BOARD_BODY_BOTTOM_Y = BOARD_BODY_TOP_Y - BOARD_BODY_HEIGHT;
const SIDE_SHELF_Y = BOARD_BODY_BOTTOM_Y + 1.5;
const SIDE_SHELF_LENGTH = 5.2;
const SIDE_SHELF_DEPTH = 2.36;
const SIDE_SHELF_OFFSET = 5.28;
const TABLE_FLOOR_Y = BOARD_BODY_BOTTOM_Y - 0.99;
const MEETING_ROOM_WIDTH = 24;
const MEETING_ROOM_DEPTH = 60;
const MEETING_ROOM_BACK_Z = -32;
const MEETING_ROOM_CENTER_Z = -2;
const MEETING_ROOM_HEIGHT = 10;
const MEETING_WINDOW_WIDTH = MEETING_ROOM_WIDTH * 0.94;
const MEETING_WINDOW_HEIGHT = MEETING_ROOM_HEIGHT * 0.6;
const MEETING_BLIND_HEIGHT = MEETING_WINDOW_HEIGHT * 0.5;
const MEETING_BLIND_WIDTH = MEETING_WINDOW_WIDTH / 3 - 0.24;
const MEETING_BLIND_CENTERS = [-MEETING_WINDOW_WIDTH / 3, 0, MEETING_WINDOW_WIDTH / 3];
const MEETING_TABLE_COLUMNS = [-5.1, 5.1];
const MEETING_TABLE_ROWS = [-4.8, -14.6, -24.4];
const MEETING_BOARD_X = 5.1;
const MEETING_BOARD_Z = -14.6;
const MEETING_PUBLIC_TABLE_LAYOUT = [
  { roomId: "main", x: -5.1, z: -24.4 },
  { roomId: "main-b", x: 5.1, z: -24.4 },
  { roomId: "main-c", x: -5.1, z: -14.6 },
  { roomId: "main-e", x: -5.1, z: -4.8 },
  { roomId: "main-f", x: 5.1, z: -4.8 },
];
const MEETING_TABLE_TOP_Y = TABLE_FLOOR_Y + 2.35;
const MEETING_BOARD_SCALE = 0.3;
const MEETING_PUBLIC_BOARD_SCALE = 0.27;
const MEETING_BOARD_CLEARANCE = 0.08;
const MEETING_WHITEBOARD_X = -MEETING_WINDOW_WIDTH / 3;
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
const PASS_WORLD_POSITIONS = {
  A: [0, 1.12, 3.12],
  B: [3.12, 1.12, 0],
  C: [0, 1.12, -3.12],
  D: [-3.12, 1.12, 0],
};

function tr(text) {
  return window.goitaI18n?.translate?.(text) ?? text;
}

function createPieceGeometry() {
  const shape = new THREE.Shape();
  shape.moveTo(-0.38, -0.48);
  shape.lineTo(0.38, -0.48);
  shape.lineTo(0.35, 0.34);
  shape.lineTo(0, 0.5);
  shape.lineTo(-0.35, 0.34);
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

function createBoardLeg(material, x, z) {
  const leg = new THREE.Group();
  leg.position.set(x, 0, z);

  const collar = new THREE.Mesh(
    new THREE.BoxGeometry(0.72, 0.28, 0.72),
    material
  );
  collar.position.y = BOARD_BODY_BOTTOM_Y - 0.13;

  const stem = new THREE.Mesh(
    new THREE.CylinderGeometry(0.29, 0.43, 0.62, 8),
    material
  );
  stem.position.y = BOARD_BODY_BOTTOM_Y - 0.54;
  stem.rotation.y = Math.PI / 8;

  const foot = new THREE.Mesh(
    new THREE.CylinderGeometry(0.52, 0.46, 0.2, 8),
    material
  );
  foot.position.y = BOARD_BODY_BOTTOM_Y - 0.87;
  foot.rotation.y = Math.PI / 8;

  [collar, stem, foot].forEach((part) => {
    part.castShadow = true;
    part.receiveShadow = true;
    leg.add(part);
  });
  return leg;
}

function createSideShelves(material, supportMaterial) {
  const shelves = new THREE.Group();
  const shelfSpecs = [
    {
      size: [SIDE_SHELF_DEPTH, 0.18, SIDE_SHELF_LENGTH],
      position: [-SIDE_SHELF_OFFSET, SIDE_SHELF_Y, 0],
    },
    {
      size: [SIDE_SHELF_DEPTH, 0.18, SIDE_SHELF_LENGTH],
      position: [SIDE_SHELF_OFFSET, SIDE_SHELF_Y, 0],
    },
    {
      size: [SIDE_SHELF_LENGTH, 0.18, SIDE_SHELF_DEPTH],
      position: [0, SIDE_SHELF_Y, -SIDE_SHELF_OFFSET],
    },
  ];

  shelfSpecs.forEach(({ size, position }) => {
    const shelf = new THREE.Mesh(new THREE.BoxGeometry(...size), material);
    shelf.position.set(...position);
    shelf.castShadow = true;
    shelf.receiveShadow = true;
    shelves.add(shelf);
  });

  const supportSpecs = [
    { size: [1.5, 0.68, 0.3], position: [-5, SIDE_SHELF_Y - 0.4, -1.6] },
    { size: [1.5, 0.68, 0.3], position: [-5, SIDE_SHELF_Y - 0.4, 1.6] },
    { size: [1.5, 0.68, 0.3], position: [5, SIDE_SHELF_Y - 0.4, -1.6] },
    { size: [1.5, 0.68, 0.3], position: [5, SIDE_SHELF_Y - 0.4, 1.6] },
    { size: [0.3, 0.68, 1.5], position: [-1.6, SIDE_SHELF_Y - 0.4, -5] },
    { size: [0.3, 0.68, 1.5], position: [1.6, SIDE_SHELF_Y - 0.4, -5] },
  ];

  supportSpecs.forEach(({ size, position }) => {
    const support = new THREE.Mesh(new THREE.BoxGeometry(...size), supportMaterial);
    support.position.set(...position);
    support.castShadow = true;
    support.receiveShadow = true;
    shelves.add(support);
  });
  return shelves;
}

function addRoomBox(group, size, position, material, options = {}) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), material);
  mesh.position.set(...position);
  mesh.castShadow = options.castShadow === true;
  mesh.receiveShadow = options.receiveShadow !== false;
  group.add(mesh);
  return mesh;
}

function createMeetingChair(materials, x, z, facing) {
  const chair = new THREE.Group();
  const floorY = TABLE_FLOOR_Y;
  const direction = facing === "back" ? -1 : 1;
  addRoomBox(chair, [0.92, 0.16, 0.86], [x, floorY + 1.22, z], materials.chair, {
    castShadow: true,
  });
  addRoomBox(
    chair,
    [0.92, 1.02, 0.14],
    [x, floorY + 1.82, z + direction * 0.38],
    materials.chair,
    { castShadow: true }
  );
  [-0.34, 0.34].forEach((offsetX) => {
    [-0.27, 0.27].forEach((offsetZ) => {
      addRoomBox(
        chair,
        [0.08, 1.18, 0.08],
        [x + offsetX, floorY + 0.59, z + offsetZ],
        materials.metal,
        { castShadow: true }
      );
    });
  });
  return chair;
}

function createMeetingTable(materials, x, z) {
  const table = new THREE.Group();
  const floorY = TABLE_FLOOR_Y;
  addRoomBox(table, [7.2, 0.3, 2.9], [x, floorY + 2.2, z], materials.table, {
    castShadow: true,
  });
  [-3.05, 3.05].forEach((offsetX) => {
    [-1.05, 1.05].forEach((offsetZ) => {
      addRoomBox(
        table,
        [0.14, 2.05, 0.14],
        [x + offsetX, floorY + 1.05, z + offsetZ],
        materials.metal,
        { castShadow: true }
      );
    });
  });

  [-2.15, 0, 2.15].forEach((seatX) => {
    table.add(createMeetingChair(materials, x + seatX, z + 2.08, "front"));
    table.add(createMeetingChair(materials, x + seatX, z - 2.08, "back"));
  });
  return table;
}

function createMeetingPublicTable(materials, layout) {
  const root = new THREE.Group();
  root.position.set(
    layout.x,
    MEETING_TABLE_TOP_Y + 0.08,
    layout.z
  );
  root.scale.setScalar(MEETING_PUBLIC_BOARD_SCALE);
  root.visible = false;

  const board = addRoomBox(
    root,
    [9.05, 0.28, 9.05],
    [0, 0, 0],
    materials.publicBoard,
    { castShadow: true }
  );
  board.userData.publicRoomId = layout.roomId;
  meetingPublicTableTargets.push(board);

  const center = addRoomBox(
    root,
    [3.5, 0.04, 3.5],
    [0, 0.17, 0],
    materials.publicBoardCenter,
    { receiveShadow: true }
  );
  center.userData.publicRoomId = layout.roomId;
  meetingPublicTableTargets.push(center);

  const pieces = new THREE.Group();
  const labels = new THREE.Group();
  const passes = new THREE.Group();
  root.add(pieces, labels, passes);
  meetingPublicTables.set(layout.roomId, {
    root,
    pieces,
    labels,
    passes,
    passToken: "",
  });
  return root;
}

function publicTableStatus(snapshot) {
  if (!snapshot.isStarted) return tr("待機中");
  if (snapshot.finished) return tr("終局");
  return tr("対局中");
}

function addMeetingPublicTablePass(entry, snapshot) {
  const action = snapshot.lastPublicAction;
  if (!action || action.type !== "pass") return;
  const age = Date.now() - Number(action.atMs || 0);
  if (age < 0 || age >= 2200) return;
  const position = PASS_WORLD_POSITIONS[action.player];
  if (!position) return;

  const marker = createTextSprite(tr("パス"), {
    width: 320,
    height: 144,
    background: "rgba(255, 250, 236, 0.96)",
    borderColor: "#9b2226",
    borderWidth: 9,
    color: "#9b2226",
    fontSize: 68,
    trackLabel: false,
    depthTest: true,
  });
  marker.position.set(position[0], 0.75, position[2]);
  marker.scale.set(1.45, 0.64, 1);
  entry.passes.add(marker);

  const token = `${action.player}-${action.atMs}`;
  entry.passToken = token;
  window.setTimeout(() => {
    if (entry.passToken !== token) return;
    clearLayer(entry.passes, true);
    entry.passToken = "";
    renderFrame();
  }, Math.max(60, 2200 - age));
}

function setPublicTables(snapshots = []) {
  if (!ensureRenderer()) return false;
  const snapshotByRoom = new Map(
    snapshots.map((snapshot) => [snapshot.roomId, snapshot])
  );

  meetingPublicTables.forEach((entry, roomId) => {
    const snapshot = snapshotByRoom.get(roomId);
    entry.root.visible = !!snapshot;
    clearLayer(entry.pieces, false);
    clearLayer(entry.labels, true);
    clearLayer(entry.passes, true);
    entry.passToken = "";
    if (!snapshot) return;

    snapshot.pieces.forEach((piece) => {
      entry.pieces.add(createPieceObject(piece));
    });

    const scoreText = snapshot.isStarted || snapshot.finished
      ? `\nAC ${snapshot.scores.AC} / BD ${snapshot.scores.BD}`
      : "";
    const label = createTextSprite(
      `${tr(snapshot.roomName)}\n${publicTableStatus(snapshot)}${scoreText}`,
      {
        width: 720,
        height: 260,
        background: "rgba(255, 255, 255, 0.94)",
        borderColor: "#5f5548",
        borderWidth: 7,
        color: "#241d17",
        fontSize: 46,
        lineHeight: 68,
        trackLabel: false,
        depthTest: true,
      }
    );
    label.position.set(0, 2.2, -5.6);
    label.scale.set(6.8, 2.45, 1);
    entry.labels.add(label);
    addMeetingPublicTablePass(entry, snapshot);
  });
  renderFrame();
  return true;
}

function addMeetingRoomSilhouette(room) {
  const loader = new THREE.TextureLoader();
  loader.load(
    "/static/meeting-room-silhouette.png?v=20260806a",
    (texture) => {
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.minFilter = THREE.LinearFilter;
      texture.magFilter = THREE.LinearFilter;
      const material = new THREE.MeshBasicMaterial({
        map: texture,
        transparent: true,
        opacity: 0.94,
        alphaTest: 0.025,
        depthWrite: false,
        side: THREE.DoubleSide,
      });
      const silhouette = new THREE.Mesh(new THREE.PlaneGeometry(3.35, 5.03), material);
      silhouette.position.set(10.15, TABLE_FLOOR_Y + 2.515, -8.2);
      silhouette.rotation.y = -0.08;
      silhouette.renderOrder = 3;
      room.add(silhouette);
      renderFrame();
    },
    undefined,
    (error) => console.warn("Meeting-room silhouette could not be loaded.", error)
  );
}

function createFreestandingWhiteboard(materials) {
  const whiteboard = new THREE.Group();
  const floorY = TABLE_FLOOR_Y;
  addRoomBox(whiteboard, [5.6, 3.65, 0.2], [MEETING_WHITEBOARD_X, floorY + 4.15, -30.7], materials.frame, {
    castShadow: true,
  });
  addRoomBox(whiteboard, [5.18, 3.22, 0.22], [MEETING_WHITEBOARD_X, floorY + 4.15, -30.55], materials.whiteboard, {
    castShadow: true,
  });
  [MEETING_WHITEBOARD_X - 2, MEETING_WHITEBOARD_X + 2].forEach((x) => {
    addRoomBox(whiteboard, [0.14, 3.0, 0.14], [x, floorY + 1.52, -30.7], materials.metal, {
      castShadow: true,
    });
    addRoomBox(whiteboard, [1.2, 0.12, 0.65], [x, floorY + 0.08, -30.55], materials.metal, {
      castShadow: true,
    });
  });
  return whiteboard;
}

function createMeetingRoom() {
  const room = new THREE.Group();
  const floorY = TABLE_FLOOR_Y;
  const materials = {
    wall: new THREE.MeshStandardMaterial({ color: 0xf5f5f1, roughness: 0.92 }),
    ceiling: new THREE.MeshBasicMaterial({ color: 0xfafafa }),
    floor: new THREE.MeshStandardMaterial({ color: 0x777d80, roughness: 0.96 }),
    table: new THREE.MeshStandardMaterial({ color: 0xcfd3d2, roughness: 0.82 }),
    metal: new THREE.MeshStandardMaterial({ color: 0x777b7d, roughness: 0.45, metalness: 0.5 }),
    chair: new THREE.MeshStandardMaterial({ color: 0xb99858, roughness: 0.76 }),
    window: new THREE.MeshStandardMaterial({
      color: 0xa9c9d9,
      roughness: 0.3,
      metalness: 0.08,
      transparent: true,
      opacity: 0.84,
    }),
    frame: new THREE.MeshStandardMaterial({ color: 0xd4d7d8, roughness: 0.6 }),
    blind: new THREE.MeshStandardMaterial({ color: 0xd9d7ce, roughness: 0.82 }),
    blindSlat: new THREE.MeshStandardMaterial({ color: 0xc4c1b7, roughness: 0.78 }),
    whiteboard: new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.45 }),
    publicBoard: new THREE.MeshStandardMaterial({
      color: 0xb57b43,
      roughness: 0.78,
    }),
    publicBoardCenter: new THREE.MeshStandardMaterial({
      color: 0xe8d4a8,
      roughness: 0.86,
    }),
    light: new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0xffffff,
      emissiveIntensity: 2.6,
      roughness: 0.35,
    }),
    lightHousing: new THREE.MeshStandardMaterial({ color: 0xc8ccce, roughness: 0.62 }),
  };

  const wallCenterY = floorY + MEETING_ROOM_HEIGHT / 2;
  const ceilingY = floorY + MEETING_ROOM_HEIGHT;
  addRoomBox(
    room,
    [MEETING_ROOM_WIDTH, 0.34, MEETING_ROOM_DEPTH],
    [0, floorY - 0.17, MEETING_ROOM_CENTER_Z],
    materials.floor
  );
  addRoomBox(
    room,
    [MEETING_ROOM_WIDTH, MEETING_ROOM_HEIGHT, 0.34],
    [0, wallCenterY, MEETING_ROOM_BACK_Z],
    materials.wall
  );
  [-MEETING_ROOM_WIDTH / 2, MEETING_ROOM_WIDTH / 2].forEach((x) => {
    addRoomBox(
      room,
      [0.34, MEETING_ROOM_HEIGHT, MEETING_ROOM_DEPTH],
      [x, wallCenterY, MEETING_ROOM_CENTER_Z],
      materials.wall
    );
  });
  addRoomBox(
    room,
    [MEETING_ROOM_WIDTH, 0.28, MEETING_ROOM_DEPTH],
    [0, ceilingY, MEETING_ROOM_CENTER_Z],
    materials.ceiling
  );

  const windowY = floorY + MEETING_ROOM_HEIGHT - MEETING_WINDOW_HEIGHT / 2;
  const blindY = windowY + MEETING_WINDOW_HEIGHT / 2 - MEETING_BLIND_HEIGHT / 2;
  addRoomBox(
    room,
    [MEETING_WINDOW_WIDTH, MEETING_WINDOW_HEIGHT, 0.12],
    [0, windowY, MEETING_ROOM_BACK_Z + 0.2],
    materials.window
  );
  addRoomBox(
    room,
    [MEETING_WINDOW_WIDTH + 0.3, 0.18, 0.22],
    [0, windowY + MEETING_WINDOW_HEIGHT / 2, MEETING_ROOM_BACK_Z + 0.36],
    materials.frame
  );
  addRoomBox(
    room,
    [MEETING_WINDOW_WIDTH + 0.3, 0.18, 0.22],
    [0, windowY - MEETING_WINDOW_HEIGHT / 2, MEETING_ROOM_BACK_Z + 0.36],
    materials.frame
  );
  [
    -MEETING_WINDOW_WIDTH / 2,
    -MEETING_WINDOW_WIDTH / 6,
    MEETING_WINDOW_WIDTH / 6,
    MEETING_WINDOW_WIDTH / 2,
  ].forEach((x) => {
    addRoomBox(
      room,
      [0.18, MEETING_WINDOW_HEIGHT + 0.36, 0.22],
      [x, windowY, MEETING_ROOM_BACK_Z + 0.36],
      materials.frame
    );
  });

  MEETING_BLIND_CENTERS.forEach((x) => {
    addRoomBox(
      room,
      [MEETING_BLIND_WIDTH, MEETING_BLIND_HEIGHT, 0.13],
      [x, blindY, MEETING_ROOM_BACK_Z + 0.5],
      materials.blind
    );
    for (let index = 0; index < 9; index += 1) {
      addRoomBox(
        room,
        [MEETING_BLIND_WIDTH - 0.26, 0.08, 0.18],
        [
          x,
          blindY - MEETING_BLIND_HEIGHT / 2 + 0.22 + index * ((MEETING_BLIND_HEIGHT - 0.44) / 8),
          MEETING_ROOM_BACK_Z + 0.61,
        ],
        materials.blindSlat
      );
    }
  });
  room.add(createFreestandingWhiteboard(materials));

  MEETING_TABLE_COLUMNS.forEach((x) => {
    MEETING_TABLE_ROWS.forEach((z) => room.add(createMeetingTable(materials, x, z)));
  });
  meetingPublicTablesGroup = new THREE.Group();
  MEETING_PUBLIC_TABLE_LAYOUT.forEach((layout) => {
    meetingPublicTablesGroup.add(createMeetingPublicTable(materials, layout));
  });
  room.add(meetingPublicTablesGroup);
  addMeetingRoomSilhouette(room);

  [-7, 0, 7].forEach((x) => {
    addRoomBox(room, [0.92, 0.2, 45], [x, ceilingY - 0.28, -7], materials.lightHousing);
    addRoomBox(room, [0.56, 0.12, 44.5], [x, ceilingY - 0.42, -7], materials.light);
  });
  [-2, -14, -26].forEach((z) => {
    const light = new THREE.PointLight(0xf7fbff, 23, 20, 1.8);
    light.position.set(0, ceilingY - 1.0, z);
    room.add(light);
  });
  room.add(new THREE.AmbientLight(0xffffff, 1.25));

  room.visible = false;
  return room;
}

function createScene() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xd9c39c);

  camera = new THREE.PerspectiveCamera(environmentMode === "meeting-room" ? 58 : 34, 1, 0.1, 100);
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

  const woodTexture = createWoodTexture();
  const boardTopMaterial = new THREE.MeshStandardMaterial({
    map: woodTexture,
    roughness: 0.72,
    metalness: 0,
  });
  const boardSideMaterial = new THREE.MeshStandardMaterial({
    map: woodTexture,
    color: 0xa96f3d,
    roughness: 0.78,
    metalness: 0,
  });
  const legMaterial = new THREE.MeshStandardMaterial({
    map: woodTexture,
    color: 0x87502d,
    roughness: 0.74,
    metalness: 0,
  });
  const shelfMaterial = new THREE.MeshStandardMaterial({
    map: woodTexture,
    color: 0xa36a3b,
    roughness: 0.72,
    metalness: 0,
  });

  playAreaGroup = new THREE.Group();
  standardBoardFurniture = new THREE.Group();
  scene.add(playAreaGroup, standardBoardFurniture);

  const boardTop = new THREE.Mesh(
    new THREE.BoxGeometry(9.05, 0.54, 9.05),
    boardTopMaterial
  );
  boardTop.position.y = BOARD_TOP_Y - 0.27;
  boardTop.castShadow = true;
  boardTop.receiveShadow = true;
  playAreaGroup.add(boardTop);

  const boardBody = new THREE.Mesh(
    new THREE.BoxGeometry(8.56, BOARD_BODY_HEIGHT, 8.56),
    boardSideMaterial
  );
  boardBody.position.y = BOARD_BODY_TOP_Y - BOARD_BODY_HEIGHT / 2;
  boardBody.castShadow = true;
  boardBody.receiveShadow = true;
  standardBoardFurniture.add(boardBody);

  const lowerTrim = new THREE.Mesh(
    new THREE.BoxGeometry(8.76, 0.18, 8.76),
    legMaterial
  );
  lowerTrim.position.y = BOARD_BODY_BOTTOM_Y + 0.01;
  lowerTrim.castShadow = true;
  lowerTrim.receiveShadow = true;
  standardBoardFurniture.add(lowerTrim);

  standardBoardFurniture.add(createSideShelves(shelfMaterial, legMaterial));

  [-4.02, 4.02].forEach((x) => {
    [-4.02, 4.02].forEach((z) => {
      standardBoardFurniture.add(createBoardLeg(legMaterial, x, z));
    });
  });

  standardFloor = new THREE.Mesh(
    new THREE.PlaneGeometry(24, 24),
    new THREE.MeshStandardMaterial({
      color: 0xdac9a8,
      roughness: 1,
      metalness: 0,
    })
  );
  standardFloor.rotation.x = -Math.PI / 2;
  standardFloor.position.y = TABLE_FLOOR_Y;
  standardFloor.receiveShadow = true;
  scene.add(standardFloor);

  meetingRoomGroup = createMeetingRoom();
  scene.add(meetingRoomGroup);

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
    playAreaGroup.add(frame);
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
  playAreaGroup.add(center);

  pieceLayer = new THREE.Group();
  labelLayer = new THREE.Group();
  passLayer = new THREE.Group();
  scoreLayer = new THREE.Group();
  playAreaGroup.add(pieceLayer, labelLayer, passLayer);
  scene.add(scoreLayer);
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
  const profile = CAMERA_PROFILES[environmentMode] || CAMERA_PROFILES.board;
  const [orbitX, orbitY, orbitZ] = profile.orbitCenter;
  const [lookX, lookY, lookZ] = profile.lookTarget;
  const horizontal = Math.cos(cameraElevation) * cameraRadius;
  camera.position.set(
    orbitX + cameraPanX + Math.sin(cameraYaw) * horizontal,
    orbitY + Math.sin(cameraElevation) * cameraRadius,
    orbitZ + cameraPanZ + Math.cos(cameraYaw) * horizontal
  );
  camera.lookAt(lookX + cameraPanX, lookY, lookZ + cameraPanZ);
}

function setCameraRadius(nextRadius) {
  const profile = CAMERA_PROFILES[environmentMode] || CAMERA_PROFILES.board;
  cameraRadius = THREE.MathUtils.clamp(nextRadius, profile.minRadius, profile.maxRadius);
  updateCamera();
  renderFrame();
}

function stepCameraZoom(direction) {
  if (environmentMode !== "meeting-room") {
    setCameraRadius(cameraRadius + direction * 1.8);
    return;
  }
  const epsilon = 0.03;
  const nextRadius = direction < 0
    ? MEETING_ZOOM_STOPS.filter((radius) => radius < cameraRadius - epsilon).at(-1)
    : MEETING_ZOOM_STOPS.find((radius) => radius > cameraRadius + epsilon);
  setCameraRadius(nextRadius ?? (direction < 0 ? MEETING_ZOOM_STOPS[0] : MEETING_ZOOM_STOPS.at(-1)));
}

function panCamera(screenX, forward) {
  const step = environmentMode === "meeting-room" ? 0.65 : 0.8;
  const deltaX = screenX * Math.cos(cameraYaw) - forward * Math.sin(cameraYaw);
  const deltaZ = -screenX * Math.sin(cameraYaw) - forward * Math.cos(cameraYaw);
  if (environmentMode === "meeting-room") {
    cameraPanX += deltaX * step;
    cameraPanZ += deltaZ * step;
  } else {
    cameraPanX = THREE.MathUtils.clamp(cameraPanX + deltaX * step, -3, 3);
    cameraPanZ = THREE.MathUtils.clamp(cameraPanZ + deltaZ * step, -3, 3);
  }
  updateCamera();
  renderFrame();
}

function resetCameraForEnvironment() {
  const profile = CAMERA_PROFILES[environmentMode] || CAMERA_PROFILES.board;
  cameraYaw = profile.yaw;
  cameraElevation = profile.elevation;
  cameraRadius = profile.radius;
  cameraPanX = 0;
  cameraPanZ = 0;
  updateCamera();
}

function setEnvironment(mode) {
  const nextMode = mode === "meeting-room" ? "meeting-room" : "board";
  const changed = environmentMode !== nextMode;
  environmentMode = nextMode;
  if (scene) scene.background = new THREE.Color(nextMode === "meeting-room" ? 0xdde2e5 : 0xd9c39c);
  if (camera) {
    camera.fov = nextMode === "meeting-room" ? 58 : 34;
    camera.updateProjectionMatrix();
  }
  if (standardFloor) standardFloor.visible = nextMode === "board";
  if (meetingRoomGroup) meetingRoomGroup.visible = nextMode === "meeting-room";
  if (standardBoardFurniture) standardBoardFurniture.visible = nextMode === "board";
  if (playAreaGroup) {
    if (nextMode === "meeting-room") {
      playAreaGroup.position.set(
        MEETING_BOARD_X,
        MEETING_TABLE_TOP_Y + MEETING_BOARD_CLEARANCE - BOARD_TOP_Y * MEETING_BOARD_SCALE,
        MEETING_BOARD_Z
      );
      playAreaGroup.scale.setScalar(MEETING_BOARD_SCALE);
    } else {
      playAreaGroup.position.set(0, 0, 0);
      playAreaGroup.scale.setScalar(1);
    }
  }
  if (changed) {
    resetCameraForEnvironment();
    if (latestSnapshot) rebuildLabels(latestSnapshot);
  }
  renderFrame();
}

function getPinchDistance() {
  const points = Array.from(activePointers.values());
  if (points.length < 2) return null;
  return Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y);
}

function publicRoomIdAtPointer(event) {
  if (environmentMode !== "meeting-room" || !camera || !renderer) return "";
  const bounds = canvas.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return "";
  const pointer = new THREE.Vector2(
    ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
    -((event.clientY - bounds.top) / bounds.height) * 2 + 1
  );
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(meetingPublicTableTargets, false)[0];
  const roomId = String(hit?.object?.userData?.publicRoomId || "");
  return meetingPublicTables.get(roomId)?.root.visible ? roomId : "";
}

function aiThoughtKeyAtPointer(event) {
  if (!camera || !renderer || !pieceLayer) return "";
  const bounds = canvas.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return "";
  const pointer = new THREE.Vector2(
    ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
    -((event.clientY - bounds.top) / bounds.height) * 2 + 1
  );
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(pieceLayer.children, true)[0];
  let object = hit?.object || null;
  while (object) {
    const thoughtKey = String(object.userData?.aiThoughtKey || "");
    if (thoughtKey) return thoughtKey;
    object = object.parent;
  }
  return "";
}

function attachInteraction() {
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    setCameraRadius(cameraRadius + event.deltaY * 0.012);
  }, { passive: false });

  canvas.addEventListener("pointerdown", (event) => {
    if (activePointers.size === 0) pointerTravel = 0;
    activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    pointerState = { id: event.pointerId, x: event.clientX, y: event.clientY };
    if (activePointers.size >= 2) {
      pointerState = null;
      pinchDistance = getPinchDistance();
    }
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!activePointers.has(event.pointerId)) return;
    activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (activePointers.size >= 2) {
      const nextDistance = getPinchDistance();
      if (pinchDistance && nextDistance) {
        setCameraRadius(cameraRadius * (pinchDistance / nextDistance));
      }
      pinchDistance = nextDistance;
      return;
    }
    if (!pointerState || event.pointerId !== pointerState.id) return;
    const dx = event.clientX - pointerState.x;
    const dy = event.clientY - pointerState.y;
    pointerTravel += Math.hypot(dx, dy);
    pointerState.x = event.clientX;
    pointerState.y = event.clientY;
    const profile = CAMERA_PROFILES[environmentMode] || CAMERA_PROFILES.board;
    cameraYaw = THREE.MathUtils.clamp(cameraYaw - dx * 0.004, profile.minYaw, profile.maxYaw);
    cameraElevation = THREE.MathUtils.clamp(
      cameraElevation + dy * 0.003,
      profile.minElevation,
      profile.maxElevation
    );
    updateCamera();
    renderFrame();
  });
  const releasePointer = (event) => {
    if (activePointers.size === 1 && pointerTravel > 7) suppressNextClick = true;
    activePointers.delete(event.pointerId);
    if (activePointers.size === 1) {
      const [id, point] = activePointers.entries().next().value;
      pointerState = { id, x: point.x, y: point.y };
    } else {
      pointerState = null;
    }
    pinchDistance = activePointers.size >= 2 ? getPinchDistance() : null;
  };
  canvas.addEventListener("pointerup", releasePointer);
  canvas.addEventListener("pointercancel", releasePointer);
  canvas.addEventListener("click", (event) => {
    if (suppressNextClick) {
      suppressNextClick = false;
      return;
    }
    const thoughtKey = aiThoughtKeyAtPointer(event);
    if (thoughtKey) {
      window.dispatchEvent(new CustomEvent("goita-ai-piece-thought", {
        detail: { thoughtKey },
      }));
      return;
    }
    const roomId = publicRoomIdAtPointer(event);
    if (!roomId) return;
    window.dispatchEvent(new CustomEvent("goita-public-table-open", {
      detail: { roomId },
    }));
  });
  canvas.addEventListener("dblclick", () => {
    resetCameraForEnvironment();
    renderFrame();
  });
  zoomInButton?.addEventListener("click", () => stepCameraZoom(-1));
  zoomOutButton?.addEventListener("click", () => stepCameraZoom(1));
  panUpButton?.addEventListener("click", () => panCamera(0, 1));
  panDownButton?.addEventListener("click", () => panCamera(0, -1));
  panLeftButton?.addEventListener("click", () => panCamera(-1, 0));
  panRightButton?.addEventListener("click", () => panCamera(1, 0));
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

function pieceTextMaterial(label, englishLabel, color, opacity) {
  const cacheKey = `${label}-${englishLabel}-${color}-${opacity}`;
  if (pieceTextMaterials.has(cacheKey)) return pieceTextMaterials.get(cacheKey);

  const textureCanvas = document.createElement("canvas");
  textureCanvas.width = 256;
  textureCanvas.height = 320;
  const context = textureCanvas.getContext("2d");
  context.clearRect(0, 0, 256, 320);
  context.fillStyle = color;
  context.textAlign = "center";
  context.textBaseline = "middle";
  if (englishLabel) {
    context.font = '900 165px "Yu Kyokasho", "Yu Mincho", serif';
    context.fillText(label, 128, 136);
    context.font = '800 40px Arial, sans-serif';
    context.fillText(englishLabel, 128, 260);
  } else {
    context.font = '900 205px "Yu Kyokasho", "Yu Mincho", serif';
    context.fillText(label, 128, 170);
  }

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
  if (piece.thoughtKey) root.userData.aiThoughtKey = String(piece.thoughtKey);
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

  if ((!piece.hidden || piece.ownHidden) && piece.label) {
    const textPlane = new THREE.Mesh(
      new THREE.PlaneGeometry(0.58, 0.72),
      pieceTextMaterial(
        piece.label,
        piece.englishLabel || "",
        piece.ai ? "#a31313" : "#17130f",
        piece.ownHidden ? 0.32 : (piece.revealedHidden ? 0.55 : 1)
      )
    );
    textPlane.position.set(0, -0.05, 0.195);
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
    if (options.borderColor) {
      context.strokeStyle = options.borderColor;
      context.lineWidth = options.borderWidth || 8;
      context.stroke();
    }
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
    depthTest: options.depthTest === true,
  });
  const sprite = new THREE.Sprite(material);
  sprite.renderOrder = 20;
  if (options.trackLabel !== false) labelDisposables.push(sprite);
  return sprite;
}

function removePassMarker(marker) {
  if (!marker || !passLayer) return;
  passLayer.remove(marker);
  if (marker.material?.map) marker.material.map.dispose();
  if (marker.material) marker.material.dispose();
  renderFrame();
}

function showPass(phys) {
  if (!ensureRenderer() || !passLayer) return false;
  const position = PASS_WORLD_POSITIONS[phys];
  if (!position) return false;

  const marker = createTextSprite(tr("パス"), {
    width: 384,
    height: 168,
    background: "rgba(255, 250, 236, 0.96)",
    borderColor: "#9b2226",
    borderWidth: 10,
    color: "#9b2226",
    fontSize: 76,
    trackLabel: false,
  });
  marker.position.set(...position);
  marker.scale.set(1.55, 0.68, 1);
  marker.renderOrder = 40;
  passLayer.add(marker);
  renderFrame();

  window.setTimeout(() => removePassMarker(marker), 2000);
  return true;
}

function createFloatingScore(snapshot) {
  const width = 1024;
  const height = 360;
  const textureCanvas = document.createElement("canvas");
  textureCanvas.width = width;
  textureCanvas.height = height;
  const context = textureCanvas.getContext("2d");
  context.clearRect(0, 0, width, height);

  context.textAlign = "center";
  context.textBaseline = "middle";
  context.shadowColor = "rgba(55, 27, 11, 0.72)";
  context.shadowBlur = 8;
  context.shadowOffsetY = 4;

  context.fillStyle = "#ffe7ad";
  context.font = '800 68px "Yu Kyokasho", "Yu Mincho", serif';
  context.fillText(tr(`第 ${snapshot.round} 局`), width / 2, 58);

  context.font = '900 98px "Yu Kyokasho", "Yu Mincho", serif';
  context.fillStyle = "#ffd08c";
  context.fillText(tr(`AC  ${snapshot.scores.AC}点`), width / 2, 170);
  context.fillStyle = "#bfe1ff";
  context.fillText(tr(`BD  ${snapshot.scores.BD}点`), width / 2, 292);

  const texture = new THREE.CanvasTexture(textureCanvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const score = new THREE.Mesh(new THREE.PlaneGeometry(8.4, 2.95), material);
  if (environmentMode === "meeting-room") {
    score.rotation.x = -Math.PI / 2;
    score.position.set(
      MEETING_BOARD_X,
      MEETING_TABLE_TOP_Y + MEETING_BOARD_CLEARANCE + 0.03,
      MEETING_BOARD_Z
    );
    score.scale.setScalar(MEETING_BOARD_SCALE);
  } else {
    score.position.set(0, -0.6, -10.8);
  }
  score.renderOrder = 12;
  return score;
}

function rebuildLabels(snapshot) {
  clearLayer(labelLayer, true);
  clearLayer(scoreLayer, true);
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
    const speaking = snapshot.voiceSpeakingSeats?.includes(seat) ? "  🎙" : "";
    const countdown = snapshot.turn === seat && snapshot.turnTimeLabel
      ? `  ${snapshot.turnTimeLabel}`
      : "";
    const label = `${seat}${rawName ? `: ${rawName}` : ""}${thinking}${speaking}${countdown}`;
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
    scoreLayer.add(createFloatingScore(snapshot));
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
  latestSnapshot = snapshot;
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
  setEnvironment,
  setPublicTables,
  setVisible,
  showPass,
  getCanvas: () => canvas,
};

window.dispatchEvent(new CustomEvent("goita-board3d-ready"));
