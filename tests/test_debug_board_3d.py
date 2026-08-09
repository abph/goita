from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_3d_board_is_available_in_private_and_debug_rooms() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    chinese = (ROOT / "frontend" / "i18n.js").read_text(encoding="utf-8")
    english = (ROOT / "frontend" / "i18n-en.js").read_text(encoding="utf-8")

    assert 'id="board3d"' in html
    assert 'id="board3dCanvas"' in html
    assert 'id="board3dZoomIn"' in html
    assert 'id="board3dZoomOut"' in html
    assert 'id="board3dPanUp"' in html
    assert 'id="board3dPanDown"' in html
    assert 'id="board3dPanLeft"' in html
    assert 'id="board3dPanRight"' in html
    assert 'import("/static/board3d.js?v=20260809c")' in html
    assert 'id="boardViewSettingRow"' in html
    assert 'id="boardViewMode" type="hidden" value="2d"' in html
    assert "const PRIVATE_ROOM_IDS = new Set([" in html
    assert "PRIVATE_A_GID," in html
    assert '"room-silver-02"' in html
    assert '"room-bronze-03"' in html
    assert '"room-copper-04"' in html
    assert "function supportsAlternateBoardViews(roomId)" in html
    assert "return roomId === DEBUG_GID || roomId === MEETING_ROOM_GID || PRIVATE_ROOM_IDS.has(roomId);" in html
    assert 'return roomId === MEETING_ROOM_GID ? "meeting-room" : personalSettings.boardViewMode;' in html
    assert 'supportsAlternateBoardViews(targetGid) && targetGid !== MEETING_ROOM_GID ? "block" : "none"' in html
    assert "const boardViewsEnabled = supportsAlternateBoardViews(gid);" in html
    assert 'id="boardViewMeetingRoomButton"' in html
    assert 'id="lobbyBoardViewMeetingRoomButton"' in html
    assert "const THREE_D_BOARD_MODES = new Set([\"3d\", \"meeting-room\"]);" in html
    assert "const boardViewMode = effectiveBoardViewMode(gid);" in html
    assert 'boardViewsEnabled && THREE_D_BOARD_MODES.has(boardViewMode)' in html
    assert 'if(!supportsAlternateBoardViews(gid) || !THREE_D_BOARD_MODES.has(boardViewMode)) return;' in html
    assert 'boardViewMode: normalizeBoardViewMode(saved.boardViewMode)' in html
    assert 'boardViewMode: "2d"' in html
    assert '"集会室": "活动室"' in chinese
    assert '"集会室": "Meeting Room"' in english


def test_3d_board_uses_local_threejs_and_public_board_state() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    board_module = (ROOT / "frontend" / "board3d.js").read_text(encoding="utf-8")
    three_module = ROOT / "frontend" / "three.module.min.js"
    three_core_module = ROOT / "frontend" / "three.core.min.js"

    assert three_module.exists()
    assert three_module.stat().st_size > 300_000
    assert three_core_module.exists()
    assert three_core_module.stat().st_size > 350_000
    assert 'import * as THREE from "./three.module.min.js"' in board_module
    assert "new THREE.WebGLRenderer" in board_module
    assert "function createMeetingRoom" in board_module
    assert "function addMeetingRoomSilhouette(room)" in board_module
    assert '"/static/meeting-room-silhouette.png?v=20260806a"' in board_module
    assert "silhouette.position.set(10.15, TABLE_FLOOR_Y + 2.515, -8.2)" in board_module
    silhouette_asset = ROOT / "frontend" / "meeting-room-silhouette.png"
    assert silhouette_asset.exists()
    assert silhouette_asset.stat().st_size > 100_000
    assert 'function setEnvironment(mode)' in board_module
    assert 'setEnvironment,' in board_module
    assert "new THREE.ExtrudeGeometry" in board_module
    assert "function createBoardLeg" in board_module
    assert "function createSideShelves" in board_module
    assert "new THREE.CylinderGeometry" in board_module
    assert "const BOARD_BODY_HEIGHT = 3.44" in board_module
    assert "const SIDE_SHELF_Y = BOARD_BODY_BOTTOM_Y + 1.5" in board_module
    assert "const SIDE_SHELF_LENGTH = 5.2" in board_module
    assert "const SIDE_SHELF_DEPTH = 2.36" in board_module
    assert "position: [0, SIDE_SHELF_Y, -SIDE_SHELF_OFFSET]" in board_module
    assert "function createFloatingScore(snapshot)" in board_module
    assert "new THREE.Mesh(new THREE.PlaneGeometry(8.4, 2.95), material)" in board_module
    assert "context.fillText(tr(`AC  ${snapshot.scores.AC}点`), width / 2, 170)" in board_module
    assert "context.fillText(tr(`BD  ${snapshot.scores.BD}点`), width / 2, 292)" in board_module
    assert "score.position.set(0, -0.6, -10.8)" in board_module
    assert "scoreLayer.add(createFloatingScore(snapshot))" in board_module
    assert "score.position.set(0, 0.73, 0)" not in board_module
    assert "new THREE.BoxGeometry(8.56, BOARD_BODY_HEIGHT, 8.56)" in board_module
    assert "const TABLE_FLOOR_Y = BOARD_BODY_BOTTOM_Y - 0.99" in board_module
    assert "shape.lineTo(0.35, 0.34)" in board_module
    assert "shape.lineTo(0, 0.5)" in board_module
    assert "quadraticCurveTo" not in board_module
    assert "textPlane.position.set(0, -0.05, 0.195)" in board_module
    assert 'context.font = \'900 205px "Yu Kyokasho", "Yu Mincho", serif\'' in board_module
    assert "context.font = '800 40px Arial, sans-serif'" in board_module
    assert "ResizeObserver" in board_module
    assert 'canvas.addEventListener("wheel"' in board_module
    assert "pointermove" in board_module
    assert "function getPinchDistance()" in board_module
    assert "radius: DEFAULT_CAMERA_RADIUS" in board_module
    assert '"meeting-room": {' in board_module
    assert "meetingRoomGroup = createMeetingRoom()" in board_module
    assert "const MEETING_ROOM_DEPTH = 60" in board_module
    assert "const MEETING_ROOM_WIDTH = 24" in board_module
    assert "const MEETING_WINDOW_WIDTH = MEETING_ROOM_WIDTH * 0.94" in board_module
    assert "const MEETING_WINDOW_HEIGHT = MEETING_ROOM_HEIGHT * 0.6" in board_module
    assert "const MEETING_BLIND_HEIGHT = MEETING_WINDOW_HEIGHT * 0.5" in board_module
    assert "MEETING_TABLE_COLUMNS.forEach((x) =>" in board_module
    assert "MEETING_TABLE_ROWS.forEach((z)" in board_module
    assert "const MEETING_PUBLIC_TABLE_LAYOUT" in board_module
    assert 'const MEETING_BOARD_Z = -14.6' in board_module
    assert '{ roomId: "main", x: -5.1, z: -24.4 }' in board_module
    assert '{ roomId: "main-b", x: 5.1, z: -24.4 }' in board_module
    assert '{ roomId: "main-c", x: -5.1, z: -14.6 }' in board_module
    assert '{ roomId: "main-e", x: -5.1, z: -4.8 }' in board_module
    assert '{ roomId: "main-f", x: 5.1, z: -4.8 }' in board_module
    assert "function createMeetingPublicTable" in board_module
    assert "function setPublicTables(snapshots = [])" in board_module
    assert 'window.dispatchEvent(new CustomEvent("goita-public-table-open"' in board_module
    assert "setPublicTables," in board_module
    assert "MEETING_BLIND_CENTERS.forEach((x)" in board_module
    assert "[-7, 0, 7].forEach((x)" in board_module
    assert "function createFreestandingWhiteboard" in board_module
    assert "materials.blind" in board_module
    assert "materials.whiteboard" in board_module
    assert "materials.light" in board_module
    assert "playAreaGroup.add(pieceLayer, labelLayer, passLayer)" in board_module
    assert 'standardBoardFurniture.visible = nextMode === "board"' in board_module
    assert "playAreaGroup.scale.setScalar(MEETING_BOARD_SCALE)" in board_module
    assert "MEETING_TABLE_TOP_Y + MEETING_BOARD_CLEARANCE - BOARD_TOP_Y * MEETING_BOARD_SCALE" in board_module
    assert "score.rotation.x = -Math.PI / 2" in board_module
    assert "minRadius: 2.9" in board_module
    assert "maxRadius: 11.8" in board_module
    assert "maxElevation: 1.05" in board_module
    assert "const MEETING_ZOOM_STOPS = [2.9, 3.5, 4.05, 4.6, 6.4, 8.2, 10, 11.8]" in board_module
    assert "function stepCameraZoom(direction)" in board_module
    assert "function panCamera(screenX, forward)" in board_module
    assert "cameraPanX += deltaX * step" in board_module
    assert "cameraPanZ += deltaZ * step" in board_module
    assert 'const limit = environmentMode === "meeting-room"' not in board_module
    assert "lookX + cameraPanX" in board_module
    assert 'zoomInButton?.addEventListener("click"' in board_module
    assert 'zoomOutButton?.addEventListener("click"' in board_module
    assert 'panLeftButton?.addEventListener("click"' in board_module
    assert 'panRightButton?.addEventListener("click"' in board_module
    assert "requestAnimationFrame(animatePieces)" in board_module
    assert "state.board_public" in html
    assert "buildBoard3DSnapshot(state)" in html
    assert "piece.ownHidden ? 0.32" in board_module
    assert 'document.body.classList.contains("board-view-3d")' in html
    assert 'window.goitaBoard3D?.showPass?.(phys)' in html
    assert "const PASS_WORLD_POSITIONS" in board_module
    assert "A: [0, 1.12, 3.12]" in board_module
    assert "B: [3.12, 1.12, 0]" in board_module
    assert "C: [0, 1.12, -3.12]" in board_module
    assert "D: [-3.12, 1.12, 0]" in board_module
    assert "function showPass(phys)" in board_module
    assert "passLayer.add(marker)" in board_module
    assert "showPass," in board_module
    assert 'wrapper.classList.add("pass-anim-overlay"' in html


if __name__ == "__main__":
    test_3d_board_is_available_in_private_and_debug_rooms()
    test_3d_board_uses_local_threejs_and_public_board_state()
    print("DEBUG_BOARD_3D_TEST_OK")
