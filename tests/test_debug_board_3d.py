from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_3d_board_is_debug_only_and_defaults_to_2d() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert 'id="board3d"' in html
    assert 'id="board3dCanvas"' in html
    assert 'id="board3dZoomIn"' in html
    assert 'id="board3dZoomOut"' in html
    assert 'import("/static/board3d.js?v=20260730s")' in html
    assert 'id="boardViewSettingRow"' in html
    assert 'id="boardViewMode" type="hidden" value="2d"' in html
    assert 'targetGid === DEBUG_GID ? "block" : "none"' in html
    assert 'gid === DEBUG_GID && personalSettings.boardViewMode === "3d"' in html
    assert 'boardViewMode: normalizeBoardViewMode(saved.boardViewMode)' in html
    assert 'boardViewMode: "2d"' in html


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
    assert "context.fillText(`AC  ${snapshot.scores.AC}点`, width / 2, 170)" in board_module
    assert "context.fillText(`BD  ${snapshot.scores.BD}点`, width / 2, 292)" in board_module
    assert "score.position.set(0, -0.6, -10.8)" in board_module
    assert "labelLayer.add(createFloatingScore(snapshot))" in board_module
    assert "score.position.set(0, 0.73, 0)" not in board_module
    assert "new THREE.BoxGeometry(8.56, BOARD_BODY_HEIGHT, 8.56)" in board_module
    assert "const TABLE_FLOOR_Y = BOARD_BODY_BOTTOM_Y - 0.99" in board_module
    assert "shape.lineTo(0.35, 0.34)" in board_module
    assert "shape.lineTo(0, 0.5)" in board_module
    assert "quadraticCurveTo" not in board_module
    assert "textPlane.position.set(0, -0.05, 0.195)" in board_module
    assert "ResizeObserver" in board_module
    assert 'canvas.addEventListener("wheel"' in board_module
    assert "pointermove" in board_module
    assert "function getPinchDistance()" in board_module
    assert "cameraRadius = DEFAULT_CAMERA_RADIUS" in board_module
    assert 'zoomInButton?.addEventListener("click"' in board_module
    assert 'zoomOutButton?.addEventListener("click"' in board_module
    assert "requestAnimationFrame(animatePieces)" in board_module
    assert "state.board_public" in html
    assert "buildBoard3DSnapshot(state)" in html
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
    test_3d_board_is_debug_only_and_defaults_to_2d()
    test_3d_board_uses_local_threejs_and_public_board_state()
    print("DEBUG_BOARD_3D_TEST_OK")
