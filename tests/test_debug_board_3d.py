from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_3d_board_is_debug_only_and_defaults_to_2d() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert 'id="board3d"' in html
    assert 'id="board3dCanvas"' in html
    assert 'import("/static/board3d.js?v=20260730")' in html
    assert 'id="boardViewSettingRow"' in html
    assert 'id="boardViewMode" type="hidden" value="2d"' in html
    assert 'targetGid === DEBUG_GID ? "block" : "none"' in html
    assert 'gid === DEBUG_GID && personalSettings.boardViewMode === "3d"' in html
    assert 'boardViewMode: saved.boardViewMode === "3d" ? "3d" : "2d"' in html
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
    assert "ResizeObserver" in board_module
    assert "pointermove" in board_module
    assert "requestAnimationFrame(animatePieces)" in board_module
    assert "state.board_public" in html
    assert "buildBoard3DSnapshot(state)" in html


if __name__ == "__main__":
    test_3d_board_is_debug_only_and_defaults_to_2d()
    test_3d_board_uses_local_threejs_and_public_board_state()
    print("DEBUG_BOARD_3D_TEST_OK")
