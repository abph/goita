from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_lobby_uses_a_faint_diagonal_piece_animation() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert 'id="lobbyPieceBackground"' in html
    assert "function setupLobbyPieceBackground()" in html
    assert 'const pieces = ["し", "香", "馬", "銀", "金", "角", "飛", "玉", "王"];' in html
    assert "grid-template-columns: repeat(22, 1fr)" in html
    assert "grid-template-rows: repeat(18, 1fr)" in html
    assert "opacity: 0.055" in html
    assert "animation: lobby-piece-drift 14s linear infinite" in html
    assert "font-size: clamp(32px, 3.8vmax, 60px)" in html
    assert "@keyframes lobby-piece-drift" in html
    assert "translate3d(-8.1818vmax, 10vmax, 0)" in html
    assert "@media (prefers-reduced-motion: reduce)" in html
    assert 'url("/static/lobby-background.webp' not in html


if __name__ == "__main__":
    test_lobby_uses_a_faint_diagonal_piece_animation()
    print("LOBBY_BACKGROUND_ANIMATION_TEST_OK")
