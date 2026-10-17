from __future__ import annotations

from pathlib import Path

import backend.app as app_module


ROOT = Path(__file__).resolve().parents[1]


def test_player_icons_are_whitelisted() -> None:
    assert app_module._sanitize_player_icon("shi_sparrow") == "shi_sparrow"
    assert app_module._sanitize_player_icon(" SPECIAL_YELLOWTAIL ") == "special_yellowtail"
    assert app_module._sanitize_player_icon("piece_shi") == "piece_shi"
    assert app_module._sanitize_player_icon(" PIECE_OU ") == "piece_ou"
    assert app_module._sanitize_player_icon("https://example.com/icon.png") == ""
    assert app_module._sanitize_player_icon("../frontend/player-icons/a.png") == ""


def test_new_game_and_public_state_include_player_icons() -> None:
    game = app_module._create_game_obj()
    assert game["player_icons"] == {seat: "" for seat in app_module.ALL_SEATS}

    game["human_seats"] = {"A": "client-a"}
    game["player_icons"]["A"] = "ko_mouse"
    public = app_module._state_public_view(
        game["state"],
        game_id="icon-test",
        viewer="A",
        game_obj=game,
        client_id="client-a",
    )
    assert public["player_icons"]["A"] == "ko_mouse"

    game["player_icons"]["A"] = "invalid"
    public = app_module._state_public_view(
        game["state"],
        game_id="icon-test",
        viewer="A",
        game_obj=game,
        client_id="client-a",
    )
    assert public["player_icons"]["A"] == ""


def test_clearing_a_seat_clears_all_player_identity_fields() -> None:
    game = app_module._create_game_obj()
    game["player_names"]["B"] = "利用者"
    game["player_tags"]["B"] = "beginner"
    game["player_icons"]["B"] = "gyoku_kitten"

    app_module._clear_player_name(game, "B")

    assert game["player_names"]["B"] == ""
    assert game["player_tags"]["B"] == ""
    assert game["player_icons"]["B"] == ""


def test_frontend_has_all_icon_assets_and_safe_transport() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    icon_dir = ROOT / "frontend" / "player-icons"
    expected_files = {
        "shi-sparrow-pixel.png",
        "ko-mouse-pixel.png",
        "uma-horse-pixel.png",
        "gin-baby-squirrel-pixel.png",
        "kin-flying-squirrel-pixel.png",
        "kaku-dragon-pixel.png",
        "hi-winged-dragon-pixel.png",
        "gyoku-kitten-pixel.png",
        "ou-lynx-pixel.png",
        "special-yellowtail-pixel.png",
    }

    assert expected_files == {path.name for path in icon_dir.glob("*.png")}
    assert 'id="lobbyNamePlayerIcon"' in html
    assert 'id="lobbyPersonalPlayerIcon"' in html
    assert 'id="personalPlayerIcon"' in html
    assert 'wsParams.set("icon", playerIcon)' in html
    assert 'icon: normalizePlayerIcon(personalSettings.playerIcon)' in html
    assert "const playerIcons = state.player_icons" in html
    assert "visibleChatSenderIcon(item, viewerKind)" in html
    for piece_icon in (
        "piece_shi",
        "piece_ko",
        "piece_uma",
        "piece_gin",
        "piece_kin",
        "piece_kaku",
        "piece_hi",
        "piece_gyoku",
        "piece_ou",
    ):
        assert piece_icon in html
        assert piece_icon in app_module.PLAYER_ICON_VALUES
    assert 'appendGroup("動物", "animal")' in html
    assert 'appendGroup("駒", "piece")' in html
    assert "player-piece-icon" in html
