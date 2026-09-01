from __future__ import annotations

from pathlib import Path

import backend.app as app_module


ROOT = Path(__file__).resolve().parents[1]


def test_player_tags_are_whitelisted() -> None:
    assert app_module._sanitize_player_tag("beginner") == "beginner"
    assert app_module._sanitize_player_tag(" TOURNAMENT ") == "tournament"
    assert app_module._sanitize_player_tag("free text") == ""


def test_player_tag_controls_and_transport_are_present() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert 'id="lobbyPersonalPlayerTag"' in html
    assert 'id="personalPlayerTag"' in html
    for value in (
        "beginner",
        "human_match",
        "ai_practice",
        "spectator",
        "teacher",
        "tournament",
    ):
        assert f'<option value="{value}">' in html
    assert 'wsParams.set("tag", personalSettings.playerTag)' in html
    assert 'tag: normalizePlayerTag(personalSettings.playerTag)' in html
    assert 'const playerTags = state.player_tags' in html
    assert 'buildPlayerTagBadge(item?.tag, "chat-player-tag")' not in html
    assert 'buildPlayerTagBadge(person?.tag, "lobby-player-tag")' in html


def test_player_tags_are_not_shown_in_seat_controls_or_lobby_room_cards() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    backend = (ROOT / "backend" / "app.py").read_text(encoding="utf-8")
    room_cards = html.split("async function fetchRoomList()", 1)[1].split(
        "function togglePassInput", 1
    )[0]
    seat_buttons = html.split("function updateSeatButtons(state)", 1)[1].split(
        "async function selectSpectator", 1
    )[0]

    assert "seat_tags" not in backend
    assert "playerTagLabel" not in room_cards
    assert "playerTagLabel" not in seat_buttons


def test_player_tag_labels_are_translated() -> None:
    chinese = (ROOT / "frontend" / "i18n.js").read_text(encoding="utf-8")
    english = (ROOT / "frontend" / "i18n-en.js").read_text(encoding="utf-8")

    for source in (
        "プレイヤータグ",
        "タグなし",
        "初心者",
        "対人希望",
        "AI練習中",
        "観戦中心",
        "教えられます",
        "大会練習中",
    ):
        assert f'"{source}":' in chinese
        assert f'"{source}":' in english


def test_new_games_initialize_empty_player_tags() -> None:
    game = app_module._create_game_obj()
    assert game["player_tags"] == {seat: "" for seat in app_module.ALL_SEATS}
