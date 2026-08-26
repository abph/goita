"""Protect full-hand kifu downloads during play and share them after the round."""

from __future__ import annotations

from fastapi import HTTPException

import backend.app as app_module


def _download_test_game():
    game = app_module._create_game_obj(dealer="A")
    game["is_started"] = True
    game["human_seats"] = {"A": "owner-client"}
    return game


def test_kifu_download_allows_non_participants_after_round() -> None:
    game_id = "test-kifu-download-access"
    game = _download_test_game()
    game["state"].finished = True
    app_module.GAMES[game_id] = game
    try:
        response = app_module.get_kifu_yaml(
            game_id,
            client_id="outsider-client",
        )
        assert 'p0: "' in response
        assert 'p3: "' in response
    finally:
        app_module.GAMES.pop(game_id, None)


def test_kifu_download_rejects_everyone_during_play() -> None:
    game_id = "test-kifu-download-in-progress"
    app_module.GAMES[game_id] = _download_test_game()
    try:
        for client_id in ("owner-client", "outsider-client", ""):
            try:
                app_module.get_kifu_yaml(game_id, client_id=client_id)
            except HTTPException as error:
                assert error.status_code == 409
            else:
                raise AssertionError("An active round exposed every initial hand")
    finally:
        app_module.GAMES.pop(game_id, None)


def test_kifu_download_allows_spectator_without_client_id_after_round() -> None:
    game_id = "test-kifu-download-spectator"
    game = _download_test_game()
    game["state"].finished = True
    app_module.GAMES[game_id] = game
    try:
        response = app_module.get_kifu_yaml(game_id)
        assert 'p0: "' in response
        assert 'p3: "' in response
    finally:
        app_module.GAMES.pop(game_id, None)


def test_kifu_download_allows_participant_after_round() -> None:
    game_id = "test-kifu-download-finished"
    game = _download_test_game()
    game["state"].finished = True
    app_module.GAMES[game_id] = game
    try:
        response = app_module.get_kifu_yaml(
            game_id,
            anonymous=True,
            client_id="owner-client",
        )
        assert 'p0: "' in response
        assert 'p3: "' in response
    finally:
        app_module.GAMES.pop(game_id, None)


def test_frontend_sends_client_identity_for_kifu_download() -> None:
    html = (app_module.FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    assert "client_id: clientId" in html
    assert "if(!res.ok)" in html


if __name__ == "__main__":
    test_kifu_download_allows_non_participants_after_round()
    test_kifu_download_rejects_everyone_during_play()
    test_kifu_download_allows_spectator_without_client_id_after_round()
    test_kifu_download_allows_participant_after_round()
    test_frontend_sends_client_identity_for_kifu_download()
    print("KIFU_DOWNLOAD_ACCESS_TEST_OK")
