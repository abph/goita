import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from backend import app as app_module


def _shared_record():
    return {
        "id": "K-test",
        "title": "終盤の研究",
        "memo": "共有してはいけない個人メモ",
        "tags": ["し攻め"],
        "favorite": True,
        "payload": {
            "round_index": 2,
            "dealer": "C",
            "winner": "A",
            "hand": {
                "p0": "ししし香馬銀金飛",
                "p1": "しし香香馬銀金角",
                "p2": "ししし馬馬銀金王",
                "p3": "し香香馬銀金角玉",
            },
            "moves": [["2", "馬", "銀"]],
            "player_names": {"A": "一郎", "B": "二郎", "C": "三郎", "D": "四郎"},
            "my_seat": "B",
        },
    }


def test_seated_member_can_share_kifu_without_library_metadata(monkeypatch) -> None:
    async def scenario() -> None:
        game_id = app_module.PRIVATE_A_GID
        previous = app_module.GAMES.get(game_id)
        game = app_module._create_game_obj()
        game["human_seats"] = {"A": "host-client", "B": "member-client", "C": "other-client"}
        game["player_names"].update({"A": "ホスト", "B": "共有者", "C": "参加者"})
        app_module.GAMES[game_id] = game
        monkeypatch.setattr(app_module.MEMBER_KIFU_STORE, "access", lambda *_: _shared_record())
        monkeypatch.setattr(app_module, "require_member_origin", lambda _request: None)
        try:
            result = await app_module.share_member_kifu(
                "K-test",
                app_module.SharedKifuRequest(
                    game_id=game_id,
                    client_id="member-client",
                    seat="B",
                ),
                SimpleNamespace(cookies={app_module.MEMBER_COOKIE: "member-token"}),
            )
            shared = result["shared_kifu"]
            assert shared["title"] == "終盤の研究"
            assert shared["shared_by"] == "共有者"
            assert shared["shared_by_seat"] == "B"
            assert shared["payload"]["my_seat"] == ""
            assert "memo" not in shared
            assert "tags" not in shared
            assert "favorite" not in shared
            assert "owner_client_id" not in shared

            host_view = app_module._state_public_view(
                game["state"], game_id=game_id, viewer="A", game_obj=game, client_id="host-client"
            )
            sharer_view = app_module._state_public_view(
                game["state"], game_id=game_id, viewer="B", game_obj=game, client_id="member-client"
            )
            other_view = app_module._state_public_view(
                game["state"], game_id=game_id, viewer="C", game_obj=game, client_id="other-client"
            )
            outsider_view = app_module._state_public_view(
                game["state"], game_id=game_id, viewer="W", game_obj=game, client_id="outsider-client"
            )
            assert host_view["shared_kifu_can_stop"] is True
            assert sharer_view["shared_kifu_can_stop"] is True
            assert other_view["shared_kifu_can_stop"] is False
            assert other_view["shared_kifu"]["payload"]["hand"]["p0"]
            assert outsider_view["shared_kifu"] is None
        finally:
            if previous is None:
                app_module.GAMES.pop(game_id, None)
            else:
                app_module.GAMES[game_id] = previous

    asyncio.run(scenario())


def test_share_is_private_room_seated_and_between_rounds_only(monkeypatch) -> None:
    async def scenario() -> None:
        game_id = app_module.PRIVATE_A_GID
        previous = app_module.GAMES.get(game_id)
        game = app_module._create_game_obj()
        game["human_seats"] = {"B": "member-client"}
        game["is_started"] = True
        app_module.GAMES[game_id] = game
        monkeypatch.setattr(app_module.MEMBER_KIFU_STORE, "access", lambda *_: _shared_record())
        monkeypatch.setattr(app_module, "require_member_origin", lambda _request: None)
        request = SimpleNamespace(cookies={app_module.MEMBER_COOKIE: "member-token"})
        try:
            with pytest.raises(HTTPException) as active_round:
                await app_module.share_member_kifu(
                    "K-test",
                    app_module.SharedKifuRequest(game_id=game_id, client_id="member-client", seat="B"),
                    request,
                )
            assert active_round.value.status_code == 409

            game["is_started"] = False
            with pytest.raises(HTTPException) as wrong_owner:
                await app_module.share_member_kifu(
                    "K-test",
                    app_module.SharedKifuRequest(game_id=game_id, client_id="other-client", seat="B"),
                    request,
                )
            assert wrong_owner.value.status_code == 403

            with pytest.raises(HTTPException) as public_room:
                await app_module.share_member_kifu(
                    "K-test",
                    app_module.SharedKifuRequest(game_id=app_module.MAIN_GID, client_id="member-client", seat="B"),
                    request,
                )
            assert public_room.value.status_code == 403
        finally:
            if previous is None:
                app_module.GAMES.pop(game_id, None)
            else:
                app_module.GAMES[game_id] = previous

    asyncio.run(scenario())


def test_sharer_or_host_can_end_sharing_and_start_clears_it(monkeypatch) -> None:
    async def scenario() -> None:
        game_id = app_module.PRIVATE_A_GID
        previous = app_module.GAMES.get(game_id)
        game = app_module._create_game_obj()
        game["human_seats"] = {"A": "host-client", "B": "member-client", "C": "other-client"}
        game["shared_kifu"] = {
            "token": "shared-token",
            "owner_client_id": "member-client",
            "payload": _shared_record()["payload"],
        }
        app_module.GAMES[game_id] = game
        monkeypatch.setattr(app_module, "require_member_origin", lambda _request: None)
        request = SimpleNamespace()
        try:
            with pytest.raises(HTTPException) as unrelated:
                await app_module.stop_shared_kifu(
                    game_id,
                    app_module.SharedKifuStopRequest(client_id="other-client", seat="C"),
                    request,
                )
            assert unrelated.value.status_code == 403

            stopped = await app_module.stop_shared_kifu(
                game_id,
                app_module.SharedKifuStopRequest(client_id="member-client", seat="B"),
                request,
            )
            assert stopped == {"ok": True}
            assert game["shared_kifu"] is None

            game["shared_kifu"] = {"token": "leave-token", "owner_client_id": "member-client", "payload": {}}
            await app_module.release_seat(game_id, seat="B", client_id="member-client")
            assert game["shared_kifu"] is None

            game["human_seats"]["B"] = "member-client"
            game["shared_kifu"] = {"token": "next-token", "owner_client_id": "member-client", "payload": {}}
            started = await app_module.start_game(game_id, requester="A", client_id="host-client")
            assert started == {"ok": True}
            assert game["shared_kifu"] is None
        finally:
            app_module._cancel_turn_timeout_task(game_id)
            if previous is None:
                app_module.GAMES.pop(game_id, None)
            else:
                app_module.GAMES[game_id] = previous

    asyncio.run(scenario())


def test_shared_kifu_viewer_replaces_live_board_with_numbered_final_board() -> None:
    html = (Path(__file__).resolve().parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")

    assert 'id="researchKifuShareButton"' in html
    assert 'onclick="shareSelectedResearchKifu()"' in html
    assert 'id="sharedKifuStage" class="shared-kifu-stage"' in html
    assert 'id="sharedKifuBoard" class="research-kifu-board"' in html
    assert 'id="sharedKifuModal"' not in html
    assert 'id="sharedKifuReplayButton"' not in html
    assert "shared-kifu-board-active" in html
    assert "body.shared-kifu-board-active #handsArea { display: none; }" in html
    assert html.index('id="sharedKifuStage"') < html.index('id="handsArea"')
    assert "function syncSharedKifuViewer(state)" in html
    assert "attackSequenceNumber += 1" in html
    assert "research-kifu-move-number seat-${slot.seat}" in html
    assert ".research-kifu-move-number.seat-A { top: 0;" in html
    assert ".research-kifu-move-number.seat-B { top: 50%; left: 0;" in html
    assert ".research-kifu-move-number.seat-C { bottom: 0;" in html
    assert ".research-kifu-move-number.seat-D { top: 50%; right: 0;" in html
    assert "/shared_kifu/stop" in html
    assert "state?.shared_kifu" in html
