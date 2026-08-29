from __future__ import annotations

import asyncio

from fastapi import HTTPException

from backend import app as app_module


def _expect_http_error(status_code: int, awaitable) -> None:
    try:
        asyncio.run(awaitable)
    except HTTPException as error:
        assert error.status_code == status_code
    else:
        raise AssertionError(f"Expected HTTP {status_code}")


def test_private_room_admin_can_vacate_a_human_seat() -> None:
    game_id = app_module.PRIVATE_A_GID
    previous_game = app_module.GAMES.get(game_id)
    game = app_module._create_game_obj(dealer="A")
    game["admin_password"] = "seat-admin"
    game["admin_password_hash"] = ""
    game["human_seats"] = {"A": "client-a", "B": "client-b"}
    game["player_names"] = {"A": "山田", "B": "佐藤", "C": "", "D": ""}
    game["player_tags"] = {"A": "beginner", "B": "teacher", "C": "", "D": ""}
    app_module.GAMES[game_id] = game

    try:
        verified = app_module.verify_admin(game_id, "seat-admin")
        seats = verified["managed_human_seats"]
        assert [(item["seat"], item["name"]) for item in seats] == [
            ("A", "山田"),
            ("B", "佐藤"),
        ]
        assert all("client_id" not in item for item in seats)
        target = next(item for item in seats if item["seat"] == "B")

        _expect_http_error(
            401,
            app_module.admin_vacate_seat(
                game_id,
                app_module.AdminVacateSeatRequest(
                    admin_password="wrong",
                    seat="B",
                    occupancy_token=target["occupancy_token"],
                ),
            ),
        )
        _expect_http_error(
            409,
            app_module.admin_vacate_seat(
                game_id,
                app_module.AdminVacateSeatRequest(
                    admin_password="seat-admin",
                    seat="B",
                    occupancy_token="stale-token",
                ),
            ),
        )

        result = asyncio.run(
            app_module.admin_vacate_seat(
                game_id,
                app_module.AdminVacateSeatRequest(
                    admin_password="seat-admin",
                    seat="B",
                    occupancy_token=target["occupancy_token"],
                ),
            )
        )
        assert result["vacated_seat"] == "B"
        assert [item["seat"] for item in result["managed_human_seats"]] == ["A"]
        assert game["human_seats"] == {"A": "client-a"}
        assert game["player_names"]["B"] == ""
        assert game["player_tags"]["B"] == ""
        assert game["chat_messages"][-1]["message"] == "管理者がB席を空けました。"
    finally:
        if previous_game is None:
            app_module.GAMES.pop(game_id, None)
        else:
            app_module.GAMES[game_id] = previous_game


def test_admin_vacate_is_private_room_only() -> None:
    _expect_http_error(
        403,
        app_module.admin_vacate_seat(
            app_module.MAIN_GID,
            app_module.AdminVacateSeatRequest(
                admin_password="anything",
                seat="A",
                occupancy_token="token",
            ),
        ),
    )


def test_frontend_exposes_private_admin_seat_management() -> None:
    html = app_module.FRONTEND_DIR.joinpath("index.html").read_text(encoding="utf-8")
    assert 'id="adminSeatManagementDetails"' in html
    assert 'id="adminSeatManagementList"' in html
    assert "PRIVATE_ROOM_IDS.has(settingTargetGid)" in html
    assert "/admin_vacate_seat" in html
    assert "occupancy_token: item.occupancy_token" in html


if __name__ == "__main__":
    test_private_room_admin_can_vacate_a_human_seat()
    test_admin_vacate_is_private_room_only()
    test_frontend_exposes_private_admin_seat_management()
    print("PRIVATE_ROOM_ADMIN_VACATE_TEST_OK")
