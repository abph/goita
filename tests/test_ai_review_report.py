import asyncio
import copy
import json
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi import HTTPException

from backend import app as app_module
from backend.ai_review_report import build_review_snapshot, log_turn_numbers
from goita_ai2.state import GoitaState


def make_game():
    hands = {"A": list("73124569"), "B": list("11112345"),
             "C": list("31112456"), "D": list("17123458")}
    actions = [
        ("C", ("attack_after_block", "3", "1")),
        ("D", ("receive", "1", None)), ("D", ("attack", None, "7")),
        ("A", ("receive", "7", None)), ("A", ("attack", None, "3")),
        ("B", ("pass", None, None)),
    ]
    game = {"state": GoitaState(hands, dealer="C"), "init_hands": hands, "dealer": "C",
            "log": ["Game start. dealer=C"], "kifu_moves": [], "round_count": 2,
            "member_kifu_round_id": "review-round", "ai_profile": "current", "agents": {},
            "is_debug_room": True, "is_started": True, "human_seats": {"A": "review-host"}}
    for seat, action in actions:
        assert action in game["state"].legal_actions(seat)
        app_module._apply_action(game["state"], seat, action)
        line = app_module._format_action(seat, action)
        if action[0] == "attack_after_block":
            line += " (hidden)"
        if seat == "D":
            line += " [AI:score_fallback/test_reason]"
        if action == ("attack", None, "7"):
            candidates = {"chosen": {"attack": "7", "score": 100}, "alternatives": [{"attack": "1", "score": 80}]}
            line += f" [AI-CANDIDATES:{quote(json.dumps(candidates))}]"
        game["log"].append(line)
        game["kifu_moves"].append(app_module._action_to_kifu_row(seat, action))
    return game


def build(game):
    return build_review_snapshot(game, apply_action=app_module._apply_action,
                                 new_board=app_module._new_board_snapshot,
                                 update_board=app_module._update_board_snapshot)


def test_turn_numbers_include_pass_and_keep_receive_attack_together():
    log = make_game()["log"]
    assert log_turn_numbers(log) == [None, 1, 2, 2, 3, 3, 4]
    assert log_turn_numbers(log[:3]) == [None, 1, 2]  # Attack is still pending.
    assert log_turn_numbers(log + ["Round finished.", "Game start. dealer=C", "C: pass"])[-3:] == [None, None, 1]
    assert log_turn_numbers(log + ["C: pass", "D: pass", "A: block 1 -> attack 2"])[-3:] == [5, 6, 7]


def test_snapshot_replays_each_decision_and_preserves_original_telemetry():
    game = make_game()
    original = copy.deepcopy(game)
    report = build(game)
    receive, attack = report["decisions"][1:3]
    assert receive["turn_number"] == attack["turn_number"] == 2
    assert receive["before"]["hands"]["D"].count("1") == 2
    assert attack["before"]["hands"]["D"].count("1") == 1
    assert receive["before"]["phase"] == "receive"
    assert attack["before"]["phase"] == "attack"
    assert attack["candidate_evaluations"]["chosen"]["attack"] == "7"
    assert receive["candidate_record"] == "not_recorded"
    assert report["decisions"][0]["decision_reason"] is None
    assert attack["before"]["board"]["C"]["receive_hidden"][0] is True
    assert game["state"].__dict__ == original["state"].__dict__
    assert game["log"] == original["log"]
    game["log"].append("C: pass")
    game["init_hands"]["A"].clear()
    assert report["log"] == original["log"]
    assert len(report["initial_hands"]["A"]) == 8


def test_incomplete_or_illegal_history_cannot_be_exported_as_replayable():
    game = make_game()
    game["log"] = game["log"][:-1]
    with pytest.raises(ValueError, match="一致しない"):
        build(game)
    game = make_game()
    game["log"][1] = "C: receive 9"
    with pytest.raises(ValueError, match="再現できません"):
        build(game)


def test_endpoint_requires_debug_host_and_same_round(monkeypatch):
    game = make_game()
    monkeypatch.setitem(app_module.GAMES, app_module.DEBUG_GID, game)
    async def scenario():
        response = await app_module.ai_review_snapshot(app_module.DEBUG_GID, "review-round", "review-host")
        assert response.headers["cache-control"] == "no-store"
        assert json.loads(response.body)["decisions"][-1]["turn_number"] == 4
        for room, round_id, client, status in [
            ("main", "review-round", "review-host", 403),
            (app_module.DEBUG_GID, "review-round", "other-client", 403),
            (app_module.DEBUG_GID, "previous-round", "review-host", 409),
        ]:
            with pytest.raises(HTTPException) as error:
                await app_module.ai_review_snapshot(room, round_id, client)
            assert error.value.status_code == status
    asyncio.run(scenario())


if __name__ == "__main__":
    path = Path("results/ai_review_report/sample.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build(make_game()), ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
