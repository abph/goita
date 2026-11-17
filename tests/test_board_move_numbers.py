"""Numbers describe public attack order, never pass count or concealed pieces."""
import copy

from backend.board_move_numbers import board_attack_numbers


def test_receive_and_attack_are_one_number_and_passes_do_not_count():
    moves = [["2", "馬", "し"], ["3", "し", ""], ["3", "", "飛"],
             ["0", "飛", ""], ["0", "", "馬"], ["1", "パス", ""],
             ["2", "パス", ""], ["3", "パス", ""], ["0", "し", "金"]]
    numbers = board_attack_numbers(moves)
    assert numbers == {"A": [3, 4, None, None], "B": [None] * 4,
                       "C": [1, None, None, None], "D": [2, None, None, None]}
    assert board_attack_numbers(moves[:2])["D"] == [None] * 4


def test_new_round_starts_at_one_and_does_not_mutate_recorded_moves():
    first = [["3", "し", "香"], ["0", "パス", ""], ["1", "香", "金"]]
    before = copy.deepcopy(first)
    assert board_attack_numbers(first)["B"][0] == 2
    assert first == before
    assert board_attack_numbers([]) == {s: [None] * 4 for s in "ABCD"}
    assert board_attack_numbers([["1", "馬", "銀"]])["B"][0] == 1


def test_concealed_piece_changes_do_not_change_numbers():
    moves = [["0", "し", "金"], ["1", "パス", ""], ["2", "金", "香"]]
    changed = copy.deepcopy(moves)
    changed[0][1] = "王"
    assert board_attack_numbers(changed) == board_attack_numbers(moves)


def test_invalid_rows_do_not_create_badges_or_exceed_board_slots():
    moves = [None, [], {}, ["4", "し", "金"], ["0", "パス", "金"], ["0", "し", ""]]
    assert board_attack_numbers(moves) == {s: [None] * 4 for s in "ABCD"}
    rows = [[str(seat), "し", "銀"] for _ in range(4) for seat in range(4)]
    numbers = board_attack_numbers(rows + [["0", "し", "金"]])
    assert numbers["A"] == [1, 5, 9, 13]
    assert numbers["D"] == [4, 8, 12, 16]


def test_public_snapshot_keeps_numbers_after_log_truncation_and_hides_empty_board():
    from backend import app as app_module
    from goita_ai2.state import GoitaState

    hands = {"A": list("73124569"), "B": list("11112345"),
             "C": list("31112456"), "D": list("17123458")}
    state = GoitaState(hands, dealer="C")
    game = {"state": state, "init_hands": hands, "dealer": "C", "is_started": True,
            "board": app_module._new_board_snapshot(), "human_seats": {"A": "number-host"},
            "kifu_moves": [], "log": ["Game start. dealer=C"]}
    for seat, action in [("C", ("attack_after_block", "3", "1")),
                         ("D", ("receive", "1", None)), ("D", ("attack", None, "7")),
                         ("A", ("receive", "7", None)), ("A", ("attack", None, "3"))]:
        app_module._apply_action(state, seat, action)
        app_module._update_board_snapshot(game["board"], seat, action, hidden_receive=action[0] == "attack_after_block")
        game["kifu_moves"].append(app_module._action_to_kifu_row(seat, action))
        game["log"].append(app_module._format_action(seat, action))
    game["log"].extend(["B: pass"] * 220)
    payloads = [app_module._state_public_view(state, game_id="number-test-room", game_obj=game,
                                           viewer=viewer, client_id=client)
                for viewer, client in [("A", "number-host"), ("W", "number-spectator")]]
    for payload in payloads:
        assert len(payload["log"]) == 200
        assert not any("attack" in line for line in payload["log"])
        assert payload["board_attack_numbers"] == {"A": [3, None, None, None], "B": [None] * 4,
                                                   "C": [1, None, None, None], "D": [2, None, None, None]}
    game["is_started"] = False
    payload = app_module._state_public_view(state, game_id="number-test-room", game_obj=game, viewer="W")
    assert payload["board_attack_numbers"] == {s: [None] * 4 for s in "ABCD"}
