import random

from goita_ai2.instruction_case_audit import (
    choose_decision,
    compare_decision,
    load_cases,
    reconstruct_case,
)
from goita_ai2.current_ai.agent import RuleBasedAgent


def _current_ai_decision(case_id: str):
    case = next(case for case in load_cases() if case["id"] == case_id)
    case_number = int(case_id.rsplit("-", 1)[-1])
    random.seed(10_000 + case_number)
    agent = RuleBasedAgent(name=f"{case_id.lower()}-regression")
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    state = reconstruct_case(case, agent)
    first, second, reasons = choose_decision(state, case["actor"], agent)
    return first, second, reasons


def test_all_exact_kifu_cases_reconstruct_to_the_actor_turn():
    cases = load_cases()
    assert len(cases) == 13

    for case in cases:
        state = reconstruct_case(case)
        assert state.turn == case["actor"], case["id"]
        assert state.legal_actions(case["actor"]), case["id"]


def test_receive_and_followup_attack_are_compared_as_one_decision():
    case = {
        "expected": {
            "required_type": "receive_and_attack",
            "block": "銀",
            "attack": "し",
        }
    }
    status, checks = compare_decision(
        case,
        ("receive", "4", None),
        ("attack", None, "1"),
    )

    assert status == "pass"
    assert len(checks) == 3


def test_forbidden_followup_attack_fails_the_case():
    case = {
        "expected": {
            "required_block": "金",
            "forbidden_attack": "香",
        }
    }
    status, _ = compare_decision(
        case,
        ("receive", "5", None),
        ("attack", None, "2"),
    )

    assert status == "fail"


def test_allowed_hidden_blocks_accept_either_piece_but_not_protected_piece():
    case = {
        "expected": {
            "allowed_blocks": ["し", "香"],
            "attack": "金",
            "protected": ["銀"],
        }
    }

    shi_status, _ = compare_decision(
        case,
        ("attack_after_block", "1", "5"),
        None,
    )
    kyosha_status, _ = compare_decision(
        case,
        ("attack_after_block", "2", "5"),
        None,
    )
    silver_status, _ = compare_decision(
        case,
        ("attack_after_block", "4", "5"),
        None,
    )

    assert shi_status == "pass"
    assert kyosha_status == "pass"
    assert silver_status == "fail"


def test_kifu004_replay_receives_ally_gold_and_attacks_fourth_silver():
    case = next(case for case in load_cases() if case["id"] == "KIFU-004")
    agent = RuleBasedAgent(name="kifu-004-regression")
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    state = reconstruct_case(case, agent)

    tracker = agent._track[id(state)]
    assert tracker["my_attack_count"] == 2

    receive, attack, _reasons = choose_decision(state, "A", agent)

    assert receive == ("receive", "5", None)
    assert attack == ("attack", None, "4")


def test_kifu001_zero_shi_ally_receives_silver_and_stops_shi_attack():
    receive, attack, _reasons = _current_ai_decision("KIFU-001")

    assert receive == ("receive", "4", None)
    assert attack is not None
    assert attack[0] == "attack"
    assert attack[2] != "1"


def test_kifu003_receives_silver_for_own_guaranteed_finish():
    receive, attack, _reasons = _current_ai_decision("KIFU-003")

    assert receive == ("receive", "4", None)
    assert attack is not None
    assert attack[0] == "attack"
    assert attack[2] in ("5", "6")
