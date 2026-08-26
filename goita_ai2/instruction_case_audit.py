"""棋譜つき指導ケースを再現し、現行AIの判断との差を監査します。

指導データはAI本体から独立したまま読み込み、棋譜の途中局面を合法手で復元します。
受けを選んだ局面では直後の攻めも取得し、人間の推奨手と一組で比較します。
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from goita_ai2.constants import ALL_SEATS, PIECE_KANJI
from goita_ai2.current_ai.agent import RuleBasedAgent
from goita_ai2.state import GoitaState


Action = Tuple[str, Optional[str], Optional[str]]
CASE_PATH = Path(__file__).with_name("instruction_cases") / "regression_candidates.yaml"
KANJI_PIECE = {kanji: piece for piece, kanji in PIECE_KANJI.items()}


@dataclass
class AuditResult:
    case_id: str
    title: str
    actor: str
    status: str
    expected: str
    actual: str
    checks: List[str]
    decision_reasons: List[str]
    elapsed_seconds: float


def load_cases(path: Path = CASE_PATH) -> List[Dict[str, Any]]:
    """Load and minimally validate the exact-kifu regression cases."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("cases"), list):
        raise ValueError(f"cases が見つかりません: {path}")
    return [dict(case) for case in document["cases"]]


def _piece(value: Any) -> str:
    text = str(value)
    if text in PIECE_KANJI:
        return text
    if text in KANJI_PIECE:
        return KANJI_PIECE[text]
    raise ValueError(f"未知の駒です: {value}")


def _hands(case: Mapping[str, Any]) -> Dict[str, List[str]]:
    source = case.get("hands")
    if not isinstance(source, dict):
        raise ValueError(f"{case.get('id')}: hands がありません")
    result = {}
    for seat in ALL_SEATS:
        raw = source.get(seat)
        if not isinstance(raw, str) or len(raw) != 8:
            raise ValueError(f"{case.get('id')}: {seat}の手駒が8枚ではありません")
        result[seat] = [_piece(item) for item in raw]
    return result


def apply_action(state: GoitaState, player: str, action: Action) -> None:
    action_type, block, attack = action
    if action_type == "pass":
        state.apply_pass(player)
    elif action_type == "receive" and block is not None:
        state.apply_receive(player, block)
    elif action_type == "attack" and attack is not None:
        state.apply_attack(player, attack)
    elif action_type == "attack_after_block" and block is not None and attack is not None:
        state.apply_attack_after_block(player, block, attack)
    else:
        raise ValueError(f"不正な行動です: {action}")


def _notify(agent: Optional[RuleBasedAgent], state: GoitaState, player: str, action: Action) -> None:
    if agent is not None:
        agent.on_public_action(state, player, action)


def _apply_public(
    state: GoitaState,
    agent: Optional[RuleBasedAgent],
    player: str,
    action: Action,
) -> None:
    legal = state.legal_actions(player)
    if action not in legal:
        raise ValueError(
            f"{player}の行動{action}は合法ではありません。"
            f" turn={state.turn}, phase={state.phase}, attack={state.current_attack}, legal={legal}"
        )
    apply_action(state, player, action)
    _notify(agent, state, player, action)


def _pass_until(
    state: GoitaState,
    agent: Optional[RuleBasedAgent],
    target: str,
) -> None:
    for _ in range(4):
        if state.turn == target:
            return
        if state.phase != "receive":
            break
        player = state.turn
        _apply_public(state, agent, player, ("pass", None, None))
    if state.turn != target:
        raise ValueError(
            f"{target}までパスで進めません。turn={state.turn}, phase={state.phase}"
        )


def reconstruct_case(
    case: Mapping[str, Any],
    agent: Optional[RuleBasedAgent] = None,
) -> GoitaState:
    """Reconstruct the public position immediately before the recorded decision."""
    actor = str(case["actor"])
    state = GoitaState(hands=_hands(case), dealer=str(case["dealer"]))
    if agent is not None:
        agent.bind_player(actor)
        agent._ensure_trackers(state)

    for row_number, row in enumerate(case.get("prefix", []), start=1):
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"{case.get('id')}: prefix {row_number} の形式が不正です")
        player, raw_block, raw_attack = row
        player = str(player)
        block = _piece(raw_block)
        attack = _piece(raw_attack)
        _pass_until(state, agent, player)
        try:
            if state.phase == "receive":
                _apply_public(state, agent, player, ("receive", block, None))
                _apply_public(state, agent, player, ("attack", None, attack))
            else:
                _apply_public(
                    state,
                    agent,
                    player,
                    ("attack_after_block", block, attack),
                )
        except ValueError as exc:
            raise ValueError(
                f"{case.get('id')}: prefix {row_number} {row} を再現できません: {exc}"
            ) from exc

    _pass_until(state, agent, actor)
    expected_attack = case.get("current_attack")
    if expected_attack is not None and state.current_attack != _piece(expected_attack):
        raise ValueError(
            f"{case.get('id')}: 場の攻めが一致しません。"
            f" expected={expected_attack}, actual={PIECE_KANJI.get(state.current_attack or '', 'なし')}"
        )
    if state.turn != actor:
        raise ValueError(f"{case.get('id')}: {actor}の手番ではありません")
    return state


def _reason(agent: RuleBasedAgent) -> str:
    reason = str(getattr(agent, "last_decision_reason", "") or "")
    detail = str(getattr(agent, "last_score_fallback_detail", "") or "")
    if detail and detail != reason:
        return f"{reason}/{detail}" if reason else detail
    return reason or "理由記録なし"


def choose_decision(
    state: GoitaState,
    actor: str,
    agent: RuleBasedAgent,
) -> Tuple[Action, Optional[Action], List[str]]:
    """Ask for the current move and, after a receive, the follow-up attack."""
    first = agent.select_action(state, actor, state.legal_actions(actor))
    reasons = [_reason(agent)]
    second = None
    if first[0] == "receive":
        _apply_public(state, agent, actor, first)
        second = agent.select_action(state, actor, state.legal_actions(actor))
        reasons.append(_reason(agent))
    return first, second, reasons


def _conceptual_type(first: Action, second: Optional[Action]) -> str:
    if first[0] == "receive" and second is not None and second[0] == "attack":
        return "receive_and_attack"
    return first[0]


def _selected_block(first: Action) -> Optional[str]:
    return first[1]


def _selected_attack(first: Action, second: Optional[Action]) -> Optional[str]:
    if first[0] in ("attack", "attack_after_block"):
        return first[2]
    if second is not None and second[0] == "attack":
        return second[2]
    return None


def _exact_piece(value: Any) -> Optional[str]:
    try:
        return _piece(value)
    except (TypeError, ValueError):
        return None


def compare_decision(
    case: Mapping[str, Any],
    first: Action,
    second: Optional[Action],
) -> Tuple[str, List[str]]:
    expected = case.get("expected", {})
    if not isinstance(expected, dict):
        return "error", ["expectedの形式が不正"]

    checks: List[str] = []
    failures: List[str] = []
    advisories: List[str] = []
    actual_type = _conceptual_type(first, second)
    actual_block = _selected_block(first)
    actual_attack = _selected_attack(first, second)

    type_is_preferred = "required_type" not in expected and "preferred_type" in expected
    wanted_type = expected.get("required_type", expected.get("preferred_type"))
    if wanted_type is not None:
        # `receive` means the immediate pass/receive choice. Its mandatory
        # follow-up attack is reported separately and must not change the type
        # comparison into a false failure.
        ok = (
            first[0] == "receive"
            if str(wanted_type) == "receive"
            else actual_type == str(wanted_type)
        )
        checks.append(f"行動種別={actual_type}（期待:{wanted_type}）")
        if not ok:
            (advisories if type_is_preferred else failures).append("行動種別")

    block_is_preferred = "required_block" not in expected and "preferred_block" in expected
    wanted_block = expected.get("required_block", expected.get("preferred_block", expected.get("block")))
    exact_block = _exact_piece(wanted_block)
    if exact_block is not None:
        checks.append(
            f"受け・伏せ={PIECE_KANJI.get(actual_block or '', 'なし')}"
            f"（期待:{PIECE_KANJI[exact_block]}）"
        )
        if actual_block != exact_block:
            (advisories if block_is_preferred else failures).append("受け・伏せ駒")

    allowed_blocks = expected.get("allowed_blocks")
    if isinstance(allowed_blocks, list) and allowed_blocks:
        allowed_block_pieces = {_piece(item) for item in allowed_blocks}
        labels = "・".join(PIECE_KANJI[item] for item in allowed_block_pieces)
        checks.append(
            f"受け・伏せ={PIECE_KANJI.get(actual_block or '', 'なし')}"
            f"（許容:{labels}）"
        )
        if actual_block not in allowed_block_pieces:
            failures.append("受け・伏せ駒")

    wanted_attack = expected.get("required_attack", expected.get("attack"))
    exact_attack = _exact_piece(wanted_attack)
    preferred_attacks = expected.get("preferred_attacks")
    if exact_attack is not None:
        checks.append(
            f"攻め={PIECE_KANJI.get(actual_attack or '', 'なし')}"
            f"（期待:{PIECE_KANJI[exact_attack]}）"
        )
        if actual_attack != exact_attack:
            failures.append("攻め駒")
    elif isinstance(preferred_attacks, list):
        allowed = {_piece(item) for item in preferred_attacks}
        labels = "・".join(PIECE_KANJI[item] for item in allowed)
        checks.append(
            f"攻め={PIECE_KANJI.get(actual_attack or '', 'なし')}（期待:{labels}）"
        )
        if actual_attack not in allowed:
            failures.append("攻め駒")

    forbidden_attack = _exact_piece(expected.get("forbidden_attack"))
    if forbidden_attack is not None:
        checks.append(f"禁止攻め={PIECE_KANJI[forbidden_attack]}を回避")
        if actual_attack == forbidden_attack:
            failures.append("禁止攻め駒")

    if expected.get("followup") == "し以外を攻める":
        checks.append("受けた後はし以外で攻める")
        if actual_attack in (None, "1"):
            failures.append("受け後の攻め駒")

    forbidden = case.get("forbidden", [])
    if isinstance(forbidden, list) and "pass" in forbidden and first[0] == "pass":
        checks.append("禁止行動=パス")
        failures.append("禁止行動")

    protected = expected.get("protected", [])
    if isinstance(protected, list) and protected:
        protected_pieces = {_piece(item) for item in protected}
        spent = {item for item in (actual_block, actual_attack) if item is not None}
        labels = "・".join(PIECE_KANJI[item] for item in protected_pieces)
        checks.append(f"温存={labels}")
        if protected_pieces & spent:
            failures.append("温存対象")

    if failures:
        return "fail", checks
    if advisories:
        return "review", checks
    return "pass", checks


def action_label(first: Action, second: Optional[Action] = None) -> str:
    if first[0] == "pass":
        return "パス"
    if first[0] == "receive":
        block = PIECE_KANJI.get(first[1] or "", "?")
        if second is not None and second[0] == "attack":
            attack = PIECE_KANJI.get(second[2] or "", "?")
            return f"{block}で受けて{attack}で攻める"
        return f"{block}で受ける"
    if first[0] == "attack":
        return f"{PIECE_KANJI.get(first[2] or '', '?')}で攻める"
    if first[0] == "attack_after_block":
        block = PIECE_KANJI.get(first[1] or "", "?")
        attack = PIECE_KANJI.get(first[2] or "", "?")
        return f"{block}を伏せて{attack}で攻める"
    return str(first)


def expected_label(case: Mapping[str, Any]) -> str:
    expected = case.get("expected", {})
    wanted_type = expected.get("required_type", expected.get("preferred_type", ""))
    block = _exact_piece(
        expected.get("required_block", expected.get("preferred_block", expected.get("block")))
    )
    attack = _exact_piece(expected.get("required_attack", expected.get("attack")))
    if wanted_type == "pass":
        return "パス"
    if wanted_type == "receive" and block:
        return f"{PIECE_KANJI[block]}で受ける"
    if wanted_type == "receive_and_attack" and block:
        if attack:
            return f"{PIECE_KANJI[block]}で受けて{PIECE_KANJI[attack]}で攻める"
        return f"{PIECE_KANJI[block]}で受けて攻める"
    if wanted_type == "attack_after_block" and block and attack:
        return f"{PIECE_KANJI[block]}を伏せて{PIECE_KANJI[attack]}で攻める"
    parts = []
    allowed_blocks = expected.get("allowed_blocks")
    if isinstance(allowed_blocks, list) and allowed_blocks:
        parts.append("受け・伏せ:" + "・".join(map(str, allowed_blocks)))
    if block:
        parts.append(f"受け・伏せ:{PIECE_KANJI[block]}")
    if attack:
        parts.append(f"攻め:{PIECE_KANJI[attack]}")
    if expected.get("preferred_attacks"):
        parts.append("攻め:" + "・".join(map(str, expected["preferred_attacks"])))
    if expected.get("forbidden_attack"):
        parts.append(f"{expected['forbidden_attack']}攻めを避ける")
    return "、".join(parts) or str(expected)


def audit_case(case: Mapping[str, Any]) -> AuditResult:
    import time

    started = time.perf_counter()
    case_number = int(str(case["id"]).rsplit("-", 1)[-1])
    random.seed(10_000 + case_number)
    agent = RuleBasedAgent(name=f"instruction-audit-{case['id']}")
    # Batch replay must not leave speculative background workers running.
    # Foreground rules and the same time-limited search remain enabled.
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    state = reconstruct_case(case, agent)
    first, second, reasons = choose_decision(state, str(case["actor"]), agent)
    status, checks = compare_decision(case, first, second)
    actual = action_label(first, second)

    next_plan = case.get("expected", {}).get("next_plan")
    if isinstance(next_plan, dict):
        committed = second if first[0] == "receive" else first
        if committed is None:
            status = "fail"
            checks.append("次の攻め計画へ進める行動がない")
        else:
            _apply_public(state, agent, str(case["actor"]), committed)
            _pass_until(state, agent, str(case["actor"]))
            next_action = agent.select_action(
                state,
                str(case["actor"]),
                state.legal_actions(str(case["actor"])),
            )
            reasons.append(_reason(agent))
            actual += f" -> 次に{action_label(next_action)}"
            wanted_next_block = _piece(next_plan["block"])
            wanted_next_attack = _piece(next_plan["attack"])
            checks.append(
                "次の攻め="
                f"{PIECE_KANJI.get(next_action[1] or '', 'なし')}を伏せて"
                f"{PIECE_KANJI.get(next_action[2] or '', 'なし')}"
                f"（期待:{PIECE_KANJI[wanted_next_block]}を伏せて"
                f"{PIECE_KANJI[wanted_next_attack]}）"
            )
            if not (
                next_action[0] == "attack_after_block"
                and next_action[1] == wanted_next_block
                and next_action[2] == wanted_next_attack
            ):
                status = "fail"
    return AuditResult(
        case_id=str(case["id"]),
        title=str(case["title"]),
        actor=str(case["actor"]),
        status=status,
        expected=expected_label(case),
        actual=actual,
        checks=checks,
        decision_reasons=reasons,
        elapsed_seconds=round(time.perf_counter() - started, 3),
    )


def run_audit(cases: Optional[Iterable[Mapping[str, Any]]] = None) -> List[AuditResult]:
    return [audit_case(case) for case in (cases or load_cases())]


def print_report(results: Sequence[AuditResult]) -> None:
    for result in results:
        mark = {"pass": "OK", "review": "要確認", "fail": "NG"}.get(
            result.status,
            "ERROR",
        )
        print(f"[{mark}] {result.case_id} {result.title}")
        print(f"  現行AI: {result.actual}")
        print(f"  推奨手: {result.expected}")
        print(f"  判断理由: {' -> '.join(result.decision_reasons)}")
        print(f"  思考時間: {result.elapsed_seconds:.3f}秒")
    passed = sum(result.status == "pass" for result in results)
    reviews = sum(result.status == "review" for result in results)
    failed = sum(result.status == "fail" for result in results)
    print(
        f"\n合計: 一致{passed}件、要確認{reviews}件、不一致{failed}件"
        f"（全{len(results)}件）"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="現行ごいたAIと棋譜指導ケースを比較します")
    parser.add_argument("--case", action="append", dest="case_ids", help="KIFU-001のように対象を限定")
    parser.add_argument("--json", action="store_true", help="JSON形式で出力")
    args = parser.parse_args(argv)

    cases = load_cases()
    if args.case_ids:
        selected = set(args.case_ids)
        cases = [case for case in cases if case.get("id") in selected]
        missing = selected - {str(case.get("id")) for case in cases}
        if missing:
            parser.error(f"存在しないケース: {', '.join(sorted(missing))}")

    results = run_audit(cases)
    if args.json:
        print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    else:
        print_report(results)
    return 0 if all(result.status == "pass" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
