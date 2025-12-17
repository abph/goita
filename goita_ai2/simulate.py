# goita_ai2/simulate.py
from __future__ import annotations

from typing import Dict, Optional, Tuple, List, Any, Union
import random

from goita_ai2.state import GoitaState
from goita_ai2.utils import create_random_hands


Action = Tuple[str, Optional[str], Optional[str]]  # (action_type, block, attack)


def _maybe_bind(agent: Any, seat: str) -> None:
    if agent is None:
        return
    if hasattr(agent, "bind_player"):
        agent.bind_player(seat)


def _notify_public(agents: Dict[str, Any], state: GoitaState, player: str, action: Action) -> None:
    # 公開情報なので「全AI」に通知（自席でなくても追跡できるように）
    for ag in agents.values():
        if ag is None:
            continue
        if hasattr(ag, "on_public_action"):
            ag.on_public_action(state, player, action)


def simulate_random_game(
    hands: Optional[Dict[str, List[str]]] = None,
    dealer: str = "A",
    agents: Optional[Dict[str, Any]] = None,   # 例 {"A": RuleBasedAgent(), "C": RuleBasedAgent()}
    agent: Optional[Any] = None,               # 互換：AだけAIにしたい場合
    verbose: bool = True,
    seed: Optional[int] = None,
    max_steps: int = 500,
) -> GoitaState:
    """
    - agents を渡すと、指定席だけAI、それ以外はランダム
    - agent を渡すと、A席だけAI（互換用）
    """
    if seed is not None:
        random.seed(seed)

    if hands is None:
        hands = create_random_hands()

    state = GoitaState(hands=hands, dealer=dealer)

    # agents 辞書を正規化
    seat_agents: Dict[str, Any] = {}
    if agents is not None:
        seat_agents = dict(agents)
    elif agent is not None:
        seat_agents = {"A": agent}
    else:
        seat_agents = {}

    # ★ここで席を固定（最初から公開情報を追跡できるように）
    for seat, ag in seat_agents.items():
        _maybe_bind(ag, seat)

    if verbose:
        print("=== 初期手札 ===")
        for p in ["A", "B", "C", "D"]:
            print(f"{p}: {state.hands[p]}")
        print(f"親: {dealer}")
        print("==============")

    step = 0
    while (not state.finished) and step < max_steps:
        step += 1
        player = state.turn

        actions = state.legal_actions(player)
        if not actions:
            if verbose:
                print("合法手なしで終了")
            break

        # 行動選択：席にAIがいればAI、いなければランダム
        ag = seat_agents.get(player)
        if ag is not None and hasattr(ag, "select_action"):
            action_type, block, attack = ag.select_action(state, player, actions)
        else:
            action_type, block, attack = random.choice(actions)

        if verbose:
            print(f"\n--- Step {step} ---")
            print(f"turn: {player}, phase: {state.phase}, attacker: {state.attacker}, current_attack: {state.current_attack}")
            print(f"action: {action_type}, block={block}, attack={attack}")

        # 適用
        if action_type == "pass":
            state.apply_pass(player)
        elif action_type == "receive":
            if block is None:
                raise ValueError("receive には block が必要です")
            state.apply_receive(player, block)
        elif action_type == "attack":
            if attack is None:
                raise ValueError("attack には attack が必要です")
            state.apply_attack(player, attack)
        elif action_type == "attack_after_block":
            if block is None or attack is None:
                raise ValueError("attack_after_block には block と attack の両方が必要です")
            state.apply_attack_after_block(player, block, attack)
        else:
            raise ValueError(f"未知の action_type: {action_type}")

        # ★公開情報を全AIへ通知（適用後でOK：この行動が確定したため）
        _notify_public(seat_agents, state, player, (action_type, block, attack))

        if verbose:
            for p in ["A", "B", "C", "D"]:
                print(f"{p} hand: {state.hands[p]}")

    if verbose:
        print("\n=== 結果 ===")
        print(f"finished: {state.finished}, winner: {state.winner}")
        print(f"team_score: {state.team_score}")

    return state
