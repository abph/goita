"""強化中AIの本体となるRuleBasedAgentを定義します。
判断に使う各種設定値を保持し、攻め・受け・推定などのモジュールを一つに組み立てます。
このファイル自身は細かな戦略を持たず、AI全体の構成と共通状態を管理します。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from goita_ai2.current_ai.attack_strategy import AttackStrategyMixin
from goita_ai2.current_ai.decision import DecisionMixin
from goita_ai2.current_ai.endgame import EndgameMixin
from goita_ai2.current_ai.forced_plans import ForcedPlansMixin
from goita_ai2.current_ai.hand_evaluation import HandEvaluationMixin
from goita_ai2.current_ai.inference import PublicInferenceMixin
from goita_ai2.current_ai.receive_strategy import ReceiveStrategyMixin
from goita_ai2.current_ai.tracking import TrackingMixin

class RuleBasedAgent(
    DecisionMixin,
    TrackingMixin,
    HandEvaluationMixin,
    ForcedPlansMixin,
    EndgameMixin,
    AttackStrategyMixin,
    ReceiveStrategyMixin,
    PublicInferenceMixin,
):
    def __init__(self, name: str = "RuleBased"):
        self.name = name
        self.me: Optional[str] = None

        self._track: Dict[int, dict] = {}
        self._my_initial_hands_by_state_id: Dict[int, List[str]] = {}
        self._relative_hand_rank_table: Optional[Dict[str, Dict[str, str]]] = None

        self.WIN_NOW_BONUS = 10_000.0
        self.WIN_AFTER_RECEIVE_BONUS = 9_000.0

        self.KING_ATTACK_PENALTY = 300.0

        self.FIRST_ENEMY_RECEIVE_BONUS = 500.0
        self.FIRST_ENEMY_PASS_BONUS = 500.0
        self.FIRST_ENEMY_KING_RECEIVE_PENALTY = 12000.0

        self.LAST_ONE_BONUS = 65.0

        self.KING_GYOKU_FORCE_ORDER = True
        self.FORCE_KING_GYOKU_ON_THIRD_ATTACK = True

        self.PREFER_PUBLIC_SAFE_NONKING_ON_THIRD_ATTACK = True

        self.PUBLIC_SAFE_ATTACK_BONUS_HIGH = 60.0
        self.PUBLIC_SAFE_ATTACK_BONUS_MID = 30.0
        self.PUBLIC_SAFE_ATTACK_BONUS_LOW = 10.0

        self.KAKARI_GOTAE_BONUS = 100.0
        self.ABSOLUTE_SAFE_BONUS = 1000.0
        self.TATEWARI_BONUS = 800.0
        self.CONTINUOUS_ATTACK_BONUS = 500.0
        self.ATTACK_STRATEGY_BONUS = 120.0
        self.RECEIVE_KEEP_PENALTY = 25.0
        self.ENEMY_FIRST_ATTACK_POLICY = "hand_strength"
        self.USE_RELATIVE_HAND_RANK = True
        self.USE_PUBLIC_HAND_INFERENCE = True
        self.USE_WEAK_SHI_ATTACK_STRATEGY = True
        self.USE_ENEMY_SHI_RESPONSE = False
        self.WEAK_SHI_ATTACK_BONUS = 120.0
        self.SHI_ATTACK_MODE_BONUS = 520.0
        self.DEALER_FOUR_SHI_BLOCK_SHI_BONUS = 220.0
        self.NON_WEAK_SHI_ATTACK_PENALTY = 220.0
        self.WEAK_SHI_FALLBACK_HIGH_POINT_WEIGHT = 2.0
        self.SHI_ATTACK_PREPARE_PASS_BONUS = 180.0
        self.ENEMY_SHI_PASS_BONUS = 250.0
        self.ENEMY_SHI_RECEIVE_PENALTY = 180.0
        self.DEALER_OPENING_PLAN_ATTACK_BONUS = 220.0
        self.DEALER_OPENING_PLAN_BLOCK_PENALTY = 700.0
        self.DEALER_SURPLUS_FOUR_MIDDLE_BLOCK_BONUS = 260.0
        self.OPPONENT_FIRST_ATTACK_STRATEGY_SAFE_PENALTY = {
            "1": 40.0,
            "2": 80.0,
            "3": 55.0,
            "4": 55.0,
            "5": 55.0,
            "6": 70.0,
            "7": 70.0,
        }
        self.INFER_REPEAT_RECEIVE_PASS_BONUS = 14.0
        self.INFER_REPEAT_RECEIVE_PENALTY = 10.0
        self.INFER_ATTACK_EXHAUSTED_BONUS = 35.0
        self.INFER_ATTACK_OVERLAP_PENALTY = 8.0
        self.INFER_KAKARI_BLOCKED_PENALTY = 22.0
        self.INFER_KAKARI_CLEAR_BONUS = 25.0
        self.INFER_BLOCK_KEEP_BONUS = 14.0
        self.INFER_ALLY_STRATEGY_KEEP_BONUS = 18.0
        self.INFER_SHI_ATTACK_ALLY_BONUS = 25.0
        self.INFER_SHI_ATTACK_ENEMY_PENALTY = 14.0
        self.INFER_FORCE_KING_PRESSURE_BONUS = 18.0
        self.KAKARI_SATURATION_RECEIVE_BONUS = 280.0
        self.KAKARI_SATURATION_ATTACK_BONUS = 150.0
        self.KAKARI_SATURATION_ALLY_REMAINING_BONUS = 45.0
        self.ALLY_FORCE_KING_RECEIVE_BONUS = 720.0
        self.ALLY_FORCE_KING_ATTACK_BONUS = 950.0
        self.ALLY_STRONG_FOLLOWUP_RECEIVE_BONUS = 620.0
        self.ENDGAME_PAIR_SCORE_WEIGHT = 1.6
        self.ENDGAME_PAIR_KING_RECEIVE_BONUS = 18.0
        self.ENDGAME_PAIR_UNCERTAIN_PENALTY = 16.0
        self.ENDGAME_MIXED_SHI_PAIR_BONUS = 180.0
        self.ENDGAME_SHI_PAIR_PENALTY = 180.0
        self.CONDITIONAL_SHI_ROYAL_ROUTE_BASE_BONUS = 280.0
        self.CONDITIONAL_SHI_ROYAL_ROUTE_SCORE_WEIGHT = 8.0
        self.REACH_AVOIDANCE_CONDITIONAL_TSUME_MIN_RISK_GAP = 0.05
        self.SHI_SASHIKOMI_WAIT_BONUS = 180.0
        self.SHI_SASHIKOMI_ATTACK_BONUS = 520.0
        self.SHI_EXHAUST_RECEIVE_BONUS = 760.0
        self.SHI_EXHAUST_ATTACK_BONUS = 620.0
        self.WEAK_SHI_ENDGAME_MIXED_BLOCK_BONUS = 180.0
        self.PRESERVE_WIN_ATTACK_PASS_BONUS = 26000.0
        self.PRESERVE_WIN_ATTACK_RECEIVE_PENALTY = 12000.0
        self.FUSE_KYOSHA_BLOCK_PENALTY = 90.0
        self.FUSE_KING_BLOCK_PENALTY = 80.0
        self.FUSE_KEEP_LAST_SHI_PENALTY = 35.0
        self.FUSE_ENEMY_SHI_THREAT_BLOCK_PENALTY = 110.0
        self.FUSE_THIRD_BLOCK_KING_SHI_BONUS = 60.0
        self.FUSE_ATTACK_SATURATION_BLOCK_BONUS = 30.0
        self.FUSE_KEEP_KIN_GIN_RECEIVE_BONUS = 10.0
        self.SECOND_KYOSHA_KEEP_SINGLE_SHI_BLOCK_PENALTY = 260.0
        self.SECOND_KYOSHA_LOW_MIDDLE_BLOCK_BONUS = 120.0
        self.LOWER_ATTACK_SHAPE_BLOCK_BONUS = 55.0
        self.LOWER_ATTACK_SHAPE_ATTACK_PENALTY = 70.0
        self.TOP_ATTACK_SHAPE_BLOCK_PENALTY = 35.0
        self.SAME_PIECE_PAIR_SPEND_PENALTY = 75.0
        self.SINGLE_MIDDLE_AFTER_BIG_RECEIVE_FIRST_ATTACK_PENALTY = 220.0
        self.FOURTH_MIDDLE_EARLY_ATTACK_DELAY_PENALTY = 2400.0
        self.FOURTH_MIDDLE_THIRD_ATTACK_BONUS = 1400.0
        self.SINGLE_MIDDLE_OVER_FOUR_SHI_SIGNAL_PENALTY = 260.0
        self.FOUR_SHI_AFTER_BIG_RECEIVE_FIRST_ATTACK_BONUS = 420.0
        self.ALLY_GUARANTEED_WIN_GIVE_WAY_MAX_SCORE = 30.0
        self.ROYAL_WAIT_SHI_BASE_BONUS = 20.0
        self.ROYAL_WAIT_SHI_PRESSURE_WEIGHT = 70.0
        self.ROYAL_WAIT_SHI_PRESSURE_CAP = 3.0
        self.last_decision_reason = ""
        self.last_score_fallback_detail = ""

    def bind_player(self, player: str) -> None:
        if self.me is None:
            self.me = player
        elif self.me != player:
            raise ValueError(f"{self.name}: already bound to {self.me}, cannot bind to {player}")

    def _same_team(self, p1: str, p2: str) -> bool:
        return (
            (p1 in ("A", "C") and p2 in ("A", "C")) or
            (p1 in ("B", "D") and p2 in ("B", "D"))
        )

    def _ally_of(self, me: str) -> str:
        return "C" if me == "A" else "A" if me == "C" else "D" if me == "B" else "B"
