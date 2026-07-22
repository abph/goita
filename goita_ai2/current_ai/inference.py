"""公開情報から、他プレイヤーの手駒を推定します。
自分の手駒、場に出た駒、攻め・受け・パスの履歴から、駒ごとの残り枚数を更新します。
推定した攻めタイプや手駒ランクは、受け・パス・攻め返しの判断材料になります。
"""

from __future__ import annotations

from collections import Counter
from math import comb, factorial, log
from typing import Dict, List, Optional, Tuple


class PublicInferenceMixin:
    """Tracks public actions and estimates opponents without hidden information."""

    def _infer_first_attack_strategy(self, attack: Optional[str]) -> Optional[Dict[str, object]]:
        if attack is None:
            return None
        if attack in ("8", "9"):
            return {
                "label": "king_pair_or_king_attack",
                "attack_types": [1],
                "pieces": ["8", "9"],
                "strength": 7.0,
            }
        if attack == "2":
            return {
                "label": "kyosha_repeat",
                "attack_types": [2, 5],
                "pieces": ["2"],
                "strength": 5.0,
            }
        if attack in ("5", "4", "3"):
            return {
                "label": "middle_repeat",
                "attack_types": [3, 6],
                "pieces": [attack],
                "strength": 3.5,
            }
        if attack in ("7", "6"):
            return {
                "label": "big_pair",
                "attack_types": [4],
                "pieces": [attack],
                "strength": 4.5,
            }
        if attack == "1":
            return {
                "label": "shi_attack",
                "attack_types": [7, 8],
                "pieces": ["1"],
                "strength": 1.5,
            }
        return None

    def _first_attack_count_range(self, attack: Optional[str]) -> Optional[Tuple[int, int]]:
        if attack in ("2", "3", "4", "5"):
            return (2, 3)
        if attack in ("6", "7"):
            return (1, 2)
        if attack == "1":
            return (3, 4)
        if attack in ("8", "9"):
            return (1, 1)
        return None

    def _track_partner_first_strategy_reaction(
        self,
        tr: dict,
        player: str,
        action_type: str,
        block: Optional[str],
        attack: Optional[str],
    ) -> None:
        pending = tr.setdefault("pending_partner_first_strategy_reactions", {})

        if action_type == "receive" and block is not None:
            context = tr.get("active_attack_context") or {}
            original_attacker = context.get("attacker")
            original_piece = context.get("piece")
            if (
                original_attacker is not None
                and original_attacker != player
                and original_piece in ("1", "2", "3", "4", "5", "6", "7")
                and self._same_team(str(original_attacker), player)
            ):
                original_model = tr.get("public_hand_models", {}).get(str(original_attacker), {})
                if (
                    original_model.get("first_attack") == original_piece
                    and not bool(original_model.get("strategy_broken"))
                ):
                    pending[player] = {
                        "original_attacker": str(original_attacker),
                        "piece": str(original_piece),
                    }
            return

        if action_type != "attack" or attack is None:
            return

        response = pending.pop(player, None)
        if response is None:
            return

        original_attacker = str(response["original_attacker"])
        original_piece = str(response["piece"])
        accepted = attack == original_piece
        if accepted:
            reason = "shi_attack_continued" if original_piece == "1" else "kakarigotae_returned"
        else:
            reason = "shi_attack_rejected" if original_piece == "1" else "kakarigotae_not_returned"

        original_model = tr.get("public_hand_models", {}).get(original_attacker)
        if original_model is None:
            return
        reaction = {
            "partner": player,
            "piece": original_piece,
            "returned_piece": attack,
            "status": "accepted" if accepted else "rejected",
            "reason": reason,
        }
        original_model["partner_first_strategy_reaction"] = reaction
        original_model.setdefault("partner_first_strategy_reaction_history", []).append(reaction)
        if not accepted:
            original_model["partner_first_strategy_rejected"] = True

    def _break_first_attack_strategy_after_partner_rejection(
        self,
        state,
        tr: dict,
        player: str,
        first_attack: str,
        switched_attack: str,
    ) -> bool:
        model = tr.get("public_hand_models", {}).get(player)
        if model is None or bool(model.get("strategy_broken")):
            return False
        reaction = model.get("partner_first_strategy_reaction") or {}
        if (
            not bool(model.get("partner_first_strategy_rejected"))
            or reaction.get("status") != "rejected"
            or reaction.get("piece") != first_attack
        ):
            return False

        model["strategy_broken"] = True
        model["strategy_broken_reason"] = str(reaction.get("reason", "partner_rejected"))
        model["strategy_broken_on_attack"] = switched_attack
        model["inferred_attack_strategy_active"] = False
        model["strength"] -= 2.0

        guess = tr.get("other_first_attack_strategy_by_player", {}).get(player)
        if guess is not None:
            guess["active"] = False
            guess["broken_reason"] = model["strategy_broken_reason"]
            guess["broken_on_attack"] = switched_attack

        est = self._piece_count_estimate(tr, player, first_attack)
        if est is not None and str(est.get("source", "observed")) != "observed":
            observed_first = self._observed_piece_count_for_player(tr, player, first_attack)
            est["min"] = observed_first
            est["max"] = min(self._piece_total(first_attack), observed_first + 1)
            est["source"] = "strategy_broken_after_partner_rejection"
            self._reconcile_piece_count_estimates(state, tr, first_attack)
        return True

    def _ensure_piece_count_estimate(self, tr: dict, player: str, piece: str) -> Dict[str, object]:
        estimates = tr.setdefault("other_piece_count_estimates", {})
        by_piece = estimates.setdefault(player, {})
        return by_piece.setdefault(
            piece,
            {
                "min": 0,
                "max": self._piece_total(piece),
                "source": "observed",
            },
        )

    def _set_piece_count_estimate(
        self,
        tr: dict,
        player: str,
        piece: str,
        *,
        min_count: int,
        max_count: int,
        source: str,
    ) -> None:
        est = self._ensure_piece_count_estimate(tr, player, piece)
        est["min"] = max(int(est.get("min", 0)), int(min_count))
        est["max"] = min(int(est.get("max", self._piece_total(piece))), int(max_count))
        if int(est["max"]) < int(est["min"]):
            est["max"] = est["min"]
        if source != "observed" or str(est.get("source", "observed")) == "observed":
            est["source"] = source

    def _update_observed_piece_count_estimate(self, tr: dict, player: str, piece: str) -> None:
        if player == self.me:
            return
        models = tr.get("public_hand_models", {})
        model = models.get(player)
        if model is None:
            return
        observed_count = int(model.get("attacks", Counter()).get(piece, 0)) + int(
            model.get("blocks", Counter()).get(piece, 0)
        )
        if observed_count <= 0:
            return
        self._set_piece_count_estimate(
            tr,
            player,
            piece,
            min_count=observed_count,
            max_count=self._piece_total(piece),
            source="observed",
        )

    def _record_shi_pass_current_range(self, state, tr: dict, player: str) -> None:
        if player == self.me:
            return
        context = tr.get("active_attack_context") or {}
        if context.get("piece") != "1":
            return

        observed = self._observed_piece_count_for_player(tr, player, "1")
        hidden_slots = int(tr.get("hidden_block_counts", {}).get(player, 0))
        max_original = min(self._piece_total("1"), observed + hidden_slots + 1)
        estimate = self._ensure_piece_count_estimate(tr, player, "1")
        estimate["min"] = max(observed, min(int(estimate.get("min", 0)), max_original))
        estimate["max"] = max(
            int(estimate["min"]),
            min(int(estimate.get("max", self._piece_total("1"))), max_original),
        )
        estimate["source"] = "pass_shi_current_0_1"

        cap = {
            "min": 0,
            "max": 1,
            "observed_at_signal": observed,
            "active": True,
            "source": "pass_shi_current_0_1",
        }
        tr.setdefault("current_piece_count_caps", {}).setdefault(player, {})["1"] = cap
        model = tr.get("public_hand_models", {}).get(player)
        if model is not None:
            model["shi_pass_current_range"] = {"min": 0, "max": 1}
            model.setdefault("shi_pass_current_range_history", []).append(dict(cap))
        self._reconcile_piece_count_estimates(state, tr, "1")

    def _validate_current_piece_count_cap(self, tr: dict, player: str, piece: str) -> None:
        cap = tr.get("current_piece_count_caps", {}).get(player, {}).get(piece)
        if cap is None or not bool(cap.get("active", True)):
            return
        observed = self._observed_piece_count_for_player(tr, player, piece)
        observed_since_signal = observed - int(cap.get("observed_at_signal", observed))
        if observed_since_signal <= int(cap.get("max", 0)):
            return

        cap["active"] = False
        cap["invalidated_by_visible_count"] = observed_since_signal
        estimate = self._ensure_piece_count_estimate(tr, player, piece)
        estimate["min"] = max(observed, int(estimate.get("min", 0)))
        estimate["max"] = self._piece_total(piece)
        estimate["source"] = "observed_after_shi_pass_contradiction"

    def _active_current_piece_count_cap(self, tr: dict, player: str, piece: str) -> Optional[int]:
        cap = tr.get("current_piece_count_caps", {}).get(player, {}).get(piece)
        if cap is None or not bool(cap.get("active", True)):
            return None
        observed = self._observed_piece_count_for_player(tr, player, piece)
        observed_since_signal = max(0, observed - int(cap.get("observed_at_signal", observed)))
        return max(0, int(cap.get("max", 0)) - observed_since_signal)

    def _reconcile_piece_count_estimates(self, state, tr: dict, piece: str) -> None:
        if self.me is None:
            return
        estimates = tr.get("other_piece_count_estimates", {})
        # min/max describe the original ownership of a piece. Use the hand that
        # this observer actually saw at the deal, not the current hand after play.
        mine = int(tr.get("my_init_count", Counter()).get(piece, 0))
        other_players = [p for p in ("A", "B", "C", "D") if p != self.me]

        for p in other_players:
            self._update_observed_piece_count_estimate(tr, p, piece)

        available_for_others = max(0, self._piece_total(piece) - mine)
        while True:
            current_min_sum = sum(
                int(estimates.get(p, {}).get(piece, {}).get("min", 0))
                for p in other_players
            )
            if current_min_sum <= available_for_others:
                break

            relaxed = False
            for p in other_players:
                est = estimates.get(p, {}).get(piece)
                if est is None or str(est.get("source", "observed")) == "observed":
                    continue
                model = tr.get("public_hand_models", {}).get(p, {})
                observed_count = int(model.get("attacks", Counter()).get(piece, 0)) + int(
                    model.get("blocks", Counter()).get(piece, 0)
                )
                if int(est.get("min", 0)) > observed_count:
                    est["min"] = max(observed_count, int(est.get("min", 0)) - 1)
                    relaxed = True
                    break
            if not relaxed:
                break

        lower_bounds = {
            p: int(estimates.get(p, {}).get(piece, {}).get("min", 0))
            for p in other_players
        }
        total = self._piece_total(piece)
        for p in other_players:
            est = estimates.get(p, {}).get(piece)
            if est is None:
                continue
            others_min = sum(v for q, v in lower_bounds.items() if q != p)
            cap = max(lower_bounds[p], total - mine - others_min)
            est["max"] = min(int(est.get("max", total)), cap)
            if int(est["max"]) < int(est["min"]):
                est["max"] = est["min"]

    def _bounded_piece_expectations(
        self,
        pool_count: int,
        players: List[str],
        lower: Dict[str, int],
        upper: Dict[str, int],
        slot_counts: Dict[str, int],
    ) -> Dict[str, float]:
        expected = {p: float(lower[p]) for p in players}
        remaining = max(0.0, float(pool_count) - sum(expected.values()))

        for _ in range(8):
            if remaining <= 1e-9:
                break
            active = [p for p in players if expected[p] + 1e-9 < float(upper[p])]
            if not active:
                break
            weights = {
                p: max(0.25, float(slot_counts[p]) - expected[p])
                for p in active
            }
            total_weight = sum(weights.values())
            additions: Dict[str, float] = {}
            for p in active:
                share = remaining * weights[p] / total_weight
                additions[p] = min(float(upper[p]) - expected[p], share)
            added = sum(additions.values())
            if added <= 1e-9:
                break
            for p, value in additions.items():
                expected[p] += value
            remaining -= added

        return expected

    def _record_piece_pass_evidence(self, state, tr: dict, player: str) -> None:
        if player == self.me:
            return
        context = tr.get("active_attack_context") or {}
        piece = context.get("piece")
        attacker = context.get("attacker")
        if piece is None or attacker is None:
            return

        model = tr.get("public_hand_models", {}).get(player, {})
        relation = "ally" if self._same_team(str(attacker), player) else "enemy"
        rank = str(model.get("estimated_rank", "D"))
        attack_no = max(1, int(context.get("attack_no", 1)))
        if relation == "ally":
            pass_if_held = 0.92
        else:
            strong = rank in ("S", "A", "B", "C")
            if attack_no == 1:
                pass_if_held = 0.12 if strong else 0.55
            elif attack_no == 2:
                pass_if_held = 0.08 if strong else 0.32
            else:
                pass_if_held = 0.05 if strong else 0.18
            if piece == "2":
                pass_if_held = max(pass_if_held, 0.45)

        event = {
            "attacker": str(attacker),
            "relation": relation,
            "attack_no": attack_no,
            "rank_at_signal": rank,
            "pass_if_held": pass_if_held,
            "observed_used": self._observed_piece_count_for_player(tr, player, str(piece)),
            "hand_slots": len(state.hands[player]),
            "hidden_slots": int(tr.get("hidden_block_counts", {}).get(player, 0)),
        }
        by_player = tr.setdefault("piece_pass_evidence", {}).setdefault(player, {})
        by_player.setdefault(str(piece), []).append(event)

    def _pass_evidence_likelihood(
        self,
        tr: dict,
        player: str,
        piece: str,
        original_count: int,
    ) -> float:
        likelihood = 1.0
        events = tr.get("piece_pass_evidence", {}).get(player, {}).get(piece, [])
        for event in events:
            remaining = max(0, original_count - int(event.get("observed_used", 0)))
            hand_slots = int(event.get("hand_slots", 0))
            hidden_slots = int(event.get("hidden_slots", 0))
            combined_slots = hand_slots + hidden_slots
            if remaining > combined_slots:
                return 1e-9
            if remaining == 0 or combined_slots == 0:
                no_current_probability = 1.0
            elif remaining <= hidden_slots:
                no_current_probability = comb(hidden_slots, remaining) / comb(combined_slots, remaining)
            else:
                no_current_probability = 0.0
            pass_if_held = float(event.get("pass_if_held", 0.5))
            likelihood *= no_current_probability + (1.0 - no_current_probability) * pass_if_held
        return max(1e-12, likelihood)

    def _joint_piece_allocations(
        self,
        pool_count: int,
        lower: Tuple[int, int, int],
        upper: Tuple[int, int, int],
    ) -> List[Tuple[int, int, int]]:
        allocations: List[Tuple[int, int, int]] = []
        for first in range(lower[0], min(upper[0], pool_count) + 1):
            for second in range(lower[1], min(upper[1], pool_count - first) + 1):
                third = pool_count - first - second
                if lower[2] <= third <= upper[2]:
                    allocations.append((first, second, third))
        return allocations

    def _block_piece_weight(self, tr: dict, player: str, piece: str) -> float:
        model = tr.get("public_hand_models", {}).get(player, {})
        first_attack = model.get("first_attack")
        strategy_active = bool(model.get("inferred_attack_strategy_active")) and not bool(
            model.get("strategy_broken")
        )
        if piece == "1":
            return 3.5
        if piece == "2":
            return 0.45
        if piece in ("8", "9"):
            return 0.12
        if strategy_active and piece == first_attack:
            return 0.2
        if piece in ("6", "7"):
            return 0.7
        return 1.0

    def _most_likely_hidden_split(
        self,
        tr: dict,
        player: str,
        combined_counts: Dict[str, int],
        hidden_slots: int,
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        pieces = tuple(str(i) for i in range(1, 10))
        if hidden_slots <= 0:
            return ({piece: 0 for piece in pieces}, dict(combined_counts))

        best_score = -1.0
        best_hidden: Optional[Dict[str, int]] = None

        def visit(index: int, remaining: int, hidden: Dict[str, int], score: float) -> None:
            nonlocal best_score, best_hidden
            if index == len(pieces):
                if remaining == 0 and score > best_score:
                    best_score = score
                    best_hidden = dict(hidden)
                return
            piece = pieces[index]
            available = int(combined_counts.get(piece, 0))
            for count in range(min(available, remaining) + 1):
                hidden[piece] = count
                factor = comb(available, count) * (self._block_piece_weight(tr, player, piece) ** count)
                visit(index + 1, remaining - count, hidden, score * factor)
            hidden.pop(piece, None)

        visit(0, hidden_slots, {}, 1.0)
        hidden_result = best_hidden or {piece: 0 for piece in pieces}
        current_result = {
            piece: int(combined_counts.get(piece, 0)) - int(hidden_result.get(piece, 0))
            for piece in pieces
        }
        return hidden_result, current_result

    def _refresh_joint_hand_inference(
        self,
        state,
        tr: dict,
        current_estimates: Dict[str, Dict[str, Dict[str, object]]],
        unknown_pool: Dict[str, int],
        *,
        reason: str,
    ) -> None:
        if self.me is None:
            return

        pieces = tuple(str(i) for i in range(1, 10))
        players = tuple(p for p in ("A", "B", "C", "D") if p != self.me)
        hidden_counts = tr.get("hidden_block_counts", {})
        capacities = tuple(
            len(state.hands[player]) + int(hidden_counts.get(player, 0))
            for player in players
        )
        observed = {
            player: {
                piece: self._observed_piece_count_for_player(tr, player, piece)
                for piece in pieces
            }
            for player in players
        }

        allocations_by_piece: Dict[str, List[Tuple[Tuple[int, int, int], float]]] = {}
        for piece in pieces:
            lower_values: List[int] = []
            upper_values: List[int] = []
            for player_index, player in enumerate(players):
                estimate = self._ensure_piece_count_estimate(tr, player, piece)
                used = observed[player][piece]
                lower = max(0, int(estimate.get("min", 0)) - used)
                upper = max(0, int(estimate.get("max", self._piece_total(piece))) - used)
                current_cap = self._active_current_piece_count_cap(tr, player, piece)
                if current_cap is not None:
                    upper = min(upper, int(hidden_counts.get(player, 0)) + current_cap)
                lower_values.append(min(capacities[player_index], lower))
                upper_values.append(min(capacities[player_index], max(lower, upper)))
            allocations = self._joint_piece_allocations(
                int(unknown_pool[piece]),
                tuple(lower_values),
                tuple(upper_values),
            )
            if not allocations:
                tr["joint_hand_inference"] = {
                    "feasible": False,
                    "reason": reason,
                    "failed_piece": piece,
                }
                return
            weighted_allocations: List[Tuple[Tuple[int, int, int], float]] = []
            for allocation in allocations:
                transition_log = 0.0
                for player_index, player in enumerate(players):
                    count = allocation[player_index]
                    transition_log -= log(factorial(count))
                    original_count = count + observed[player][piece]
                    transition_log += log(
                        self._pass_evidence_likelihood(
                            tr,
                            player,
                            piece,
                            original_count,
                        )
                    )
                weighted_allocations.append((allocation, transition_log))
            allocations_by_piece[piece] = weighted_allocations

        states: Dict[Tuple[int, int, int], Tuple[float, List[Tuple[int, int, int]]]] = {
            (0, 0, 0): (0.0, [])
        }
        for piece_index, piece in enumerate(pieces):
            next_states: Dict[Tuple[int, int, int], Tuple[float, List[Tuple[int, int, int]]]] = {}
            for used_slots, (best_log, best_path) in states.items():
                for allocation, transition_log in allocations_by_piece[piece]:
                    new_slots = tuple(used_slots[i] + allocation[i] for i in range(3))
                    if any(new_slots[i] > capacities[i] for i in range(3)):
                        continue
                    candidate_log = best_log + transition_log
                    previous = next_states.get(new_slots)
                    if previous is None or candidate_log > previous[0]:
                        next_states[new_slots] = (candidate_log, best_path + [allocation])
            states = next_states

        result = states.get(capacities)
        if result is None:
            tr["joint_hand_inference"] = {
                "feasible": False,
                "reason": reason,
                "failed_capacity": capacities,
            }
            return

        best_log, map_path = result
        map_combined: Dict[str, Dict[str, int]] = {player: {} for player in players}
        for piece_index, piece in enumerate(pieces):
            allocation = map_path[piece_index]
            for player_index, player in enumerate(players):
                map_combined[player][piece] = int(allocation[player_index])

        map_original: Dict[str, Dict[str, int]] = {player: {} for player in players}
        map_current: Dict[str, Dict[str, int]] = {}
        map_hidden: Dict[str, Dict[str, int]] = {}
        for player in players:
            map_original[player] = {
                piece: map_combined[player][piece] + observed[player][piece]
                for piece in pieces
            }
            hidden, current = self._most_likely_hidden_split(
                tr,
                player,
                map_combined[player],
                int(hidden_counts.get(player, 0)),
            )
            map_hidden[player] = hidden
            map_current[player] = current

            for piece in pieces:
                estimate = current_estimates[player][piece]
                estimate["map_count"] = map_current[player][piece]

        tr["joint_hand_inference"] = {
            "feasible": True,
            "reason": reason,
            "players": list(players),
            "map_original_counts": map_original,
            "map_combined_counts": map_combined,
            "map_current_counts": map_current,
            "map_hidden_counts": map_hidden,
            "map_log_weight": best_log,
        }

    def _refresh_public_piece_inference(
        self,
        state,
        tr: dict,
        *,
        reason: str,
    ) -> None:
        if self.me is None:
            return

        players = [p for p in ("A", "B", "C", "D") if p != self.me]
        hidden_counts = tr.get("hidden_block_counts", {})
        current_estimates: Dict[str, Dict[str, Dict[str, object]]] = {
            p: {} for p in players
        }
        unknown_pool: Dict[str, int] = {}

        for piece in (str(i) for i in range(1, 10)):
            for other in players:
                self._validate_current_piece_count_cap(tr, other, piece)
                self._ensure_piece_count_estimate(tr, other, piece)
            self._reconcile_piece_count_estimates(state, tr, piece)

            observed = {
                p: self._observed_piece_count_for_player(tr, p, piece)
                for p in players
            }
            pool_count = max(
                0,
                self._piece_total(piece)
                - int(tr.get("my_init_count", Counter()).get(piece, 0))
                - sum(observed.values()),
            )
            unknown_pool[piece] = pool_count

            hand_slots = {p: len(state.hands[p]) for p in players}
            hidden_slots = {p: int(hidden_counts.get(p, 0)) for p in players}
            combined_slots = {
                p: hand_slots[p] + hidden_slots[p]
                for p in players
            }
            lower: Dict[str, int] = {}
            upper: Dict[str, int] = {}
            for p in players:
                estimate = self._ensure_piece_count_estimate(tr, p, piece)
                remaining_min = max(0, int(estimate.get("min", 0)) - observed[p])
                remaining_max = max(
                    0,
                    int(estimate.get("max", self._piece_total(piece))) - observed[p],
                )
                lower[p] = min(combined_slots[p], remaining_min)
                upper[p] = min(combined_slots[p], max(lower[p], remaining_max))

            # Propagate the exact number still hidden across the three players.
            for _ in range(6):
                changed = False
                for p in players:
                    other_players = [q for q in players if q != p]
                    new_min = max(
                        lower[p],
                        pool_count - sum(upper[q] for q in other_players),
                    )
                    new_max = min(
                        upper[p],
                        pool_count - sum(lower[q] for q in other_players),
                    )
                    new_min = max(0, min(combined_slots[p], new_min))
                    new_max = max(new_min, min(combined_slots[p], new_max))
                    if new_min != lower[p] or new_max != upper[p]:
                        lower[p] = new_min
                        upper[p] = new_max
                        changed = True
                if not changed:
                    break

            expected_combined = self._bounded_piece_expectations(
                pool_count,
                players,
                lower,
                upper,
                combined_slots,
            )
            for p in players:
                hand_min = max(0, lower[p] - hidden_slots[p])
                hand_max = min(hand_slots[p], upper[p])
                current_cap = self._active_current_piece_count_cap(tr, p, piece)
                if current_cap is not None:
                    hand_min = min(hand_min, current_cap)
                    hand_max = min(hand_max, current_cap)
                if combined_slots[p] > 0:
                    expected_hand = expected_combined[p] * hand_slots[p] / combined_slots[p]
                else:
                    expected_hand = 0.0
                expected_hand = max(float(hand_min), min(float(hand_max), expected_hand))
                width = max(0, hand_max - hand_min)
                confidence = 1.0 if hand_slots[p] == 0 else max(
                    0.05,
                    min(1.0, 1.0 - width / max(1, hand_slots[p])),
                )
                original_estimate = self._ensure_piece_count_estimate(tr, p, piece)
                current_estimates[p][piece] = {
                    "min": hand_min,
                    "max": hand_max,
                    "expected": round(expected_hand, 3),
                    "confidence": round(confidence, 3),
                    "observed_used": observed[p],
                    "hidden_slots": hidden_slots[p],
                    "source": str(original_estimate.get("source", "public_pool")),
                }

        tr["estimated_current_hands"] = current_estimates
        tr["unknown_piece_pool"] = unknown_pool
        self._refresh_joint_hand_inference(
            state,
            tr,
            current_estimates,
            unknown_pool,
            reason=reason,
        )
        tr["piece_inference_revision"] = int(tr.get("piece_inference_revision", 0)) + 1
        tr["last_piece_inference_reason"] = reason

    def _estimated_current_piece(
        self,
        tr: Optional[dict],
        player: Optional[str],
        piece: Optional[str],
    ) -> Optional[Dict[str, object]]:
        if tr is None or player is None or piece is None:
            return None
        return tr.get("estimated_current_hands", {}).get(player, {}).get(piece)

    def _record_first_attack_count_estimate(
        self,
        state,
        tr: dict,
        player: str,
        attack: Optional[str],
        strategy: Optional[Dict[str, object]],
    ) -> None:
        if player == self.me or attack is None:
            return
        count_range = self._first_attack_count_range(attack)
        if count_range is None:
            return
        min_count, max_count = count_range
        label = str(strategy.get("label", "first_attack")) if strategy is not None else "first_attack"
        self._set_piece_count_estimate(
            tr,
            player,
            attack,
            min_count=min_count,
            max_count=max_count,
            source=label,
        )
        if attack in ("8", "9") and strategy is not None:
            other_king = "8" if attack == "9" else "9"
            self._set_piece_count_estimate(
                tr,
                player,
                other_king,
                min_count=1,
                max_count=1,
                source=label,
            )
            self._reconcile_piece_count_estimates(state, tr, other_king)
        self._reconcile_piece_count_estimates(state, tr, attack)

    def _infer_single_remaining_unique_piece_holder(
        self,
        state,
        tr: dict,
        piece: str,
        *,
        source: str,
    ) -> None:
        if self.me is None or self._piece_total(piece) != 1:
            return
        if state.hands[self.me].count(piece) > 0:
            return
        if int(tr.get("public_seen_counts", {}).get(piece, 0)) > 0:
            return

        candidates: List[str] = []
        for other in ("A", "B", "C", "D"):
            if other == self.me:
                continue
            observed = self._observed_piece_count_for_player(tr, other, piece)
            if observed > 0:
                return
            est = self._piece_count_estimate(tr, other, piece)
            max_count = self._piece_total(piece) if est is None else int(est.get("max", self._piece_total(piece)))
            if max_count > 0:
                candidates.append(other)

        if len(candidates) != 1:
            return

        self._set_piece_count_estimate(
            tr,
            candidates[0],
            piece,
            min_count=1,
            max_count=1,
            source=source,
        )
        self._reconcile_piece_count_estimates(state, tr, piece)

    def _infer_missing_kings_from_third_visible_attack(
        self,
        state,
        tr: dict,
        player: str,
        action_type: str,
        attack: Optional[str],
    ) -> None:
        if player == self.me or action_type != "attack" or attack in ("8", "9"):
            return

        model = tr.get("public_hand_models", {}).get(player, {})
        if int(model.get("attack_count", 0)) != 3:
            return

        for king in ("8", "9"):
            if self._observed_piece_count_for_player(tr, player, king) > 0:
                continue
            self._set_piece_count_estimate(
                tr,
                player,
                king,
                min_count=0,
                max_count=0,
                source="third_attack_no_king",
            )
            self._reconcile_piece_count_estimates(state, tr, king)
            self._infer_single_remaining_unique_piece_holder(
                state,
                tr,
                king,
                source="single_remaining_after_third_attack",
            )

    def _opponent_first_attack_strategy_penalty(self, tr: dict, player: str, attack: str) -> float:
        penalty = 0.0
        guesses = tr.get("other_first_attack_strategy_by_player", {})
        for other, guess in guesses.items():
            if self._same_team(other, player):
                continue
            if guess.get("active", True) is False:
                continue
            if attack in set(str(p) for p in guess.get("pieces", [])):
                estimates = tr.get("other_piece_count_estimates", {})
                est = estimates.get(other, {}).get(attack)
                if est is not None:
                    model = tr.get("public_hand_models", {}).get(other, {})
                    observed_count = int(model.get("attacks", Counter()).get(attack, 0)) + int(
                        model.get("blocks", Counter()).get(attack, 0)
                    )
                    if int(est.get("max", 0)) <= observed_count:
                        continue
                penalty += float(self.OPPONENT_FIRST_ATTACK_STRATEGY_SAFE_PENALTY.get(attack, 0.0))
        return penalty

    def _observed_piece_count_for_player(self, tr: dict, player: str, piece: str) -> int:
        model = tr.get("public_hand_models", {}).get(player, {})
        return int(model.get("attacks", Counter()).get(piece, 0)) + int(
            model.get("blocks", Counter()).get(piece, 0)
        )

    def _piece_count_estimate(self, tr: Optional[dict], player: Optional[str], piece: Optional[str]) -> Optional[Dict[str, object]]:
        if tr is None or player is None or piece is None:
            return None
        return tr.get("other_piece_count_estimates", {}).get(player, {}).get(piece)

    def _estimate_remaining_range(self, tr: Optional[dict], player: Optional[str], piece: Optional[str]) -> Tuple[int, int]:
        current = self._estimated_current_piece(tr, player, piece)
        if current is not None:
            return (int(current.get("min", 0)), int(current.get("max", 0)))
        est = self._piece_count_estimate(tr, player, piece)
        if tr is None or player is None or piece is None:
            return (0, self._piece_total(piece or "1"))
        observed = self._observed_piece_count_for_player(tr, player, piece)
        if est is None:
            return (0, max(0, self._piece_total(piece) - observed))
        min_count = int(est.get("min", 0))
        max_count = int(est.get("max", self._piece_total(piece)))
        return (max(0, min_count - observed), max(0, max_count - observed))

    def _estimate_remaining_expected(
        self,
        tr: Optional[dict],
        player: Optional[str],
        piece: Optional[str],
    ) -> float:
        current = self._estimated_current_piece(tr, player, piece)
        if current is None:
            return 0.0
        expected = max(0.0, float(current.get("expected", 0.0)))
        map_count = current.get("map_count")
        if map_count is None:
            return expected
        return (expected * 0.45) + (max(0.0, float(map_count)) * 0.55)

    def _opponents_piece_pressure(self, tr: Optional[dict], player: str, piece: Optional[str]) -> float:
        if tr is None or piece is None:
            return 0.0
        pressure = 0.0
        for other in ("A", "B", "C", "D"):
            if other == player or self._same_team(other, player):
                continue
            remaining_min, remaining_max = self._estimate_remaining_range(tr, other, piece)
            remaining_expected = self._estimate_remaining_expected(tr, other, piece)
            if remaining_min > 0:
                pressure += 1.0 + remaining_min + min(1.0, remaining_expected * 0.25)
            elif remaining_max > 0:
                pressure += 0.2 + min(1.3, remaining_expected)
        return pressure

    def _opponents_piece_exhausted(self, tr: Optional[dict], player: str, piece: Optional[str]) -> bool:
        if tr is None or piece is None:
            return False
        saw_estimate = False
        for other in ("A", "B", "C", "D"):
            if other == player or self._same_team(other, player):
                continue
            est = self._piece_count_estimate(tr, other, piece)
            if est is not None:
                saw_estimate = True
            _remaining_min, remaining_max = self._estimate_remaining_range(tr, other, piece)
            if remaining_max > 0:
                return False
        return saw_estimate

    def _enemy_likely_to_repeat_shi(self, tr: Optional[dict], player: str) -> bool:
        if tr is None:
            return False
        models = tr.get("public_hand_models", {})
        for enemy in ("A", "B", "C", "D"):
            if enemy == player or self._same_team(enemy, player):
                continue
            attacks = models.get(enemy, {}).get("attacks", Counter())
            if int(attacks.get("1", 0)) <= 0:
                continue
            remaining_min, _remaining_max = self._estimate_remaining_range(tr, enemy, "1")
            remaining_expected = self._estimate_remaining_expected(tr, enemy, "1")
            if remaining_min > 0 or remaining_expected >= 0.75:
                return True
        return False

    def _public_attack_evidence(self, attack: str) -> float:
        if attack in ("8", "9"):
            return 7.0
        if attack in ("6", "7"):
            return 4.0
        if attack == "2":
            return 3.5
        if attack in ("3", "4", "5"):
            return 3.0
        if attack == "1":
            return 0.5
        return 0.0

    def _estimated_rank_from_public_score(self, score: float) -> str:
        if score >= 18.0:
            return "S"
        if score >= 15.0:
            return "A"
        if score >= 12.0:
            return "B"
        if score >= 8.0:
            return "C"
        if score >= 4.0:
            return "D"
        if score >= 1.0:
            return "E"
        if score >= -1.0:
            return "F"
        return "X"

    def _public_rank_action_signal(
        self,
        tr: dict,
        player: str,
        action_type: str,
        block: Optional[str],
        attack: Optional[str],
        *,
        is_first_attack: bool,
        attack_seen_before: int,
    ) -> Tuple[float, float, str]:
        context = tr.get("active_attack_context") or {}

        if action_type == "pass":
            attacker = context.get("attacker")
            attack_no = max(1, int(context.get("attack_no", 1)))
            if attacker is not None and not self._same_team(str(attacker), player):
                delta = {1: -0.3, 2: -0.7, 3: -1.0}.get(attack_no, -1.2)
                return delta, 0.3 + min(0.3, attack_no * 0.1), f"pass_enemy_attack_{attack_no}"
            if attacker is not None:
                return -0.1, 0.15, f"pass_ally_attack_{attack_no}"
            return -0.1, 0.1, "pass_unknown_timing"

        if action_type == "receive":
            attack_no = max(1, int(context.get("attack_no", 1)))
            attacker = context.get("attacker")
            enemy_attack = attacker is not None and not self._same_team(str(attacker), player)
            delta = 1.0 if block in ("8", "9") else 0.7
            if enemy_attack:
                delta += min(0.6, max(0, attack_no - 1) * 0.3)
            else:
                delta *= 0.5
            block_label = "royal" if block in ("8", "9") else "same"
            relation = "enemy" if enemy_attack else "ally"
            return delta, 0.7, f"receive_{relation}_attack_{attack_no}_{block_label}"

        if action_type in ("attack", "attack_after_block") and attack is not None:
            if is_first_attack:
                strategy = self._infer_first_attack_strategy(attack)
                label = str(strategy.get("label", "other")) if strategy is not None else "other"
                delta_by_label = {
                    "king_pair_or_king_attack": 9.0,
                    "kyosha_repeat": 4.5,
                    "middle_repeat": 3.0,
                    "big_pair": 4.0,
                    "shi_attack": 0.0,
                    "other": 0.5,
                }
                return delta_by_label.get(label, 0.5), 1.5, f"first_attack_{label}"

            first_attack = tr.get("public_hand_models", {}).get(player, {}).get("first_attack")
            if attack_seen_before > 0 or attack == first_attack:
                delta_by_piece = {
                    "1": 0.5,
                    "2": 3.0,
                    "3": 2.5,
                    "4": 2.5,
                    "5": 2.5,
                    "6": 3.5,
                    "7": 3.5,
                    "8": 2.0,
                    "9": 2.0,
                }
                return delta_by_piece.get(attack, 1.0), 1.0, f"repeat_attack_{attack}"
            model = tr.get("public_hand_models", {}).get(player, {})
            reaction = model.get("partner_first_strategy_reaction") or {}
            if (
                first_attack is not None
                and reaction.get("status") == "rejected"
                and reaction.get("piece") == first_attack
            ):
                reason = str(reaction.get("reason", "partner_rejected"))
                return -1.0, 1.0, f"first_strategy_broken_{reason}"
            return 0.4, 0.5, f"different_attack_{attack}"

        return 0.0, 0.0, "no_rank_signal"

    def _update_estimated_public_rank(
        self,
        tr: dict,
        player: str,
        *,
        delta: float,
        evidence_weight: float,
        reason: str,
    ) -> None:
        model = tr.get("public_hand_models", {}).get(player)
        if model is None:
            return

        score = max(-3.0, min(22.0, float(model.get("estimated_rank_score", 5.0)) + delta))
        weight = float(model.get("rank_evidence_weight", 0.0)) + max(0.0, evidence_weight)
        rank = self._estimated_rank_from_public_score(score)
        confidence = 0.0 if weight <= 0 else min(0.95, 0.12 + weight * 0.12)
        model["estimated_rank_score"] = score
        model["estimated_rank"] = rank
        model["rank_evidence_weight"] = weight
        model["rank_confidence"] = confidence

        history = model.setdefault("rank_history", [])
        history.append(
            {
                "rank": rank,
                "score": round(score, 2),
                "confidence": round(confidence, 3),
                "reason": reason,
            }
        )
        if len(history) > 24:
            del history[:-24]

    def _public_hand_rank_estimate(self, tr: Optional[dict], player: Optional[str]) -> Dict[str, object]:
        if tr is None or player is None:
            return {"rank": "D", "score": 5.0, "confidence": 0.0, "reason": "no_evidence"}
        model = tr.get("public_hand_models", {}).get(player)
        if model is None:
            return {"rank": "D", "score": 5.0, "confidence": 0.0, "reason": "no_evidence"}
        history = model.get("rank_history", [])
        reason = str(history[-1].get("reason", "no_evidence")) if history else "no_evidence"
        return {
            "rank": str(model.get("estimated_rank", "D")),
            "score": float(model.get("estimated_rank_score", 5.0)),
            "confidence": float(model.get("rank_confidence", 0.0)),
            "reason": reason,
        }

    def _update_public_hand_model(
        self,
        state,
        tr: dict,
        player: str,
        action_type: str,
        block: Optional[str],
        attack: Optional[str],
    ) -> None:
        models = tr.get("public_hand_models")
        if models is None or player not in models:
            return

        self._track_partner_first_strategy_reaction(tr, player, action_type, block, attack)
        model = models[player]
        attack_seen_before = int(model.get("attacks", Counter()).get(attack, 0)) if attack is not None else 0
        is_first_attack = action_type in ("attack", "attack_after_block") and int(model.get("attack_count", 0)) == 0
        rank_delta, rank_weight, rank_reason = self._public_rank_action_signal(
            tr,
            player,
            action_type,
            block,
            attack,
            is_first_attack=is_first_attack,
            attack_seen_before=attack_seen_before,
        )

        if action_type == "pass":
            self._record_piece_pass_evidence(state, tr, player)
            model["pass_count"] += 1
            model["strength"] -= 0.25
            self._record_shi_pass_current_range(state, tr, player)

        if action_type in ("receive", "attack_after_block") and block is not None:
            model["receive_count"] += 1
            model["blocks"][block] += 1
            if player != self.me:
                self._update_observed_piece_count_estimate(tr, player, block)
                self._reconcile_piece_count_estimates(state, tr, block)
            model["strength"] += 4.0 if block in ("8", "9") else 1.5

        if action_type in ("attack", "attack_after_block") and attack is not None:
            first_attack = model.get("first_attack")
            model["attack_count"] += 1
            model["attacks"][attack] += 1
            if is_first_attack:
                strategy = self._infer_first_attack_strategy(attack)
                model["first_attack"] = attack
                model["inferred_attack_strategy"] = strategy["label"] if strategy is not None else None
                model["inferred_attack_strategy_active"] = strategy is not None
                if player != self.me and strategy is not None:
                    strategy["active"] = True
                    tr.setdefault("other_first_attack_strategy_by_player", {})[player] = strategy
                    self._record_first_attack_count_estimate(state, tr, player, attack, strategy)
                    model["strength"] += float(strategy.get("strength", 0.0))
            elif player != self.me:
                self._update_observed_piece_count_estimate(tr, player, attack)
                self._reconcile_piece_count_estimates(state, tr, attack)
                if first_attack is not None and attack != first_attack:
                    self._break_first_attack_strategy_after_partner_rejection(
                        state,
                        tr,
                        player,
                        str(first_attack),
                        attack,
                    )
            model["strength"] += self._public_attack_evidence(attack)
            if attack_seen_before >= 1:
                model["strength"] += 3.0 + attack_seen_before
            if action_type == "attack_after_block":
                model["strength"] += 1.5

            tr["active_attack_context"] = {
                "attacker": player,
                "piece": attack,
                "attack_no": int(model.get("attack_count", 0)),
            }

        model["strength"] = max(-6.0, min(40.0, float(model["strength"])))
        self._update_estimated_public_rank(
            tr,
            player,
            delta=rank_delta,
            evidence_weight=rank_weight,
            reason=rank_reason,
        )
        if action_type == "receive":
            tr["active_attack_context"] = None

    def _public_hand_strength(self, tr: Optional[dict], player: Optional[str]) -> float:
        if not getattr(self, "USE_PUBLIC_HAND_INFERENCE", True):
            return 0.0
        if tr is None or player is None:
            return 0.0
        model = tr.get("public_hand_models", {}).get(player)
        if model is None:
            return 0.0
        return float(model.get("strength", 0.0))
