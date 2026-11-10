"""Choose how to finish an established shi attack from public evidence."""


class FinalShiContinuationMixin:
    # The debug room can opt into the risk policy for this decision only.
    final_shi_risk = False

    def _shi_attack_package_action(
        self, state, player, actions, *, has_non_king_attack_option,
    ):
        established = super()._shi_attack_package_action(
            state, player, actions,
            has_non_king_attack_option=has_non_king_attack_option,
        )
        if established is not None or not self.SHI_ATTACK_PACKAGE_ENABLED:
            return established

        tracker = self._track.get(id(state), {})
        analysis = self.last_shi_attack_package_analysis
        if (
            state.phase != "attack" or state.turn != player
            or len(state.hands[player]) != 4
            or not analysis or analysis["attack_number"] != 3
            or analysis["own_current_shi"] != 1
            or tracker.get("ally_first_attack") != "1"
            or tracker.get("ally_last_attack") != "1"
            or tracker.get("ally_shi_signal") in ("weak", "sashikomi")
            or tuple(tracker.get("my_attack_history", ())) != ("1", "1")
            or analysis["enemy_map_remaining"] > 1
            or analysis["enemy_expected_remaining"] > 2.25
            or analysis["enemy_inference_confidence"] < self.SHI_ATTACK_PACKAGE_MIN_CONFIDENCE
        ):
            return None

        # A full lap on the previous shi is evidence about this same attack.
        # Earlier passes, or passes on an intervening piece, are not enough.
        history = self._neural_public_history_by_state_id.get(id(state), ())
        lap = history[-4:]
        if (
            len(lap) != 4 or lap[0]["player"] != player
            or lap[0]["action"][2] != "1"
            or {event["player"] for event in lap[1:] if event["action"][0] == "pass"}
            != set("ABCD") - {player}
        ):
            return None
        enemies = analysis["enemy_seats"]
        # A ready enemy may win on receipt; a ready partner may have a better
        # handoff. Leave those cases to the existing endgame comparisons.
        if any(len(state.hands[seat]) <= 2 for seat in "ABCD" if seat != player):
            return None
        pool = self._public_unknown_piece_pool(state, player, tracker)
        if pool is None:
            return None

        def enemy_may_hold(piece):
            return pool.get(piece, 0) > 0 and any(
                self._estimated_piece_hold_risk(tracker, seat, piece) != 0.0
                for seat in enemies
            )

        # Prefer the shi plan over ordinary attacks exposed to a royal
        # receive. Do not force shi over a different credible route
        # such as kyosha or a public last-copy attack after both royals are gone.
        alternative_pieces = {attack for _kind, _block, attack in actions if attack not in (None, "1")}
        if not alternative_pieces or not alternative_pieces <= set("34567") or not any(
            enemy_may_hold(royal) for royal in ("8", "9")
        ):
            return None

        possible_returns = {piece for piece in "23456789" if enemy_may_hold(piece)}
        candidates = []
        for action in actions:
            if action[0] != "attack_after_block" or action[2] != "1":
                continue
            pair = self._hand_after_attack_action(state.hands[player], action)
            if pair is None or len(pair) != 2:
                continue
            cover = set(pair)
            if cover & {"8", "9"}:
                cover.update("3456789")
            returns = tuple(sorted(cover & possible_returns))
            finish = int(self._pair_finish_score(pair))
            candidates.append((action, returns, finish))
        if not candidates:
            return None

        risk = self.final_shi_risk
        # Recovery can only score the piece left after receiving; a double
        # is possible only if the attack makes a full lap.
        def preference(candidate):
            action, returns, finish = candidate
            cover = len(returns)
            return (finish, cover, action[1]) if risk else (cover, finish, action[1])

        chosen, returns, finish = max(candidates, key=preference)
        self.last_attack_candidate_scores = [
            {"action": action, "score": points if risk else len(covered),
             "candidate_role": "final_shi_continuation"}
            for action, covered, points in candidates
        ]
        analysis.update(
            purpose="finish_third_attack_after_full_lap",
            selected_action=list(chosen),
            continuation_style="risk" if risk else "safe",
            continuation_cover=returns,
            continuation_finish=finish,
        )
        return chosen, analysis

    def _shi_attack_package_detail(self, analysis):
        if analysis.get("purpose") != "finish_third_attack_after_full_lap":
            return super()._shi_attack_package_detail(analysis)
        return (
            f"shi_package_continuation_{analysis['continuation_style']}_"
            f"enemy_{analysis['enemy_map_remaining']}_"
            f"cover_{''.join(analysis['continuation_cover']) or 'none'}_"
            f"finish_{analysis['continuation_finish']}"
        )
