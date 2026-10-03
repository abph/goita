"""Create a private review of how well an AI profile follows one player.

The recorded move is hidden until after the AI has selected its move.  Rounds
where any player was dealt five or more shi are excluded.  A receive and its
immediate attack are treated as one decision route.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from goita_ai2.constants import PIECE_KANJI
from goita_ai2.kifu_validation import (
    _apply_action,
    iter_kifu_decisions,
    replay_validation_case,
    write_json,
)
from goita_ai2.current_ai.agent import RuleBasedAgent as CurrentRuleBasedAgent
from goita_ai2.experimental_ai2.agent import (
    RuleBasedAgent as ExperimentalAI2RuleBasedAgent,
)


Action = Tuple[str, Optional[str], Optional[str]]
DEFAULT_KIFU_PATH = Path("private_data/kifu_data_raw.json")
DEFAULT_JSON_PATH = Path("private_data/1222_ai_understanding_review.json")
DEFAULT_HTML_PATH = Path("private_data/1222_ai_understanding_review.html")
DEFAULT_AI_PROFILE = "experimental_ai2"
AI_PROFILES = {
    "current": ("強化中AI", CurrentRuleBasedAgent),
    "experimental_ai2": ("強化中AI2", ExperimentalAI2RuleBasedAgent),
}
RATINGS = (
    "理解できる",
    "少し理解できる",
    "なんともいえない",
    "あまり理解できない",
    "理解できない",
)
REVIEW_FIELDS = (
    "final_rating",
    "recorded_move_quality",
    "purpose",
    "ai_explanation_close",
    "note",
)


def _resolve_ai_profile(profile: str):
    try:
        return AI_PROFILES[str(profile)]
    except KeyError as error:
        choices = ", ".join(sorted(AI_PROFILES))
        raise ValueError(
            f"unknown AI profile: {profile!r}; choose one of {choices}"
        ) from error


def _empty_review() -> Dict[str, str]:
    return {field: "" for field in REVIEW_FIELDS}


def _normalized_review(value: object) -> Dict[str, str]:
    source = dict(value or {}) if isinstance(value, Mapping) else {}
    return {field: str(source.get(field, "") or "") for field in REVIEW_FIELDS}


def _has_review(value: Mapping[str, str]) -> bool:
    return any(item.strip() for item in value.values())


def _player_seats(archive: Mapping[str, object], player_name: str) -> Dict[str, str]:
    seats: Dict[str, str] = {}
    for match in archive.get("matches", []):  # type: ignore[union-attr]
        players = dict(match.get("players", {}) or {})
        for index, seat in enumerate("ABCD"):
            if str(players.get(f"p{index}", "")) == player_name:
                seats[str(match.get("id", ""))] = seat
                break
    return seats


def _excluded_five_shi_rounds(
    archive: Mapping[str, object],
    included_matches: Iterable[str],
) -> set[Tuple[str, int]]:
    included = set(included_matches)
    excluded = set()
    for match in archive.get("matches", []):  # type: ignore[union-attr]
        match_id = str(match.get("id", ""))
        if match_id not in included:
            continue
        for round_item in match.get("rounds", []):
            hands = dict(round_item.get("hand", {}) or {})
            if any(str(hand).count("し") >= 5 for hand in hands.values()):
                excluded.add((match_id, int(round_item.get("round_index", 0))))
    return excluded


def _source_key(case: Mapping[str, object]) -> Tuple[str, int, int]:
    source = dict(case.get("source", {}) or {})
    return (
        str(source.get("match_id", "")),
        int(source.get("round_index", 0)),
        int(source.get("decision_index", 0)),
    )


def _stratum(case: Mapping[str, object]) -> str:
    phase = str(case.get("category", "middle_unknown")).split("_", 1)[0]
    action = str(list(case.get("actual_action", []))[0])
    kind = "response" if action in ("pass", "receive") else "attack"
    return f"{phase}_{kind}"


def build_player_decision_units(
    kifu_path: Path,
    *,
    player_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Return non-forced decision routes for one named player."""
    archive = json.loads(kifu_path.read_text(encoding="utf-8"))
    seats = _player_seats(archive, player_name)
    excluded = _excluded_five_shi_rounds(archive, seats)
    player_cases = []
    for case in iter_kifu_decisions(kifu_path):
        match_id, round_index, _decision_index = _source_key(case)
        if (match_id, round_index) in excluded:
            continue
        if str(case.get("player")) != seats.get(match_id):
            continue
        player_cases.append(case)

    units: List[Dict[str, Any]] = []
    skipped_followups = set()
    by_key = {_source_key(case): case for case in player_cases}
    for case in player_cases:
        key = _source_key(case)
        if key in skipped_followups:
            continue
        if int(case.get("position", {}).get("legal_action_count", 0)) <= 1:
            continue
        actual: Action = tuple(case["actual_action"])  # type: ignore[assignment]
        followup = None
        if actual[0] == "receive":
            next_key = (key[0], key[1], key[2] + 1)
            candidate = by_key.get(next_key)
            if (
                candidate is not None
                and candidate.get("player") == case.get("player")
                and list(candidate.get("actual_action", []))[0] == "attack"
            ):
                followup = copy.deepcopy(candidate)
                skipped_followups.add(next_key)
        units.append({
            "id": str(case["id"]),
            "stratum": _stratum(case),
            "case": copy.deepcopy(case),
            "recorded_followup": followup,
        })
    summary = {
        "declared_matches": int(archive.get("match_count", 0)),
        "player_matches": len(seats),
        "excluded_five_shi_rounds": len(excluded),
        "eligible_decision_units": len(units),
    }
    return units, summary


def select_balanced_units(
    units: Sequence[Dict[str, Any]],
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    """Deterministically balance opening/middle/endgame and response/attack."""
    buckets: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for unit in units:
        rank = hashlib.sha256(str(unit["id"]).encode("utf-8")).hexdigest()
        buckets[str(unit["stratum"])].append((rank, unit))
    for bucket in buckets.values():
        bucket.sort(key=lambda item: item[0])

    selected: List[Dict[str, Any]] = []
    ordered_strata = sorted(buckets)
    cursor = 0
    while len(selected) < max(0, int(limit)) and ordered_strata:
        stratum = ordered_strata[cursor % len(ordered_strata)]
        bucket = buckets[stratum]
        index = cursor // len(ordered_strata)
        if index < len(bucket):
            selected.append(copy.deepcopy(bucket[index][1]))
        cursor += 1
        if cursor > max(1, int(limit)) * len(ordered_strata) * 2:
            break
    selected.sort(key=lambda item: (str(item["stratum"]), str(item["id"])))
    return selected[: max(0, int(limit))]


def _action_score_snapshot(agent, state, player: str, actions: Sequence[Action]) -> list:
    has_non_royal = any(
        action[0] in ("attack", "attack_after_block")
        and action[2] not in (None, "8", "9")
        for action in actions
    )
    scores = []
    for action in actions:
        action_type, block, attack = action
        try:
            if action_type == "attack_after_block":
                score = agent._score_receive_phase(state, player, "receive", block)
                score += agent._score_attack_phase(
                    state, player, action_type, block, attack,
                    has_non_king_attack_option=has_non_royal,
                )
            elif action_type == "attack":
                score = agent._score_attack_phase(
                    state, player, action_type, block, attack,
                    has_non_king_attack_option=has_non_royal,
                )
            else:
                score = agent._score_receive_phase(state, player, action_type, block)
            scores.append({"action": list(action), "score": round(float(score), 2)})
        except Exception:
            continue
    scores.sort(key=lambda item: (-float(item["score"]), str(item["action"])))
    for index, item in enumerate(scores, start=1):
        item["rank"] = index
    return scores


def _public_history(case: Mapping[str, object]) -> list:
    owner = str(case.get("player", ""))
    result = []
    for item in case.get("history", []):
        actor = str(item.get("player", ""))
        action = list(item.get("action", []))
        if len(action) != 3:
            continue
        if action[0] == "attack_after_block" and actor != owner:
            action[1] = None
        result.append({"player": actor, "action": action})
    return result


def _route(root: Action, followup: Optional[Action]) -> List[List[Optional[str]]]:
    route = [list(root)]
    if followup is not None:
        route.append(list(followup))
    return route


def _explain_ai_reason(reason: str, detail: str) -> str:
    """Turn internal reason codes into a short review-facing explanation."""
    if reason == "win_now":
        return "この手で上がれるため選びました。"
    if reason == "tsume":
        return "上がりが確定する、または高得点の上がり筋を優先しました。"
    if reason == "upside_finish":
        return "上がりを保ちながら、より高い得点を狙える筋を選びました。"
    if reason == "time_search":
        return "見えていない手駒を複数通り推定し、その後の展開を比較しました。"
    if reason == "response_dictionary":
        return "同じ公開情報の局面で事前に調べた応答計画を使いました。"
    if reason == "shi_insertion":
        return "しを相方まで届ける差し込みの条件を満たすと判断しました。"
    if reason == "shi_signal":
        return "しの攻めと相方の反応を合図として判断しました。"
    if reason == "inferred_endgame":
        return "公開情報から推定した終盤の上がり筋を続けました。"
    if reason == "neural_tiebreak":
        return "現AIの候補評価が僅差だったため、棋譜から学習したニューラル候補を参考にしました。"
    if reason == "score_fallback":
        if "attack_tatewari" in detail:
            return "公開された枚数から、受けられにくい攻め駒を優先しました。"
        if "attack_occupancy" in detail:
            return "自分の所持枚数と公開枚数が多い駒を攻めに選びました。"
        if "shallow_eight_card_plan" in detail:
            return "最初の手駒全体から作った、おおまかな攻め順を使いました。"
        if "ally_passed_same_piece_receive" in detail:
            return "相方のパスと自分の受け駒を合わせ、受ける方を選びました。"
        if "pass_hand_strength" in detail:
            return "敵方の序盤の攻めに対し、自分の手駒の強さと受け幅からパスしました。"
        if detail == "pass_enemy_big_piece_weak_followup":
            return "飛・角を受けても有力な攻めが続かないため、パスして様子を見ました。"
        if detail == "receive_enemy_big_piece_ally_signal":
            return "相方へ2回目の攻め情報を伝えるため、飛・角を同じ駒で受けました。"
        if detail.startswith("attack_"):
            return "攻め駒の安全性、残り手駒、次の攻めを総合して選びました。"
        if detail.startswith("pass_"):
            return "受け駒を残す価値と、その後の展開を比較してパスしました。"
        if "receive" in detail:
            return "受けた後の攻めと、手駒に残る受け幅を評価して受けました。"
        return "複数の戦術評価を合計し、最も高い候補を選びました。"
    return "AIの内部判断理由を日本語へ変換できませんでした。"


def _provisional_rating(
    human_route: Sequence[Sequence[object]],
    ai_route: Sequence[Sequence[object]],
    scores: Sequence[Mapping[str, object]],
    search: Optional[Mapping[str, object]],
) -> Tuple[str, str]:
    if list(ai_route) == list(human_route):
        return "理解できる", "手順が一致。理由は人間による確認が必要"
    if ai_route and human_route and list(ai_route[0]) == list(human_route[0]):
        return "少し理解できる", "最初の判断は一致したが、その後の攻めが不一致"

    actual = list(human_route[0]) if human_route else []
    actual_score = next((item for item in scores if item.get("action") == actual), None)
    best_score = float(scores[0]["score"]) if scores else None
    if actual_score is None or best_score is None:
        return "なんともいえない", "AIの候補評価が不足している"
    gap = best_score - float(actual_score["score"])
    margin = float((search or {}).get("margin", 0.0))
    decisive = bool((search or {}).get("decisive", False))
    if int(actual_score.get("rank", 99)) <= 2 and gap <= 75.0:
        return "少し理解できる", "別の手を選んだが、棋譜の手も上位候補"
    if decisive and margin >= 200.0 and gap >= 200.0:
        return "理解できない", "探索が大差で別の手を選び、棋譜の手の評価も低い"
    if int(actual_score.get("rank", 99)) <= max(2, (len(scores) + 1) // 2):
        return "なんともいえない", "別の手だが、棋譜の手は候補の中位以上"
    return "あまり理解できない", "別の手を選び、棋譜の手の候補順位も低い"


def evaluate_unit(
    unit: Mapping[str, object],
    *,
    ai_profile: str = DEFAULT_AI_PROFILE,
) -> Dict[str, Any]:
    case = copy.deepcopy(unit["case"])
    ai_label, agent_class = _resolve_ai_profile(ai_profile)
    agent = agent_class(name=f"player-kifu-understanding-audit-{ai_profile}")
    state = replay_validation_case(case, agent)
    player = str(case["player"])
    legal = list(state.legal_actions(player))
    scores = _action_score_snapshot(agent, state, player, legal)
    started = time.perf_counter()
    ai_root: Action = agent.select_action(state, player, legal)
    ai_reason = str(agent.last_decision_reason or "")
    ai_detail = str(agent.last_score_fallback_detail or "")
    ai_explanation = _explain_ai_reason(ai_reason, ai_detail)
    root_neural_shadow = copy.deepcopy(
        getattr(agent, "last_neural_shadow", None)
    )
    search = copy.deepcopy(agent._track.get(id(state), {}).get("last_time_limited_search"))
    ai_followup = None
    followup_reason = ""
    followup_detail = ""
    followup_neural_shadow = None
    if ai_root[0] == "receive":
        _apply_action(state, player, ai_root)
        agent.on_public_action(state, player, ai_root)
        follow_legal = state.legal_actions(player)
        if follow_legal:
            ai_followup = agent.select_action(state, player, follow_legal)
            followup_reason = str(agent.last_decision_reason or "")
            followup_detail = str(agent.last_score_fallback_detail or "")
            followup_neural_shadow = copy.deepcopy(
                getattr(agent, "last_neural_shadow", None)
            )

    actual_root: Action = tuple(case["actual_action"])  # type: ignore[assignment]
    recorded_followup = unit.get("recorded_followup")
    actual_followup = None
    if isinstance(recorded_followup, Mapping):
        actual_followup = tuple(recorded_followup["actual_action"])
    human_route = _route(actual_root, actual_followup)
    ai_route = _route(ai_root, ai_followup)
    rating, rating_reason = _provisional_rating(
        human_route, ai_route, scores, search if isinstance(search, Mapping) else None
    )
    return {
        "id": str(unit["id"]),
        "stratum": str(unit["stratum"]),
        "seat": player,
        "ai_profile": ai_profile,
        "ai_label": ai_label,
        "source": copy.deepcopy(case["source"]),
        "position": {
            "phase": case["position"]["phase"],
            "hand": list(case["position"]["hand"]),
            "hand_size": int(case["position"]["hand_size"]),
            "current_attack": case["position"]["current_attack"],
            "attacker_relation": case["position"]["attacker_relation"],
            "legal_action_count": len(legal),
            "public_history": _public_history(case),
        },
        "recorded_route": human_route,
        "ai_route": ai_route,
        "ai_reason": ai_reason,
        "ai_detail": ai_detail,
        "ai_explanation": ai_explanation,
        "ai_followup_reason": followup_reason,
        "ai_followup_detail": followup_detail,
        "ai_followup_explanation": (
            _explain_ai_reason(followup_reason, followup_detail)
            if followup_reason or followup_detail else ""
        ),
        "candidate_scores": scores,
        "search": search,
        "neural_shadow": root_neural_shadow,
        "followup_neural_shadow": followup_neural_shadow,
        "provisional_rating": rating,
        "provisional_rating_reason": rating_reason,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "review": _empty_review(),
    }


def build_report(
    kifu_path: Path,
    *,
    player_name: str,
    limit: int,
    ai_profile: str = DEFAULT_AI_PROFILE,
) -> Dict[str, Any]:
    ai_label, _agent_class = _resolve_ai_profile(ai_profile)
    units, source_summary = build_player_decision_units(
        kifu_path, player_name=player_name
    )
    selected = select_balanced_units(units, limit=limit)
    results = []
    for index, unit in enumerate(selected, start=1):
        result = evaluate_unit(unit, ai_profile=ai_profile)
        results.append(result)
        print(
            f"[{index}/{len(selected)}] {result['stratum']} "
            f"{result['provisional_rating']}",
            flush=True,
        )
    ratings = Counter(item["provisional_rating"] for item in results)
    strata = Counter(item["stratum"] for item in results)
    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "private_player_ai_understanding_review",
        "private": True,
        "player": player_name,
        "ai_profile": ai_profile,
        "ai_label": ai_label,
        "rules": {
            "exclude_round_when_any_hand_has_five_or_more_shi": True,
            "forced_decisions_excluded": True,
            "receive_and_immediate_attack_are_one_route": True,
            "ai_moves_before_recorded_move_is_revealed": True,
            "provisional_rating_is_not_proof_of_human_intent": True,
        },
        "source_summary": source_summary,
        "sample": {
            "requested": int(limit),
            "evaluated": len(results),
            "strata": dict(sorted(strata.items())),
            "provisional_ratings": {rating: ratings.get(rating, 0) for rating in RATINGS},
        },
        "cases": results,
    }


def merge_previous_answers(
    report: Mapping[str, object],
    previous_report: Mapping[str, object],
) -> Dict[str, Any]:
    """Carry reviews forward and mark changed AI decisions for rechecking."""
    merged = copy.deepcopy(dict(report))
    previous_cases = {
        str(case.get("id", "")): case
        for case in previous_report.get("cases", [])
        if isinstance(case, Mapping) and case.get("id")
    }
    counts = Counter()
    previous_ai_profile = str(previous_report.get("ai_profile", "current") or "current")
    previous_ai_label = str(previous_report.get("ai_label", "強化中AI") or "強化中AI")
    current_ai_profile = str(merged.get("ai_profile", DEFAULT_AI_PROFILE))

    for case in merged.get("cases", []):
        if not isinstance(case, dict):
            continue
        previous = previous_cases.get(str(case.get("id", "")))
        if not isinstance(previous, Mapping):
            continue
        counts["matched_cases"] += 1

        previous_route = copy.deepcopy(list(previous.get("ai_route", []) or []))
        current_route = list(case.get("ai_route", []) or [])
        route_changed = previous_route != current_route
        reason_fields = (
            "ai_reason",
            "ai_detail",
            "ai_followup_reason",
            "ai_followup_detail",
            "ai_explanation",
            "ai_followup_explanation",
        )
        reason_changed = any(
            str(previous.get(field, "") or "")
            != str(case.get(field, "") or "")
            for field in reason_fields
        )
        decision_changed = route_changed or reason_changed
        if route_changed:
            counts["ai_route_changed"] += 1
        if reason_changed:
            counts["ai_reason_changed"] += 1
        if decision_changed:
            counts["ai_decision_changed"] += 1

        old_review = _normalized_review(previous.get("review"))
        had_review = _has_review(old_review)
        needs_recheck = decision_changed and had_review
        if had_review:
            counts["previously_reviewed"] += 1
            case["previous_review"] = old_review
            if needs_recheck:
                case["review"] = _empty_review()
                counts["reviews_needing_recheck"] += 1
            else:
                case["review"] = old_review
                counts["reviews_carried"] += 1

        case["comparison"] = {
            "route_changed": route_changed,
            "reason_changed": reason_changed,
            "decision_changed": decision_changed,
            "needs_recheck": needs_recheck,
            "previous_ai_route": previous_route,
            "previous_ai_reason": str(previous.get("ai_reason", "") or ""),
            "previous_ai_detail": str(previous.get("ai_detail", "") or ""),
            "previous_provisional_rating": str(
                previous.get("provisional_rating", "") or ""
            ),
            "previous_ai_profile": str(
                previous.get("ai_profile", previous_ai_profile)
                or previous_ai_profile
            ),
            "previous_ai_label": str(
                previous.get("ai_label", previous_ai_label)
                or previous_ai_label
            ),
            "profile_changed": previous_ai_profile != current_ai_profile,
        }

    merged["comparison_summary"] = {
        "previous_cases": len(previous_cases),
        **{
            key: int(counts.get(key, 0))
            for key in (
                "matched_cases",
                "previously_reviewed",
                "reviews_carried",
                "reviews_needing_recheck",
                "ai_route_changed",
                "ai_reason_changed",
                "ai_decision_changed",
            )
        },
    }
    merged["previous_report_generated_at"] = str(
        previous_report.get("generated_at", "") or ""
    )
    merged["previous_ai_profile"] = previous_ai_profile
    merged["previous_ai_label"] = previous_ai_label
    return merged


def _json_for_script(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def render_review_html(report: Mapping[str, object]) -> str:
    data = _json_for_script(report)
    ratings = _json_for_script(RATINGS)
    template = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>1222 棋譜・AI理解度確認</title><style>
body{font-family:system-ui,sans-serif;margin:0;background:#f5f2ea;color:#29261f}header{position:sticky;top:0;background:#fff;padding:16px 5%;border-bottom:1px solid #d8d1c2;z-index:2}main{max-width:1100px;margin:auto;padding:20px}button,select,input,textarea{font:inherit}.toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.case{background:#fff;border:1px solid #d8d1c2;border-radius:12px;padding:18px;margin:16px 0}.case.reviewed{border-color:#4d8469}.meta{color:#6c6558;font-size:.9rem}.routes{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0}.route{background:#f8f7f3;padding:12px;border-radius:8px}.same{color:#17643d;font-weight:700}.different{color:#9b3b31;font-weight:700}.history{max-height:180px;overflow:auto;font-size:.88rem;background:#f8f7f3;padding:8px;border-radius:8px}.review{border-top:1px dashed #bcb4a6;margin-top:14px;padding-top:14px}.rating{display:flex;gap:10px;flex-wrap:wrap}label{display:inline-flex;gap:5px;align-items:center}textarea{width:100%;min-height:70px;box-sizing:border-box;margin-top:8px}.row{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0}.summary{font-weight:700}
.case.ai-changed{border-left:5px solid #c56818}.ai-change-notice{margin:10px 0;padding:9px 11px;border-radius:7px;background:#fff0d9;color:#7a3f0d;font-weight:800}.previous-ai,.previous-review{margin:8px 0;padding:10px 12px;border-radius:7px;background:#f3f0ea;color:#554d42;font-size:.9rem;line-height:1.65}.previous-review{border:1px dashed #9b8972}.previous-review-note{white-space:pre-wrap}
.complete-review-button{display:block;margin:14px 0 0 auto;padding:9px 14px;border:1px solid #4d8469;border-radius:7px;background:#edf6ef;color:#24563d;font-weight:800;cursor:pointer}.complete-review-button:disabled{border-color:#aaa;background:#eee;color:#777;cursor:default}
.board-title{text-align:center;font-weight:800;margin:14px 0 6px}.board-note{text-align:center;color:#6c6558;font-size:.78rem;margin:5px 0 12px}.review-board-wrap{width:fit-content;max-width:100%;margin:0 auto;padding:8px;overflow:hidden;border:3px solid #8b5a2b;background:#d1ab75}.review-board{--cell:min(39px,calc((100vw - 112px)/8));--gap:5px;display:grid;grid-template-columns:repeat(8,var(--cell));grid-template-rows:repeat(8,var(--cell));gap:var(--gap);width:fit-content;container-type:inline-size;user-select:none}.board-cell{position:relative;display:flex;min-width:0;align-items:center;justify-content:center;border:1px dashed rgba(139,90,43,.38)}.board-piece{display:flex;width:80%;height:94%;align-items:center;justify-content:center;background:linear-gradient(135deg,#fceeb5,#e6c875);clip-path:polygon(50% 0%,94% 16%,98% 100%,2% 100%,6% 16%);filter:drop-shadow(1px 2px 2px rgba(0,0,0,.28))}.board-piece span{color:#1f2b9c;font-family:"Yu Kyokasho","游教科書体","YuKyokasho","Hiragino Mincho ProN",serif;font-size:clamp(12px,2.5vw,20px);font-weight:900;line-height:1;transform:translateY(3px)}.board-piece.team-b span{color:#b62020}.board-piece.seat-A{transform:rotate(0)}.board-piece.seat-B{transform:rotate(-90deg)}.board-piece.seat-C{transform:rotate(180deg)}.board-piece.seat-D{transform:rotate(90deg)}.board-piece.is-hand{opacity:.42;filter:none}.board-piece.is-unknown{opacity:.3;filter:none;background:repeating-linear-gradient(45deg,#8c6a43,#8c6a43 4px,#b68a56 4px,#b68a56 8px)}.board-piece.is-face-down{opacity:.48;filter:none}.board-piece.is-current{background:linear-gradient(135deg,#f7e1a0,#f1c40f);outline:3px solid #b74a18;outline-offset:-2px;opacity:1;z-index:3}.board-piece.is-current span{color:#a40000}.move-number{position:absolute;z-index:5;display:grid;min-width:17px;height:17px;place-items:center;padding:0 1px;border:1px solid rgba(17,24,39,.55);border-radius:50%;background:#fff;color:#111827;font-size:10px;font-weight:900;line-height:1}.move-number.seat-A{top:0;left:50%;transform:translate(-50%,-52%)}.move-number.seat-B{top:50%;left:0;transform:translate(-52%,-50%) rotate(-90deg)}.move-number.seat-C{bottom:0;left:50%;transform:translate(-50%,52%) rotate(180deg)}.move-number.seat-D{top:50%;right:0;transform:translate(52%,-50%) rotate(90deg)}.seat-label{z-index:2;display:flex;align-items:center;justify-content:center;border:2px solid #8b5a2b;background:#f4efdf;color:#b00000;font-size:clamp(14px,2.8vw,21px);font-weight:900}.seat-label.is-turn{background:#edf6ef;outline:2px solid #176b57;outline-offset:-4px}.seat-content{display:flex;flex-direction:column;align-items:center;justify-content:center;flex:0 0 auto;width:calc(25cqw - 8px);gap:1px;line-height:1.2}.seat-badge{background:#176b57;color:#fff;border-radius:3px;padding:1px 3px;font-size:9px}.seat-label.seat-A .seat-content{transform:rotate(0)}.seat-label.seat-B .seat-content{transform:rotate(-90deg)}.seat-label.seat-C .seat-content{transform:rotate(180deg)}.seat-label.seat-D .seat-content{transform:rotate(90deg)}.center-info{grid-column:4/6;grid-row:4/6;z-index:3;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:4px;border:2px dashed #8b5a2b;background:rgba(244,239,223,.94);color:#2b2b2b;font-size:clamp(9px,2vw,13px);font-weight:900;line-height:1.35;text-align:center}.center-info strong{display:block}
@media(max-width:650px){.routes{grid-template-columns:1fr}header{position:static}.case{padding:12px}.review-board-wrap{padding:5px;border-width:2px}.review-board{--gap:3px;--cell:min(36px,calc((100vw - 80px)/8))}.board-piece span{font-size:clamp(14px,2.8vw,21px)}.move-number{min-width:14px;height:14px;font-size:8px}}
</style></head><body><header><h1>1222 棋譜・AI理解度確認</h1><div class="toolbar"><select id="filter"><option value="all">すべて</option><option value="different">不一致のみ</option><option value="unreviewed">未確認のみ</option></select><button id="export">確認結果を出力</button><span id="progress"></span></div></header><main><p>5し以上の局と合法手が1つだけの場面は除外しています。AIの分類は暫定です。特に不一致場面を確認してください。</p><div id="summary" class="summary"></div><div id="cases"></div></main>
<script>
const report=__REPORT_DATA__;const ratings=__RATINGS_DATA__;const aiLabel=report.ai_label||'強化中AI';const key='goita-1222-understanding-v1';const saved=JSON.parse(localStorage.getItem(key)||'{}');const piece={'1':'し','2':'香','3':'馬','4':'銀','5':'金','6':'角','7':'飛','8':'玉','9':'王'};
const seats=['A','B','C','D'];
const slots={A:{receive:[[3,7],[4,7],[5,7],[6,7]],attack:[[3,8],[4,8],[5,8],[6,8]]},B:{receive:[[7,6],[7,5],[7,4],[7,3]],attack:[[8,6],[8,5],[8,4],[8,3]]},C:{receive:[[6,2],[5,2],[4,2],[3,2]],attack:[[6,1],[5,1],[4,1],[3,1]]},D:{receive:[[2,3],[2,4],[2,5],[2,6]],attack:[[1,3],[1,4],[1,5],[1,6]]}};
const labels={A:{column:'4 / 6',row:'6'},B:{column:'6',row:'4 / 6'},C:{column:'4 / 6',row:'3'},D:{column:'3',row:'4 / 6'}};
const skipped=new Set(['5,6','6,5','5,3','3,5']);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function action(a){if(!a)return'';if(a[0]==='pass')return'パス';if(a[0]==='receive')return `${piece[a[1]]}で受ける`;if(a[0]==='attack')return `${piece[a[2]]}で攻める`;return `${a[1]?piece[a[1]]:'伏せ駒'}を伏せる → ${piece[a[2]]}で攻める`;}
function route(r){return r.map(action).join(' → ');}function current(c){return Object.assign({},c.review,saved[c.id]||{});}
function store(id,field,value){saved[id]=Object.assign({},saved[id]||{},{[field]:value});localStorage.setItem(key,JSON.stringify(saved));render();}
function boardState(c){
  const board=Object.fromEntries(seats.map(s=>[s,{receive:[],attack:[]}]));const used=Object.fromEntries(seats.map(s=>[s,0]));let attackNo=0;let currentKey='';
  for(const item of c.position.public_history||[]){const s=item.player;const a=item.action||[];if(!board[s])continue;
    if(a[0]==='receive'){board[s].receive.push({piece:a[1],faceDown:false});used[s]++;}
    else if(a[0]==='attack'){attackNo++;board[s].attack.push({piece:a[2],number:attackNo});used[s]++;currentKey=`${s}:attack:${board[s].attack.length-1}`;}
    else if(a[0]==='attack_after_block'){board[s].receive.push({piece:a[1]||'',faceDown:true});used[s]++;attackNo++;board[s].attack.push({piece:a[2],number:attackNo});used[s]++;currentKey=`${s}:attack:${board[s].attack.length-1}`;}
  }
  return {board,used,currentKey};
}
function makePiece(value,seat,opts={}){const el=document.createElement('div');el.className=`board-piece seat-${seat} ${seat==='B'||seat==='D'?'team-b':'team-a'}`;if(opts.hand)el.classList.add('is-hand');if(opts.unknown)el.classList.add('is-unknown');if(opts.faceDown)el.classList.add('is-face-down');if(opts.current)el.classList.add('is-current');const span=document.createElement('span');span.textContent=opts.unknown?'':(piece[value]||'');el.appendChild(span);return el;}
function renderBoard(container,c){container.replaceChildren();const state=boardState(c);const slotMap=new Map();const concealed=new Map();
  for(const s of seats){for(const kind of ['receive','attack'])slots[s][kind].forEach((pos,i)=>slotMap.set(`${pos[0]},${pos[1]}`,{seat:s,kind,index:i}));const open=[];for(const kind of ['receive','attack'])slots[s][kind].forEach((pos,i)=>{if(!state.board[s][kind][i])open.push(pos);});const values=s===c.seat?[...(c.position.hand||[])]:Array(Math.max(0,8-state.used[s])).fill(null);open.forEach((pos,i)=>{if(i<values.length)concealed.set(`${pos[0]},${pos[1]}`,{seat:s,value:values[i],unknown:s!==c.seat});});}
  for(let row=1;row<=8;row++){for(let column=1;column<=8;column++){const key=`${column},${row}`;if(skipped.has(key))continue;const labelSeat=seats.find(s=>labels[s].column.startsWith(String(column))&&labels[s].row.startsWith(String(row)));if(labelSeat){const label=document.createElement('div');label.className=`seat-label seat-${labelSeat}${labelSeat===c.seat?' is-turn':''}`;label.style.gridColumn=labels[labelSeat].column;label.style.gridRow=labels[labelSeat].row;const content=document.createElement('div');content.className='seat-content';const name=document.createElement('span');name.textContent=labelSeat+(labelSeat===c.seat?' 1222':'');content.appendChild(name);if(labelSeat===c.seat){const badge=document.createElement('span');badge.className='seat-badge';badge.textContent='判断する席';content.appendChild(badge);}label.appendChild(content);container.appendChild(label);continue;}
      const cell=document.createElement('div');cell.className='board-cell';cell.style.gridColumn=String(column);cell.style.gridRow=String(row);const slot=slotMap.get(key);if(slot){const item=state.board[slot.seat][slot.kind][slot.index];if(item){const itemKey=`${slot.seat}:${slot.kind}:${slot.index}`;cell.appendChild(makePiece(item.piece,slot.seat,{faceDown:item.faceDown,current:itemKey===state.currentKey&&Boolean(c.position.current_attack)}));if(slot.kind==='attack'){const n=document.createElement('span');n.className=`move-number seat-${slot.seat}`;n.textContent=String(item.number);cell.appendChild(n);}}else if(concealed.has(key)){const hidden=concealed.get(key);cell.appendChild(makePiece(hidden.value,hidden.seat,{hand:!hidden.unknown,unknown:hidden.unknown}));}}container.appendChild(cell);}}
  const center=document.createElement('div');center.className='center-info';const title=document.createElement('strong');title.textContent='判断直前';const turn=document.createElement('span');turn.textContent=`${c.seat}の手番`;const attack=document.createElement('span');attack.textContent=`場の攻め ${c.position.current_attack?(piece[c.position.current_attack]||c.position.current_attack):'なし'}`;center.append(title,turn,attack);container.appendChild(center);
}
function render(){const filter=document.getElementById('filter').value;let shown=0,done=0;const root=document.getElementById('cases');root.innerHTML='';for(const c of report.cases){const rv=current(c);if(rv.final_rating)done++;const same=JSON.stringify(c.recorded_route)===JSON.stringify(c.ai_route);if(filter==='different'&&same)continue;if(filter==='unreviewed'&&rv.final_rating)continue;shown++;const el=document.createElement('section');el.className='case '+(rv.final_rating?'reviewed':'');const hist=c.position.public_history.map(x=>`${x.player}：${action(x.action)}`).join('<br>');const cand=(c.candidate_scores||[]).map(x=>`${x.rank}位：${action(x.action)}（評価 ${x.score}）`).join('<br>');const attack=c.position.current_attack?piece[c.position.current_attack]:'なし';el.innerHTML=`<div class="meta">${esc(c.stratum)}／${esc(c.seat)}席／場の攻め ${attack}／手駒 ${c.position.hand.map(x=>piece[x]).join(' ')}</div><div class="board-title">判断直前の盤面</div><div class="review-board-wrap"><div class="review-board" data-board aria-label="判断直前の盤面"></div></div><div class="board-note">薄い駒は1222の手駒、模様だけの駒は他家の非公開手駒です。黄色の駒が現在の攻めです。</div><div class="routes"><div class="route"><b>1222の棋譜</b><br>${esc(route(c.recorded_route))}</div><div class="route"><b>${esc(aiLabel)}</b><br>${esc(route(c.ai_route))}<br><span class="${same?'same':'different'}">${same?'手順一致':'手順不一致'}</span></div></div><p><b>AIの暫定分類：</b>${esc(c.provisional_rating)}（${esc(c.provisional_rating_reason)}）</p><p><b>AIの説明：</b>${esc(c.ai_explanation||'')}${c.ai_followup_explanation?'<br>'+esc(c.ai_followup_explanation):''}</p><details><summary>内部の判断記録</summary>${esc(c.ai_reason)}／${esc(c.ai_detail)}${c.ai_followup_reason?'<br>'+esc(c.ai_followup_reason)+'／'+esc(c.ai_followup_detail):''}</details><details><summary>AIの候補評価</summary><div class="history">${cand||'候補別の評価はありません'}</div></details><details><summary>ここまでの公開履歴</summary><div class="history">${hist||'初手'}</div></details><div class="review"><b>最終的な理解度</b><div class="rating">${ratings.map(x=>`<label><input type="radio" name="r-${c.id}" value="${x}" ${rv.final_rating===x?'checked':''}>${x}</label>`).join('')}</div><div class="row"><label>棋譜の手 <select data-field="recorded_move_quality"><option value="">選択</option>${['良い手','迷う手','ミスだった','判断できない'].map(x=>`<option ${rv.recorded_move_quality===x?'selected':''}>${x}</option>`).join('')}</select></label><label>主な目的 <select data-field="purpose"><option value="">選択</option>${['自分の上がり','敵方に王・玉を使わせる','相方の上がりを助ける','し攻め・連続攻め','受け駒・攻め駒の温存','相方への情報','その他','覚えていない'].map(x=>`<option ${rv.purpose===x?'selected':''}>${x}</option>`).join('')}</select></label><label>AIの説明 <select data-field="ai_explanation_close"><option value="">選択</option>${['近い','一部近い','違う','判断できない'].map(x=>`<option ${rv.ai_explanation_close===x?'selected':''}>${x}</option>`).join('')}</select></label></div><textarea placeholder="考えていたこと（任意）">${esc(rv.note||'')}</textarea></div>`;renderBoard(el.querySelector('[data-board]'),c);el.querySelectorAll('input[type=radio]').forEach(x=>x.onchange=()=>store(c.id,'final_rating',x.value));el.querySelectorAll('select[data-field]').forEach(x=>x.onchange=()=>store(c.id,x.dataset.field,x.value));el.querySelector('textarea').onchange=e=>store(c.id,'note',e.target.value);root.appendChild(el);}document.getElementById('progress').textContent=`確認済み ${done} / ${report.cases.length}`;document.getElementById('summary').textContent=`表示 ${shown}件`;}
document.getElementById('filter').onchange=render;document.getElementById('export').onclick=()=>{const out=structuredClone(report);for(const c of out.cases)c.review=current(c);const blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='1222_ai_understanding_review_answered.json';a.click();URL.revokeObjectURL(a.href);};render();
</script></body></html>"""
    page = template.replace("__REPORT_DATA__", data).replace("__RATINGS_DATA__", ratings)
    page = page.replace(
        '<option value="unreviewed">未確認のみ</option>',
        '<option value="changed">AI判断更新のみ</option><option value="recheck">再確認のみ</option><option value="unreviewed">未確認のみ</option>',
        1,
    )
    page = page.replace(
        "const key='goita-1222-understanding-v1';",
        "const key=`goita-1222-understanding-v5-${report.ai_profile||'current'}`;",
        1,
    )
    page = page.replace(
        "function store(id,field,value){saved[id]=Object.assign({},saved[id]||{},{[field]:value});localStorage.setItem(key,JSON.stringify(saved));render();}",
        "function store(id,field,value){saved[id]=Object.assign({},saved[id]||{},{[field]:value});localStorage.setItem(key,JSON.stringify(saved));render();}function completeReview(id){const c=report.cases.find(item=>item.id===id);if(!c)return;const rv=current(c);if(!rv.final_rating){alert('最終的な理解度を選択してください。');return;}saved[id]=Object.assign({},saved[id]||{},{review_completed_after_update:true});localStorage.setItem(key,JSON.stringify(saved));render();}",
        1,
    )
    page = page.replace(
        "function render(){const filter=document.getElementById('filter').value;let shown=0,done=0;",
        "function render(){const filter=document.getElementById('filter').value;let shown=0,done=0,changedCount=0,recheckCount=0;",
        1,
    )
    page = page.replace(
        "const same=JSON.stringify(c.recorded_route)===JSON.stringify(c.ai_route);if(filter==='different'&&same)continue;if(filter==='unreviewed'&&rv.final_rating)continue;shown++;",
        "const same=JSON.stringify(c.recorded_route)===JSON.stringify(c.ai_route);const comparison=c.comparison||{};const changed=Boolean(comparison.decision_changed);const completed=Boolean(c.review?.final_rating)||Boolean(saved[c.id]?.review_completed_after_update);const needsRecheck=Boolean(comparison.needs_recheck)&&!completed;if(completed)done++;if(changed)changedCount++;if(needsRecheck)recheckCount++;if(filter==='different'&&same)continue;if(filter==='changed'&&!changed)continue;if(filter==='recheck'&&!needsRecheck)continue;if(filter==='unreviewed'&&completed)continue;shown++;",
        1,
    )
    page = page.replace(
        "const rv=current(c);if(rv.final_rating)done++;const same=",
        "const rv=current(c);const same=",
        1,
    )
    page = page.replace(
        "el.className='case '+(rv.final_rating?'reviewed':'');",
        "el.className='case '+(rv.final_rating?'reviewed ':'')+(changed?'ai-changed':'');",
        1,
    )
    page = page.replace(
        "const attack=c.position.current_attack?piece[c.position.current_attack]:'なし';el.innerHTML=`",
        "const attack=c.position.current_attack?piece[c.position.current_attack]:'なし';const changedKind=comparison.route_changed?'AIの手順が変わりました':'AIの判断理由が変わりました';const changeNotice=changed?`<div class=\"ai-change-notice\">${esc(changedKind)}${needsRecheck?'。前回の回答を確認して、もう一度評価してください。':''}</div>`:'';const previousAiLabel=comparison.previous_ai_label||'強化中AI';const previousAi=changed&&comparison.previous_ai_route?`<div class=\"previous-ai\"><b>修正前の${esc(previousAiLabel)}：</b>${esc(route(comparison.previous_ai_route))}<br><b>修正前の暫定分類：</b>${esc(comparison.previous_provisional_rating||'')}</div>`:'';const old=c.previous_review||{};const previousReview=needsRecheck?`<div class=\"previous-review\"><b>前回の確認内容（保存済み）</b><br>理解度：${esc(old.final_rating||'未選択')}／棋譜の手：${esc(old.recorded_move_quality||'未選択')}／目的：${esc(old.purpose||'未選択')}／AIの説明：${esc(old.ai_explanation_close||'未選択')}${old.note?`<div class=\"previous-review-note\">${esc(old.note)}</div>`:''}</div>`:'';const changeBlock=changeNotice+previousAi+previousReview;el.innerHTML=`",
        1,
    )
    page = page.replace(
        "</div><div class=\"board-title\">判断直前の盤面</div>",
        "</div>${changeBlock}<div class=\"board-title\">判断直前の盤面</div>",
        1,
    )
    page = page.replace(
        "</textarea></div>`;renderBoard",
        "</textarea><button class=\"complete-review-button\" type=\"button\" data-complete-review ${completed?'disabled':''}>${completed?'確認済み':'この場面の確認を完了'}</button></div>`;renderBoard",
        1,
    )
    page = page.replace(
        "el.querySelector('textarea').onchange=e=>store(c.id,'note',e.target.value);root.appendChild(el);",
        "el.querySelector('textarea').onchange=e=>store(c.id,'note',e.target.value);el.querySelector('[data-complete-review]').onclick=()=>completeReview(c.id);root.appendChild(el);",
        1,
    )
    page = page.replace(
        "document.getElementById('progress').textContent=`確認済み ${done} / ${report.cases.length}`;document.getElementById('summary').textContent=`表示 ${shown}件`;}",
        "document.getElementById('progress').textContent=`確認済み ${done} / ${report.cases.length}　再確認 ${recheckCount}`;document.getElementById('summary').textContent=`対象 ${aiLabel}／表示 ${shown}件／AI判断更新 ${changedCount}件`;}",
        1,
    )
    page = page.replace(
        "document.getElementById('filter').onchange=render;",
        "const initialRecheckCount=report.cases.filter(c=>Boolean(c.comparison?.needs_recheck)&&!Boolean(c.review?.final_rating)&&!Boolean(saved[c.id]?.review_completed_after_update)).length;if(initialRecheckCount)document.getElementById('filter').value='recheck';document.getElementById('filter').onchange=render;",
        1,
    )
    return page


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kifu", type=Path, default=DEFAULT_KIFU_PATH)
    parser.add_argument("--player", default="1222")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--ai-profile",
        choices=tuple(sorted(AI_PROFILES)),
        default=DEFAULT_AI_PROFILE,
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--html-output", type=Path, default=DEFAULT_HTML_PATH)
    parser.add_argument("--previous-answers", type=Path)
    args = parser.parse_args(argv)
    report = build_report(
        args.kifu,
        player_name=args.player,
        limit=args.limit,
        ai_profile=args.ai_profile,
    )
    if args.previous_answers:
        previous_report = json.loads(args.previous_answers.read_text(encoding="utf-8"))
        report = merge_previous_answers(report, previous_report)
    write_json(args.json_output, report)
    args.html_output.parent.mkdir(parents=True, exist_ok=True)
    args.html_output.write_text(render_review_html(report), encoding="utf-8")
    print(f"JSON: {args.json_output}")
    print(f"HTML: {args.html_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
