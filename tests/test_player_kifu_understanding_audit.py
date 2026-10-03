from __future__ import annotations

import json

from goita_ai2.player_kifu_understanding_audit import (
    DEFAULT_AI_PROFILE,
    _explain_ai_reason,
    _provisional_rating,
    _resolve_ai_profile,
    build_player_decision_units,
    merge_previous_answers,
    render_review_html,
    select_balanced_units,
)


def test_understanding_audit_defaults_to_experimental_ai2() -> None:
    assert DEFAULT_AI_PROFILE == "experimental_ai2"
    ai2_label, ai2_class = _resolve_ai_profile(DEFAULT_AI_PROFILE)
    current_label, current_class = _resolve_ai_profile("current")
    assert ai2_label == "強化中AI2"
    assert current_label == "強化中AI"
    assert ai2_class is not current_class


def _archive():
    return {
        "schema_version": 1,
        "match_count": 1,
        "matches": [{
            "id": "sample",
            "players": {"p0": "1222", "p1": "B", "p2": "C", "p3": "D"},
            "rounds": [
                {
                    "round_index": 1,
                    "hand": {
                        "p0": "しし香馬銀金角飛",
                        "p1": "ししし香馬銀金王",
                        "p2": "しし香香馬銀金玉",
                        "p3": "しし香馬馬銀金角",
                    },
                    "uchidashi": 0,
                    "score": [0, 0],
                    "game": [["0", "し", "飛"]],
                },
                {
                    "round_index": 2,
                    "hand": {
                        "p0": "しし香馬銀金角飛",
                        "p1": "ししししし香馬王",
                        "p2": "しし香香馬銀金玉",
                        "p3": "しし香馬馬銀金角",
                    },
                    "uchidashi": 0,
                    "score": [0, 0],
                    "game": [["0", "し", "飛"]],
                },
            ],
        }],
    }


def test_player_units_exclude_every_round_with_any_five_shi(tmp_path) -> None:
    path = tmp_path / "kifu.json"
    path.write_text(json.dumps(_archive(), ensure_ascii=False), encoding="utf-8")

    units, summary = build_player_decision_units(path, player_name="1222")

    assert summary["excluded_five_shi_rounds"] == 1
    assert summary["eligible_decision_units"] == 1
    assert units[0]["case"]["source"]["round_index"] == 1


def test_balanced_selection_is_deterministic() -> None:
    units = [
        {"id": f"{kind}-{index}", "stratum": kind, "case": {}}
        for kind in ("opening_attack", "opening_response", "middle_attack")
        for index in range(5)
    ]
    first = select_balanced_units(units, limit=6)
    second = select_balanced_units(list(reversed(units)), limit=6)
    assert [item["id"] for item in first] == [item["id"] for item in second]
    assert set(item["stratum"] for item in first) == {
        "opening_attack", "opening_response", "middle_attack"
    }


def test_provisional_rating_separates_route_match_from_reason_review() -> None:
    route = [["pass", None, None]]
    rating, reason = _provisional_rating(route, route, [], None)
    assert rating == "理解できる"
    assert "人間による確認" in reason


def test_review_html_has_filters_and_export() -> None:
    report = {
        "cases": [],
        "sample": {},
        "source_summary": {},
        "ai_profile": "experimental_ai2",
        "ai_label": "強化中AI2",
    }
    page = render_review_html(report)
    assert "不一致のみ" in page
    assert "AI判断更新のみ" in page
    assert "再確認のみ" in page
    assert "goita-1222-understanding-v5-${report.ai_profile" in page
    assert "const aiLabel=report.ai_label" in page
    assert "initialRecheckCount" in page
    assert "value='recheck'" in page
    assert "review_completed_after_update:true" in page
    assert "この場面の確認を完了" in page
    assert "最終的な理解度を選択してください。" in page
    assert "const completed=Boolean(c.review?.final_rating)||Boolean(saved[c.id]?.review_completed_after_update)" in page
    assert "Boolean(comparison.needs_recheck)&&!completed" in page
    assert "確認結果を出力" in page
    assert "5し以上" in page
    assert "判断直前の盤面" in page
    assert "renderBoard" in page
    assert "棋譜再生" not in page


def test_previous_reviews_are_carried_or_marked_for_recheck() -> None:
    current = {
        "schema_version": 2,
        "generated_at": "new",
        "cases": [
            {
                "id": "same",
                "ai_route": [["pass", None, None]],
                "ai_reason": "score_fallback",
                "ai_detail": "pass_safe",
                "ai_followup_reason": "",
                "ai_followup_detail": "",
                "ai_explanation": "同じ説明",
                "ai_followup_explanation": "",
                "review": {},
            },
            {
                "id": "changed",
                "ai_route": [["receive", "3", None], ["attack", None, "1"]],
                "ai_reason": "score_fallback",
                "ai_detail": "receive_new",
                "ai_followup_reason": "score_fallback",
                "ai_followup_detail": "attack_new",
                "ai_explanation": "新しい説明",
                "ai_followup_explanation": "新しい続き",
                "review": {},
            },
        ],
    }
    answered = {
        "generated_at": "old",
        "cases": [
            {
                "id": "same",
                "ai_route": [["pass", None, None]],
                "ai_reason": "score_fallback",
                "ai_detail": "pass_safe",
                "ai_followup_reason": "",
                "ai_followup_detail": "",
                "ai_explanation": "同じ説明",
                "ai_followup_explanation": "",
                "review": {"final_rating": "理解できる", "note": "維持"},
            },
            {
                "id": "changed",
                "ai_route": [["receive", "3", None], ["attack", None, "5"]],
                "ai_reason": "score_fallback",
                "ai_detail": "receive_old",
                "ai_followup_reason": "score_fallback",
                "ai_followup_detail": "attack_old",
                "ai_explanation": "以前の説明",
                "ai_followup_explanation": "以前の続き",
                "provisional_rating": "あまり理解できない",
                "review": {"final_rating": "理解できない", "note": "再確認用"},
            },
        ],
    }

    merged = merge_previous_answers(current, answered)

    same, changed = merged["cases"]
    assert same["review"]["final_rating"] == "理解できる"
    assert same["comparison"]["needs_recheck"] is False
    assert changed["review"]["final_rating"] == ""
    assert changed["previous_review"]["note"] == "再確認用"
    assert changed["comparison"]["route_changed"] is True
    assert changed["comparison"]["needs_recheck"] is True
    assert merged["comparison_summary"]["reviews_carried"] == 1
    assert merged["comparison_summary"]["reviews_needing_recheck"] == 1


def test_internal_reason_has_review_facing_explanation() -> None:
    assert "手駒全体" in _explain_ai_reason(
        "score_fallback", "attack_shallow_eight_card_plan_30"
    )
    assert "ニューラル候補" in _explain_ai_reason("neural_tiebreak", "")
