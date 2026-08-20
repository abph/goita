from pathlib import Path


ROOT = Path(__file__).parents[1]
HTML = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
I18N_EN = (ROOT / "frontend" / "i18n-en.js").read_text(encoding="utf-8")
I18N_ZH = (ROOT / "frontend" / "i18n.js").read_text(encoding="utf-8")


def test_auto_hand_reveal_is_an_opt_in_personal_setting() -> None:
    for expected in (
        'id="lobbyCheckAutoRevealOwnAndAiHands"',
        'id="checkAutoRevealOwnAndAiHands"',
        "autoRevealOwnAndAiHands: saved.autoRevealOwnAndAiHands === true",
        "autoRevealOwnAndAiHands: false",
        "personalSettings.autoRevealOwnAndAiHands !== true",
        "function maybeAutoRevealOwnAndAiHands(state)",
        "const targets = [mySeat, ...aiSeats]",
        "await requestSeatHandReveal(target, false)",
    ):
        assert expected in HTML


def test_auto_hand_reveal_label_is_localized_without_extra_notice() -> None:
    label = "終局後に自分とAIの手札を自動で公開する"
    assert label in HTML
    assert label in I18N_EN
    assert label in I18N_ZH
    assert "公開した手札は参加者全員に表示されます。" not in HTML


if __name__ == "__main__":
    test_auto_hand_reveal_is_an_opt_in_personal_setting()
    test_auto_hand_reveal_label_is_localized_without_extra_notice()
    print("AUTO_HAND_REVEAL_TEST_OK")
