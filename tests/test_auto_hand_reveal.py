from pathlib import Path


ROOT = Path(__file__).parents[1]
HTML = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
I18N_EN = (ROOT / "frontend" / "i18n-en.js").read_text(encoding="utf-8")
I18N_ZH = (ROOT / "frontend" / "i18n.js").read_text(encoding="utf-8")


def test_auto_hand_reveal_has_separate_personal_settings() -> None:
    for expected in (
        'id="lobbyCheckAutoRevealOwnHand"',
        'id="lobbyCheckAutoRevealAiHands"',
        'id="checkAutoRevealOwnHand"',
        'id="checkAutoRevealAiHands"',
        "autoRevealOwnHand: false",
        "autoRevealAiHands: true",
        "saved.autoRevealOwnAndAiHands === true",
        "personalSettings.autoRevealOwnHand !== true && personalSettings.autoRevealAiHands === false",
        "function maybeAutoRevealOwnAndAiHands(state)",
        "personalSettings.autoRevealOwnHand === true ? [mySeat] : []",
        "personalSettings.autoRevealAiHands !== false ? aiSeats : []",
        "await requestSeatHandReveal(target, false, true, true)",
    ):
        assert expected in HTML


def test_auto_hand_reveal_label_is_localized_without_extra_notice() -> None:
    labels = (
        "終局後に自分の手札を自動で公開する",
        "終局後にAIの手札を自動で公開する",
    )
    for label in labels:
        assert label in HTML
        assert label in I18N_EN
        assert label in I18N_ZH
    assert "公開した手札は参加者全員に表示されます。" not in HTML


if __name__ == "__main__":
    test_auto_hand_reveal_has_separate_personal_settings()
    test_auto_hand_reveal_label_is_localized_without_extra_notice()
    print("AUTO_HAND_REVEAL_TEST_OK")
