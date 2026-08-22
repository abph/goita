from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
ZH = (ROOT / "frontend" / "i18n.js").read_text(encoding="utf-8")
EN = (ROOT / "frontend" / "i18n-en.js").read_text(encoding="utf-8")


def test_notice_visibility_setting_is_available_in_lobby_and_room() -> None:
    assert 'id="lobbyChatNoticeVisibility"' in HTML
    assert 'id="chatNoticeVisibility"' in HTML
    assert HTML.count('<option value="important" selected>重要な連絡のみ（おすすめ）</option>') == 2
    assert HTML.count('<option value="all">すべて表示</option>') == 2
    assert HTML.count('<option value="none">表示しない</option>') == 2


def test_notice_visibility_defaults_to_important_and_is_persisted() -> None:
    assert 'function normalizeChatNoticeVisibility(value)' in HTML
    assert 'chatNoticeVisibility: normalizeChatNoticeVisibility(saved.chatNoticeVisibility)' in HTML
    assert 'chatNoticeVisibility: "important"' in HTML
    assert 'document.getElementById("lobbyChatNoticeVisibility").value' in HTML
    assert 'document.getElementById("chatNoticeVisibility").value' in HTML


def test_chat_filters_all_notices_and_hides_disabled_toasts() -> None:
    assert 'function hintNoticeImportance(text)' in HTML
    assert 'function shouldDisplayChatNotice(item)' in HTML
    assert 'item?.local_notice === true || item?.seat === "notice"' in HTML
    assert 'if(visibility === "all") return true;' in HTML
    assert 'if(visibility === "none") return false;' in HTML
    assert 'item?.notice_importance || hintNoticeImportance(item?.message)' in HTML
    assert '...serverMessages.filter(shouldDisplayChatNotice)' in HTML
    assert '...localChatNotices.filter(shouldDisplayChatNotice)' in HTML
    assert '...lobbyChatNotices.filter(shouldDisplayChatNotice)' in HTML
    assert 'if(!shouldDisplayChatNotice(item)) return;' in HTML
    assert 'hideChatToast();' in HTML


def test_notice_visibility_labels_are_translated() -> None:
    for source in (ZH, EN):
        assert '"チャット内の連絡表示"' in source
        assert '"重要な連絡のみ（おすすめ）"' in source
        assert '"すべて表示"' in source
        assert '"表示しない"' in source


if __name__ == "__main__":
    test_notice_visibility_setting_is_available_in_lobby_and_room()
    test_notice_visibility_defaults_to_important_and_is_persisted()
    test_chat_filters_all_notices_and_hides_disabled_toasts()
    test_notice_visibility_labels_are_translated()
    print("CHAT_NOTICE_VISIBILITY_TEST_OK")
