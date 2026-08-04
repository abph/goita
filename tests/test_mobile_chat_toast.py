from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_desktop_chat_toast_shows_up_to_three_lines() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    base_start = html.index("    .chat-toast-text {")
    base_end = html.index("    @media (min-width: 702px)", base_start)
    base_css = html[base_start:base_end]

    assert "white-space: normal" in base_css
    assert "-webkit-line-clamp: 3" in base_css
    assert "max-height: calc(1.45em * 3)" in base_css


def test_mobile_chat_toast_shows_up_to_three_lines() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    mobile_start = html.index("@media (max-width: 701px)")
    mobile_end = html.index("</style>", mobile_start)
    mobile_css = html[mobile_start:mobile_end]

    assert ".chat-toast-text" in mobile_css
    assert "white-space: normal" in mobile_css
    assert "-webkit-line-clamp: 3" in mobile_css
    assert "max-height: calc(1.45em * 3)" in mobile_css
