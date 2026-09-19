from pathlib import Path


ROOT = Path(__file__).parents[1]
HTML = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
SCRIPT = (ROOT / "frontend" / "kifu-files.js").read_text(encoding="utf-8")


def test_how_to_button_is_directly_below_score_attack_start():
    start = HTML.index('id="debugTraceRandomButton"')
    how_to = HTML.index('id="scoreAttackHowToButton"')
    history = HTML.index("挑戦履歴・ランキング", how_to)
    assert start < how_to < history
    assert 'onclick="openScoreAttackHowTo()"' in HTML


def test_how_to_modal_explains_score_attack_and_returns_to_menu():
    assert 'id="scoreAttackHowToModal"' in HTML
    for text in [
        "元の棋譜と同じ手駒・同じ親で対局します。",
        "あなたがA席を担当し、B・C・D席はAIが担当します。",
        "対局中は元の棋譜に縛られず、自由に手を選べます。",
        "対局結果と元の棋譜の得点を比べ、その差がスコアになります。",
        "スコアはデイリー・ウィークリーランキングに反映されます。",
        "会員は、デイリー1位やウィークリー上位に入ると報酬を獲得できます。",
    ]:
        assert text in HTML
    assert "function openScoreAttackHowTo()" in SCRIPT
    assert "function closeScoreAttackHowTo()" in SCRIPT
    assert "openDebugTrace();" in SCRIPT
