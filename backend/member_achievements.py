"""Personal achievement definitions and rules for completed games."""

from collections import Counter


GROUPS = (
    ("はじめて・遊び方", (
        ("tutorial_basic", "基本チュートリアルを終えた"),
        ("tutorial_royal", "王（玉）で受ける練習を終えた"),
        ("tutorial_attack", "チュートリアルで「同じ駒で攻める」を終えた"),
        ("tutorial_cooperation", "チュートリアルで「相方と協力する」を終えた"),
        ("beginner_round", "初心者サポートを使って1局打った"),
        ("ai_round", "AIと1局打ち切った"),
        ("first_win", "初めて自分が上がった"),
        ("first_match", "150点に到達するまで1ゲーム打った"),
        ("both_public_modes", "「AIとごいた」と「みんなでごいた」の両方で対局した"),
        ("spectate_round", "他のプレイヤーの対局を1局観戦した"),
    )),
    ("打ち方・上がり方", (
        ("kakarigotae", "かかりごたえをした"),
        ("same_three", "同じ駒で3回続けて攻めた"),
        ("different_three", "3回の攻めに、それぞれ違う駒を出した"),
        ("royal_second_third", "2回目と3回目の攻めに、王と玉を1枚ずつ出した"),
        ("third_round_trip", "自分の3回目の攻めが一巡し、そのまま上がった"),
        ("receive_ready_win", "リーチしている敵方の攻めを受け、そのまま自分が上がった"),
        ("no_royal_win", "王も玉もない配牌で上がった"),
        ("double_piece_win", "同じ駒2枚で上がった"),
        ("royal_fifty", "王（玉）で50点以上の上がりを決めた"),
        ("royal_hundred", "王と玉で100点上がりを決めた"),
    )),
    ("みんなと対局", (
        ("public_human", "公開部屋で人と対局した"),
        ("human_partner_match", "人とペアを組んで1ゲームを終えた"),
        ("four_human_match", "人間4人で1ゲーム打った"),
        ("both_partners_won", "同じゲームで、自分と相方がそれぞれ上がった"),
        ("partner_match_win", "相方の上がりでゲームに勝った"),
        ("overcome_hundred", "100点以上を先に取った敵方に逆転勝ちした"),
        ("close_match_win", "最終点差10点以内で勝った"),
        ("shutout", "敵方を0点に抑え、自分たちが150点以上を取って勝った"),
        ("three_partners", "3人の異なる相方と、それぞれ1ゲーム打った"),
        ("pair_practice", "ペア練習で、相方と一緒に1局打ち切った"),
    )),
    ("練習・研究", (
        ("different_deal", "配牌方式を変えて練習した"),
        ("frequent_deal", "「よくある手駒」で1局練習した"),
        ("balanced_deal", "「均衡配碑」で1局練習した"),
        ("wardna_deal", "「均衡配碑（wardna式）」で1局練習した"),
        ("preset_deal", "手札を指定して1局打った"),
        ("rough_deal", "「大雑把に指定」で、枚数の範囲を決めて打った"),
        ("replay_round", "「もう一度練習する」で同じ配牌を打ち直した"),
        ("change_first_attack", "同じ配牌で、最初の攻め駒を変えて打ち直した"),
        ("replay_scene", "「場面を指定して練習する」で気になる一手から打ち直した"),
        ("compare_scene", "同じ場面から、「受ける」と「パスする」の両方を試した"),
    )),
    ("棋譜・振り返り", (
        ("save_kifu", "初めて棋譜を保存した"),
        ("auto_save_kifu", "棋譜の自動保存で、初めて1局保存された"),
        ("replay_kifu", "保存した棋譜を再生した"),
        ("favorite_kifu", "気に入った棋譜に☆を付けた"),
        ("tag_kifu", "棋譜に自分でタグを付けた"),
        ("stats_kifu", "棋譜の統計を見た"),
        ("analyze_kifu", "「分析する」で自分の対局を振り返った"),
        ("import_kifu", "棋譜ファイルを読み込んだ"),
        ("image_kifu", "棋譜の盤面を画像で保存した"),
        ("share_kifu", "プライベートルームで棋譜をみんなに見せた"),
    )),
    ("スコアアタック", (
        ("score_first", "初めてスコアアタックを完了した"),
        ("score_improved", "元の棋譜より良い点数を取った"),
        ("score_plus_30", "元の棋譜を30点以上上回った"),
        ("score_exact", "元の棋譜と同じ手順で打ち切った"),
        ("score_divergent", "結果が分かれた棋譜でプラスを出した"),
        ("score_both_winners", "自分の上がりと相方の上がりの両方でプラスを出した"),
        ("score_three_kifu", "3種類の棋譜でプラスを出した"),
        ("score_streak_3", "3局連続でプラスを出した"),
        ("score_daily_top3", "デイリーランキングで3位以内に入った"),
        ("score_daily_first", "デイリーランキングで1位になった"),
        ("score_weekly_top3", "ウィークリーランキングで3位以内に入った"),
        ("score_weekly_first", "ウィークリーランキングで1位になった"),
        ("score_day_50", "1日で50局スコアアタックを完了した"),
        ("score_day_100", "1日で100局スコアアタックを完了した"),
        ("score_streak_10", "10局連続でプラスを出した"),
    )),
)

TITLES = {
    "score_weekly_top3": "週間入賞", "score_weekly_first": "週間王者",
    "score_day_50": "ごいたジャンキー", "score_day_100": "ごいた命", "score_streak_10": "ごいた無双",
}
REWARDS = {"score_weekly_top3": 0, "score_weekly_first": 5, "score_day_50": 5,
           "score_day_100": 5, "score_streak_10": 5}
CATALOG = {key: {"id": key, "label": label, "bonus": REWARDS.get(key, 1),
                 "title": TITLES.get(key, "")} for _, rows in GROUPS for key, label in rows}

# Controls that run entirely in the browser can report these one-time actions.
# Game results, ranking, kifu mutations and earned title tags are server-owned.
BROWSER_ACTIONS = frozenset({"replay_kifu", "import_kifu", "image_kifu"})
SEATS = "ABCD"


def round_items(game, seat, *, ai, other_humans, public, support):
    items = set()
    if ai:
        items.add("ai_round")
    if public and other_humans:
        items.add("public_human")
    if support:
        items.add("beginner_round")
    deal = game.get("deal_mode", "normal")
    if deal != "normal":
        items.add("different_deal")
        items.add({"balanced": "balanced_deal", "balanced_wardna": "wardna_deal"}.get(deal, "frequent_deal"))
    preset = game.get("achievement_preset")
    if preset:
        items.update(("different_deal", "preset_deal"))
        if preset == "rough":
            items.add("rough_deal")
    moves = game.get("kifu_moves", [])
    actor = str(SEATS.index(seat))
    partner = str(SEATS.index(seat) ^ 2)
    attacks = [a for who, _, a in moves if who == actor and a]
    if len(attacks) >= 3:
        count = len(set(attacks[:3]))
        if count in (1, 3):
            items.add("same_three" if count == 1 else "different_three")
        if set(attacks[1:3]) == {"王", "玉"}:
            items.add("royal_second_third")
    first = {}
    for who, _, attack in moves:
        if attack and who not in first:
            first[who] = attack
            if who == actor and attack in "香馬銀金" and first.get(partner) == attack:
                items.add("kakarigotae")
    if game["state"].winner != seat or not moves:
        return items
    items.add("first_win")
    _, block, attack = moves[-1]
    points = game.get("last_round_score", 0)
    if not {"8", "9"}.intersection(game.get("init_hands", {}).get(seat, ())):
        items.add("no_royal_win")
    if block and block == attack:
        items.add("double_piece_win")
    if attack in {"王", "玉"} and points >= 50:
        items.add("royal_fifty")
    if {block, attack} == {"王", "玉"} and points == 100:
        items.add("royal_hundred")
    if len(attacks) == 4 and len(moves) >= 5 and moves[-5][0] == actor and moves[-5][2]:
        if all(receive == "パス" for _, receive, _ in moves[-4:-1]):
            items.add("third_round_trip")
    if len(moves) >= 3 and not block and moves[-2][0] == actor and moves[-2][1] not in ("", "パス"):
        opponent = next((who for who, _, played in reversed(moves[:-2]) if played), None)
        if opponent is not None and int(opponent) % 2 != int(actor) % 2:
            if len(game["state"].hands[SEATS[int(opponent)]]) == 2:
                items.add("receive_ready_win")
    return items


def score_progress(rows):
    """Use completed attempts in order, with the raw difference shown in results."""
    progress = {"score_first": int(bool(rows))}
    days, positive_kifu, winners = Counter(), set(), set()
    streak = best_streak = 0
    for row in rows:
        days[row["day"]] += 1
        if row["exact_match_bonus"]:
            progress["score_exact"] = 1
        if row["improvement"] > 0:
            streak += 1
            best_streak = max(best_streak, streak)
            positive_kifu.add(row["challenge"])
            if row["winner_seat"] in ("A", "C"):
                winners.add(row["winner_seat"])
            progress["score_improved"] = 1
            if row["improvement"] >= 30:
                progress["score_plus_30"] = 1
            if row["selection_mode"] == "divergent":
                progress["score_divergent"] = 1
        else:
            streak = 0
    progress.update(score_both_winners=len(winners), score_three_kifu=len(positive_kifu),
                    score_streak_3=best_streak, score_streak_10=best_streak,
                    score_day_50=max(days.values(), default=0), score_day_100=max(days.values(), default=0))
    return progress


TARGETS = {"both_public_modes": 2, "three_partners": 3, "compare_scene": 2,
           "score_both_winners": 2, "score_three_kifu": 3, "score_streak_3": 3, "score_streak_10": 10,
           "score_day_50": 50, "score_day_100": 100}
