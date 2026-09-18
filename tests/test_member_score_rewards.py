from datetime import datetime
import hashlib

import pytest

from backend.member_store import MemberError, MemberStore
from backend.member_kifu import MemberKifuStore
from backend.trace_results import TraceStore


def stamp(value):
    return datetime.fromisoformat(value).timestamp()


def seed_member(store, member_id, *, free=True):
    now = store.clock()
    store._schema()
    with store._db(write=True) as db:
        db.execute("""INSERT INTO members
            (member_id,password_hash,must_change_password,temporary_expires_at,enabled,
             paid_enabled,paid_until,created_at,updated_at,is_operator,research_enabled,
             managed_room_id,registration_source,last_login_at)
            VALUES (?, 'unused', 0, NULL, 1, ?, NULL, ?, ?, 0, 0, '', 'self', ?)""",
            (member_id, 0 if free else 1, now, now, now))


def test_weekly_awards_are_idempotent_and_titles_expire(tmp_path):
    now = [stamp("2026-09-14T12:00:00+09:00")]
    store = MemberStore(tmp_path / "members.sqlite3", clock=lambda: now[0])
    for member_id in ("gold-one", "gold-two", "bronze-one"):
        seed_member(store, member_id)

    candidates = [
        {"member_id": "gold-one", "rank": 1},
        {"member_id": "gold-two", "rank": 1},
        {"member_id": "bronze-one", "rank": 3},
    ]
    assert len(store.grant_weekly_score_awards("2026-09-07", candidates)) == 3
    assert store.grant_weekly_score_awards("2026-09-07", candidates) == []

    members = {item["member_id"]: item for item in store.list_members()}
    assert members["gold-one"]["kifu_limit"] == 30
    assert members["gold-two"]["reward_kifu_bonus"] == 10
    assert members["bronze-one"]["kifu_limit"] == 23
    assert members["gold-one"]["score_title"]["key"] == "gold"
    assert members["gold-one"]["score_champion_stamp"] is True

    now[0] = stamp("2026-09-21T00:00:00+09:00")
    with store._db() as db:
        row = db.execute("SELECT * FROM members WHERE member_id='gold-one'").fetchone()
    expired = store.with_usage(store._public(row))
    assert expired["score_title"] is None
    assert expired["score_award_history"]["gold"] == 1
    assert expired["score_champion_stamp"] is True


def test_daily_first_place_adds_one_slot_up_to_fifty_and_keeps_counting(tmp_path):
    store = MemberStore(tmp_path / "members.sqlite3", clock=lambda: stamp("2026-09-18T12:00:00+09:00"))
    seed_member(store, "daily-one")
    with store._db(write=True) as db:
        db.execute("UPDATE members SET daily_kifu_bonus = 49 WHERE member_id = 'daily-one'")

    candidates = [{"member_id": "daily-one", "rank": 1}]
    assert store.grant_daily_score_awards("2026-09-16", candidates) == [
        {"member_id": "daily-one", "rank": 1, "kifu_bonus": 1}
    ]
    assert store.grant_daily_score_awards("2026-09-16", candidates) == []
    assert store.grant_daily_score_awards("2026-09-17", candidates) == [
        {"member_id": "daily-one", "rank": 1, "kifu_bonus": 0}
    ]

    member = store.list_members()[0]
    assert member["daily_score_first_place_count"] == 2
    assert member["daily_kifu_bonus"] == 50
    assert member["daily_kifu_bonus_cap"] == 50
    assert member["kifu_limit"] == 70
    assert store.daily_score_settlement_date() == "2026-09-17"


def test_configurable_base_and_admin_bonus_do_not_delete_existing_usage(tmp_path):
    store = MemberStore(tmp_path / "members.sqlite3", clock=lambda: stamp("2026-09-14T12:00:00+09:00"))
    seed_member(store, "free-one")
    store.update_reward_settings(free_base_limit=25, paid_base_limit=1200,
                                 rank1_bonus=12, rank2_bonus=6, rank3_bonus=3,
                                 reward_bonus_cap=120)
    member = store.update_admin_kifu_bonus("free-one", 7, "イベント参加分")
    assert member["kifu_base_limit"] == 25
    assert member["kifu_limit"] == 32
    with store._db() as db:
        history = db.execute("SELECT old_bonus,new_bonus,note FROM member_kifu_quota_history").fetchone()
    assert tuple(history) == (0, 7, "イベント参加分")
    with pytest.raises(MemberError):
        store.update_reward_settings(free_base_limit=20, paid_base_limit=1000,
                                     rank1_bonus=3, rank2_bonus=5, rank3_bonus=1,
                                     reward_bonus_cap=100)


def test_lowering_quota_keeps_saved_kifu_and_blocks_new_saves(tmp_path):
    store = MemberStore(tmp_path / "members.sqlite3", clock=lambda: stamp("2026-09-14T12:00:00+09:00"))
    seed_member(store, "free-one")
    token = "test-session"
    with store._db(write=True) as db:
        db.execute("INSERT INTO member_sessions VALUES (?, ?, ?, ?)",
                   (hashlib.sha256(token.encode()).hexdigest(), "free-one", store.clock(), store.clock() + 3600))
    kifu = MemberKifuStore(store)
    store.update_admin_kifu_bonus("free-one", 2)
    for index in range(22):
        kifu.save(token, title=str(index), memo="", tags=[], payload={})
    store.update_admin_kifu_bonus("free-one", 0, "基本枠へ戻す")
    assert len(kifu.list(token)) == 22
    with pytest.raises(MemberError) as error:
        kifu.save(token, title="blocked", memo="", tags=[], payload={})
    assert error.value.status == 409
    assert len(kifu.list(token)) == 22


def test_ranking_candidates_keep_guest_positions_and_ties(tmp_path):
    now = [stamp("2026-09-10T12:00:00+09:00")]
    trace = TraceStore(tmp_path / "trace.sqlite3", clock=lambda: now[0])
    payload = {"hands": {}, "dealer": "A", "moves": [],
               "score_before": {"AC": 0, "BD": 0}, "score_after": {"AC": 0, "BD": 0}}
    for owner, guest, points in (("guest:one", True, 100), ("member:silver-one", False, 90),
                                  ("member:silver-two", False, 90)):
        attempt = trace.start(owner, guest, owner, payload)
        trace.finish(attempt, {"AC": points, "BD": 0})
    now[0] = stamp("2026-09-14T00:00:00+09:00")
    result = trace.weekly_award_candidates()
    assert result["week_start"] == "2026-09-07"
    assert result["candidates"] == [
        {"member_id": "silver-one", "rank": 2},
        {"member_id": "silver-two", "rank": 2},
    ]


def test_daily_candidates_keep_guest_positions_and_first_place_ties(tmp_path):
    now = [stamp("2026-09-17T12:00:00+09:00")]
    trace = TraceStore(tmp_path / "daily-awards.sqlite", clock=lambda: now[0])
    payload = {"hands": {}, "dealer": "A", "moves": [],
               "score_before": {"AC": 0, "BD": 0}, "score_after": {"AC": 0, "BD": 0}}
    for owner, points in (("guest:one", 100), ("member:one", 100), ("member:two", 90)):
        attempt = trace.start(owner, owner.startswith("guest:"), owner, payload)
        trace.finish(attempt, {"AC": points, "BD": 0})

    now[0] = stamp("2026-09-18T00:00:00+09:00")
    assert trace.daily_award_candidates() == {
        "award_date": "2026-09-17",
        "candidates": [{"member_id": "one", "rank": 1}],
    }
    assert trace.daily_award_candidates("2026-09-16")["candidates"] == []
