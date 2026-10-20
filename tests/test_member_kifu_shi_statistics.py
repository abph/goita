import pytest

from backend.member_kifu import MemberKifuStore
from backend.member_store import MemberError, MemberStore


@pytest.fixture
def library(tmp_path):
    members = MemberStore(tmp_path / "members.sqlite3")

    def account(name):
        issued = members.create(name)
        _, temporary, _ = members.login(name, issued["temporary_password"])
        return members.change_password(temporary, issued["temporary_password"], "test-password-2026")[1]

    return MemberKifuStore(members), members, account("alice"), account("bobby")


def save(store, token, initial_hand, seat="A", **changes):
    payload = {"my_seat": seat, "hand": {"p0": initial_hand}, "winner": "A", "gained_score": 20}
    payload.update(changes)
    return store.save(token, title="sample", memo="", tags=[], payload=payload)


def test_shi_average_and_distribution_include_zero_and_four_or_more(library):
    store, _, token, _ = library
    for count in range(6):
        save(store, token, "し" * count + "香馬銀金角飛王玉"[:8 - count])
    result = store.statistics(token)
    shi = result["initial_shi"]
    assert shi["counted"] == 6
    assert shi["average"] == 2.5
    assert shi["missing_hand"] == 0
    assert [row["rounds"] for row in shi["distribution"]] == [1, 1, 1, 1, 2]
    assert [row["percentage"] for row in shi["distribution"]] == [16.7, 16.7, 16.7, 16.7, 33.3]
    assert result["wins"] == 6


def test_shi_statistics_read_the_owners_initial_seat_and_ignore_result(library):
    store, _, token, _ = library
    for seat in "ABCD":
        hands = {f"p{index}": "しししし香馬銀金" for index in range(4)}
        hands[f"p{'ABCD'.index(seat)}"] = "し香馬銀金角飛王"
        save(store, token, "", seat=seat, hand=hands, winner="", gained_score=0,
             hands={seat: ["し", "し", "し"]})
    result = store.statistics(token)
    assert result["counted"] == 0
    assert result["incomplete"] == 4
    assert result["initial_shi"]["counted"] == 4
    assert result["initial_shi"]["average"] == 1
    assert result["initial_shi"]["distribution"][1]["rounds"] == 4


def test_missing_initial_hands_are_excluded_instead_of_counted_as_zero(library):
    store, _, token, _ = library
    for hand in [None, "", "し香馬銀金角飛", "xxxxxxxx", ["し"] * 8]:
        save(store, token, hand)
    save(store, token, "", hand={"p1": "しし香馬銀金角飛"})
    save(store, token, "", hand=None)
    save(store, token, "しし香馬銀金角飛", seat="")
    save(store, token, "しし香馬銀金角飛", seat="spectator")
    shi = store.statistics(token)["initial_shi"]
    assert shi["counted"] == 0
    assert shi["average"] is None
    assert shi["missing_hand"] == 7
    assert all(row["rounds"] == 0 and row["percentage"] is None for row in shi["distribution"])
    save(store, token, "香馬銀金角飛王玉")
    shi = store.statistics(token)["initial_shi"]
    assert shi["average"] == 0
    assert shi["distribution"][0]["percentage"] == 100


def test_shi_statistics_are_owner_scoped_read_only_and_follow_edits_and_deletes(library):
    store, members, token, other = library
    first = save(store, token, "しし香馬銀金角飛", hand={
        "p0": "しし香馬銀金角飛", "p1": "しししし香馬銀金",
    })
    second = save(store, token, "しし香馬銀金角飛")
    save(store, token, "ししし香馬銀金角")
    save(store, other, "ししししし香馬銀")
    with members._db() as db:
        before = [tuple(row) for row in db.execute("SELECT * FROM member_kifu ORDER BY id")]
    assert store.statistics(token)["initial_shi"]["average"] == 2.33
    assert store.statistics(other)["initial_shi"]["average"] == 5
    with members._db() as db:
        assert [tuple(row) for row in db.execute("SELECT * FROM member_kifu ORDER BY id")] == before
    store.access(token, first["id"], action="edit", my_seat="B")
    assert store.statistics(token)["initial_shi"]["average"] == 3
    store.access(token, second["id"], action="delete")
    assert store.statistics(token)["initial_shi"]["average"] == 3.5
    assert store.statistics(token)["initial_shi"]["counted"] == 2
    members.update("alice", enabled=True, paid_enabled=False, paid_until=None)
    assert store.statistics(token)["initial_shi"]["average"] == 3.5
    with pytest.raises(MemberError):
        store.statistics("invalid-token")
