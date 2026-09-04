"""Account-owned records. Ownership and entitlement are checked in each transaction."""

import json
import secrets
from datetime import datetime, timezone

from backend.member_store import MemberError


class MemberKifuStore:
    LIMIT = 1000

    def __init__(self, members):
        self.members = members

    def _owner(self, db, token, *, paid=False):
        row = self.members._session_row(db, token)
        member = self.members._public(row)
        if member["must_change_password"]:
            raise MemberError(403, "先にパスワードを変更してください。")
        if paid and not member["paid_active"]:
            raise MemberError(403, "新規保存には有効な有料権限が必要です。")
        return member["member_id"]

    @staticmethod
    def _record(row, detail=True):
        payload = json.loads(row["payload_json"])
        result = {key: row[key] for key in ("id", "created_at", "title", "memo")}
        result["tags"] = json.loads(row["tags_json"])
        result["my_seat"] = payload.get("my_seat", "")
        for key, default in (("round_index", 1), ("dealer", "A"), ("winner", ""), ("gained_score", 0)):
            result[key] = payload.get(key, default)
        if detail:
            result["payload"] = payload
        return result

    def list(self, token):
        with self.members._db() as db:
            owner = self._owner(db, token)
            rows = db.execute("SELECT * FROM member_kifu WHERE member_id = ? ORDER BY created_at DESC, id DESC", (owner,)).fetchall()
            return [self._record(row, False) for row in rows]

    def statistics(self, token):
        # Read one owner-scoped snapshot; neither derived results nor seat data
        # are written to analytics or training stores.
        with self.members._db() as db:
            owner = self._owner(db, token)
            rows = db.execute("SELECT payload_json FROM member_kifu WHERE member_id = ?", (owner,)).fetchall()
        result = dict(total=len(rows), counted=0, wins=0, losses=0, points_for=0,
                      points_against=0, self_finishes=0, partner_finishes=0,
                      unset=0, spectator=0, incomplete=0)
        for row in rows:
            payload = json.loads(row["payload_json"])
            seat = payload.get("my_seat", "")
            if seat == "spectator":
                result["spectator"] += 1
                continue
            if seat not in ("A", "B", "C", "D"):
                result["unset"] += 1
                continue
            winner = payload.get("winner")
            points = payload.get("gained_score")
            if winner not in ("A", "B", "C", "D") or type(points) is not int or points <= 0:
                result["incomplete"] += 1
                continue
            result["counted"] += 1
            if (seat in "AC") == (winner in "AC"):
                result["wins"] += 1
                result["points_for"] += points
                result["self_finishes" if seat == winner else "partner_finishes"] += 1
            else:
                result["losses"] += 1
                result["points_against"] += points
        result["win_rate"] = round(100 * result["wins"] / result["counted"], 1) if result["counted"] else None
        result["point_difference"] = result["points_for"] - result["points_against"]
        return result

    def save(self, token, *, title, memo, tags, payload):
        with self.members._db(write=True) as db:
            owner = self._owner(db, token, paid=True)
            if db.execute("SELECT COUNT(*) FROM member_kifu WHERE member_id = ?", (owner,)).fetchone()[0] >= self.LIMIT:
                raise MemberError(409, "保存上限の1000件に達しました。不要な棋譜を削除してください。")
            record_id = "K-" + secrets.token_hex(16)
            db.execute("INSERT INTO member_kifu VALUES (?, ?, ?, ?, ?, ?, ?)", (
                record_id, owner, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                title, memo, json.dumps(tags, ensure_ascii=False), json.dumps(payload, ensure_ascii=False),
            ))
            return self._record(db.execute("SELECT * FROM member_kifu WHERE id = ? AND member_id = ?", (record_id, owner)).fetchone())

    def access(self, token, record_id, *, action="get", title="", memo="", tags=None, my_seat=None):
        with self.members._db(write=action != "get") as db:
            owner = self._owner(db, token)
            row = db.execute("SELECT * FROM member_kifu WHERE member_id = ? AND id = ?", (owner, record_id)).fetchone()
            if row is None:
                raise MemberError(404, "棋譜が見つかりません")
            if action == "delete":
                db.execute("DELETE FROM member_kifu WHERE member_id = ? AND id = ?", (owner, record_id))
                return None
            if action == "edit":
                payload = json.loads(row["payload_json"])
                if my_seat is not None:
                    payload["my_seat"] = my_seat
                db.execute("UPDATE member_kifu SET title = ?, memo = ?, tags_json = ?, payload_json = ? WHERE member_id = ? AND id = ?", (
                    title.strip() or row["title"], memo.strip(),
                    row["tags_json"] if tags is None else json.dumps(tags, ensure_ascii=False),
                    json.dumps(payload, ensure_ascii=False), owner, record_id,
                ))
                row = db.execute("SELECT * FROM member_kifu WHERE member_id = ? AND id = ?", (owner, record_id)).fetchone()
            return self._record(row)
