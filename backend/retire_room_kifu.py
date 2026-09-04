"""Remove the explicitly retired test-only room library, never AI datasets."""

import sqlite3
from contextlib import closing


def retire_room_kifu(path):
    if not path.is_file():
        return
    with closing(sqlite3.connect(path, timeout=10)) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='research_kifu'").fetchone():
            return
        # The member library uses a different table and is never deleted here.
        db.execute("PRAGMA secure_delete = ON")
        db.execute("DROP TABLE research_kifu")
        db.commit()
