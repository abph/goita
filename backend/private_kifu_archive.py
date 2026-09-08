"""Server-only archive storage. No raw archive download endpoint is provided."""

import json
import os
import tempfile
from pathlib import Path


MAX_ARCHIVE_BYTES = 20 * 1024 * 1024


def archive_path(base_dir: Path) -> Path:
    persistent = os.environ.get("GOITA_PERSISTENT_DATA_DIR", "").strip()
    if os.environ.get("RENDER") and not persistent:
        raise ValueError("棋譜の保存にはGOITA_PERSISTENT_DATA_DIRの永続保存先を設定してください。")
    directory = Path(persistent) / "private-kifu" if persistent else base_dir / "private_data"
    path = (directory / "kifu_data.json").resolve()
    if path.is_relative_to((base_dir / "frontend").resolve()):
        raise ValueError("棋譜の保存先に公開ディレクトリは指定できません。")
    return path


def parse_archive(raw: bytes) -> dict:
    if len(raw) > MAX_ARCHIVE_BYTES:
        raise ValueError("棋譜ファイルは20MB以下にしてください。")
    try:
        archive = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ValueError("棋譜データの形式が正しくありません。") from error
    if not isinstance(archive, dict) or not isinstance(archive.get("matches"), list):
        raise ValueError("棋譜データの形式が正しくありません。")
    if not archive["matches"] or any(
        not isinstance(match, dict) or not isinstance(match.get("rounds"), list)
        for match in archive["matches"]
    ):
        raise ValueError("棋譜データに有効な対局がありません。")
    return archive


def save_archive(path: Path, raw: bytes) -> dict:
    archive = parse_archive(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A unique temporary file and atomic replacement preserve the previous file
    # if validation or writing fails. mkstemp creates owner-only files on Unix.
    descriptor, temporary = tempfile.mkstemp(prefix=".kifu-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"match_count": len(archive["matches"]),
            "round_count": sum(len(match["rounds"]) for match in archive["matches"])}
