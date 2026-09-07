"""Split external match records into independently validated rounds."""

import hashlib
import re


def parse_kifu_rounds(text, parse):
    normalized = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    starts = list(re.finditer(r"(?m)^\s*-?\s*hand:\s*$", normalized))
    if len(starts) <= 1:
        return [parse(normalized)]
    header = normalized[:starts[0].start()]
    group = hashlib.sha256(normalized.encode()).hexdigest()
    rounds = []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(normalized)
        try:
            payload = parse(header + normalized[start.start():end], score_is_before=True)
            if rounds and rounds[-1]["score_after"] != payload["score_before"]:
                raise ValueError("前の局の終了点数と開始点数が一致しません")
        except ValueError as error:
            raise ValueError(f"第{index + 1}局: {error}") from error
        payload.update(round_index=index + 1, import_round_count=len(starts), import_group_id=group)
        rounds.append(payload)
    return rounds
