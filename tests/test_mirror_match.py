"""Tests paired team swapping for current-versus-frozen AI matches.

Both orientations must receive the same deal and dealer, finish normally, and
contribute wins, points, and timing to the combined mirror-match report.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from goita_ai2.mirror_match import (
    MirrorMatchConfig,
    run_mirror_match,
    write_mirror_report,
)


def test_one_mirror_pair_swaps_teams_on_the_same_deal() -> None:
    report = run_mirror_match(MirrorMatchConfig(
        pairs=1,
        seed=1_261_234,
        max_steps=120,
        search_seconds=0.01,
        search_samples=2,
        search_depth=1,
        search_nodes=500,
    ))

    assert report["summary"]["games"] == 2
    assert report["summary"]["finished"] == 2
    assert report["summary"]["errors"] == 0
    first, second = report["games"]
    assert first["initial_hands"] == second["initial_hands"]
    assert first["dealer"] == second["dealer"]
    assert {first["current_team"], second["current_team"]} == {"AC", "BD"}
    assert report["summary"]["decision_timing"]["current"]["decisions"] > 0

    with TemporaryDirectory() as directory:
        output = Path(directory) / "mirror.json"
        write_mirror_report(output, report)
        stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["config"]["pairs"] == 1


if __name__ == "__main__":
    test_one_mirror_pair_swaps_teams_on_the_same_deal()
    print("MIRROR_MATCH_TEST_OK")
