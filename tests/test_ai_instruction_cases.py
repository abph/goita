import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
CASE_ROOT = ROOT / "goita_ai2" / "instruction_cases"

EXPECTED_CASE_COUNTS = {
    "attack.yaml": 12,
    "endgame.yaml": 11,
    "hand_inference.yaml": 8,
    "kyosha_strategy.yaml": 10,
    "receive.yaml": 10,
    "reference.yaml": 5,
    "regression_candidates.yaml": 13,
    "shi_strategy.yaml": 10,
    "superseded.yaml": 7,
}


def _case_ids(text: str):
    return re.findall(r"^\s+- id:\s+([A-Z]+-\d+)\s*$", text, re.MULTILINE)


def test_instruction_case_catalog_has_expected_files_and_unique_ids():
    all_ids = []
    for filename, expected_count in EXPECTED_CASE_COUNTS.items():
        path = CASE_ROOT / filename
        assert path.exists(), filename
        text = path.read_text(encoding="utf-8")
        assert "schema_version: 1" in text
        ids = _case_ids(text)
        assert len(ids) == expected_count, (filename, ids)
        all_ids.extend(ids)

    assert len(all_ids) == 86
    assert len(all_ids) == len(set(all_ids))


def test_catalog_separates_confirmed_reference_and_superseded_cases():
    reference = (CASE_ROOT / "reference.yaml").read_text(encoding="utf-8")
    superseded = (CASE_ROOT / "superseded.yaml").read_text(encoding="utf-8")
    confirmed_files = [
        "attack.yaml",
        "endgame.yaml",
        "hand_inference.yaml",
        "kyosha_strategy.yaml",
        "receive.yaml",
        "shi_strategy.yaml",
        "regression_candidates.yaml",
    ]

    assert reference.count("status: reference") == 5
    assert superseded.count("status: superseded") == 7
    for filename in confirmed_files:
        text = (CASE_ROOT / filename).read_text(encoding="utf-8")
        assert "status: reference" not in text
        assert "status: superseded" not in text


def test_index_documents_catalog_counts_and_runtime_state():
    index = (CASE_ROOT / "index.yaml").read_text(encoding="utf-8")
    readme = (CASE_ROOT / "README.md").read_text(encoding="utf-8")

    assert "runtime_connected: false" in index
    assert "confirmed_policies: 61" in index
    assert "regression_candidates: 13" in index
    assert "total_records: 86" in index
    assert "AI本体から独立" in readme
    assert "superseded" in readme


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    print("AI_INSTRUCTION_CASES_TEST_OK")

