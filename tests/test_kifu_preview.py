from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend import app as game_app
from backend.member_kifu import MemberKifuStore
from test_research_kifu_library import _valid_kifu_text


@pytest.fixture
def preview(monkeypatch):
    def no_save(*args, **kwargs):
        raise AssertionError("Preview must not save records")
    monkeypatch.setattr(MemberKifuStore, "save", no_save)
    monkeypatch.setattr(MemberKifuStore, "save_many", no_save)
    monkeypatch.setattr(game_app, "GAMES", {})
    app = FastAPI()
    app.add_api_route('/kifu/preview', game_app.preview_kifu_file, methods=['POST'])
    with TestClient(app) as client:
        yield client
    assert game_app.GAMES == {}


def test_guest_can_preview_all_rounds_without_saving(preview):
    text = (Path(__file__).parent / 'fixtures' / 'external_match.yaml').read_text(encoding='utf-8-sig')
    response = preview.post('/kifu/preview', json={'kifu_text': text})
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'
    rounds = response.json()['rounds']
    assert len(rounds) == 10
    assert rounds[-1]['score_after'] == {'AC':140, 'BD':160}
    assert rounds[0]['player_names']['A'] == 'Sample player 0'


def test_single_round_and_invalid_files(preview):
    response = preview.post('/kifu/preview', json={'kifu_text':_valid_kifu_text()})
    assert response.status_code == 200
    assert len(response.json()['rounds']) == 1
    assert preview.post('/kifu/preview', json={'kifu_text':'invalid'}).status_code == 400
    assert preview.post('/kifu/preview', json={'kifu_text':'x' * 200001}).status_code == 422
