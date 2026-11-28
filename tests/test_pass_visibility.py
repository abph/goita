from pathlib import Path


HTML = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")


def test_pass_visual_is_not_gated_by_audio_unlock():
    assert "const shouldProcessLogFeedback = (" in HTML
    assert "if (shouldProcessLogFeedback) {" in HTML
    assert "shouldProcessLogFeedback &&\n    audioUnlocked" in HTML

    pass_block = HTML.split("if (isPass) {", 1)[1].split("if (shouldPlayLogAudio && cVoiceKind", 1)[0]
    assert "if(shouldPlayLogAudio) {" in pass_block
    assert "showPassAnimation(physSeat);" in pass_block
    assert pass_block.index("showPassAnimation(physSeat);") > pass_block.index("if(shouldPlayLogAudio) {")


def test_two_dimensional_pass_marker_survives_board_rerenders():
    animation_block = HTML.split("function showPassAnimation(phys) {", 1)[1].split(
        "function showBoardStamp", 1
    )[0]

    assert 'board.replaceChildren(effects);' in HTML
    assert 'activeBoardSlotEffects()' in animation_block
    assert 'SLOTS[phys].attack.slice(1, 3)' in animation_block
    assert 'COMBO_POS[phys].rotate' in animation_block
    assert "wrapper.dataset.passSeat" in animation_block
    assert '.pass-anim-wrapper[data-pass-seat=' in animation_block
    assert "setTimeout(() => wrapper.remove(), 2000);" in animation_block
