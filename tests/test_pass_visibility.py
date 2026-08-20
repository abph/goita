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
    target_block = HTML.split("function activePassAnimationTarget() {", 1)[1].split(
        "function showPassAnimation", 1
    )[0]
    animation_block = HTML.split("function showPassAnimation(phys) {", 1)[1].split(
        "function showSpecialAnimation", 1
    )[0]

    assert 'document.querySelector(".board-wrap")' in target_block
    assert 'document.getElementById("board")' not in target_block
    assert "wrapper.dataset.passSeat" in animation_block
    assert '.pass-anim-wrapper[data-pass-seat=' in animation_block
    assert "setTimeout(() =>" in animation_block
    assert "}, 2000);" in animation_block
