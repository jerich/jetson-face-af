"""Tests for AF steering command logic."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.command_logic import CommandLogic, FaceInfo, Command
from pipeline.overlay_detect import AFState
import config


def _make_face(x_center, y_center=400, width=200, height=200,
               is_target=False, similarity=0.0,
               af_state=AFState.NONE) -> FaceInfo:
    """Helper to create a FaceInfo at a given position."""
    x1 = x_center - width // 2
    x2 = x_center + width // 2
    y1 = y_center - height // 2
    y2 = y_center + height // 2
    return FaceInfo(
        box=np.array([x1, y1, x2, y2], dtype=np.float32),
        is_target=is_target,
        similarity=similarity,
        af_state=af_state,
    )


def test_no_target_no_command():
    """No target face → no command."""
    logic = CommandLogic()
    faces = [_make_face(500, af_state=AFState.TRACKED)]
    cmd = logic.update(faces)
    assert cmd == Command.NONE


def test_target_already_tracked():
    """Target already has AF → no command."""
    logic = CommandLogic()
    faces = [_make_face(500, is_target=True, similarity=0.8,
                        af_state=AFState.TRACKED)]
    cmd = logic.update(faces)
    assert cmd == Command.NONE


def test_target_already_active_focus():
    """Target already has active focus → no command."""
    logic = CommandLogic()
    faces = [_make_face(500, is_target=True, similarity=0.8,
                        af_state=AFState.ACTIVE_FOCUS)]
    cmd = logic.update(faces)
    assert cmd == Command.NONE


def test_debounce_requires_consistent_frames():
    """First command requires INITIAL_CONFIRM_FRAMES consistent decisions."""
    logic = CommandLogic()

    # Target at left (x=300) with a DETECTED AF point, active AF at right (x=700)
    # The gray AF point inside target makes it a valid steering scenario
    faces = [
        _make_face(300, is_target=True, similarity=0.8, af_state=AFState.DETECTED),
        _make_face(700, is_target=False, af_state=AFState.TRACKED),
    ]

    # Should return NONE for first (INITIAL_CONFIRM_FRAMES - 1) frames
    for i in range(config.INITIAL_CONFIRM_FRAMES - 1):
        cmd = logic.update(faces)
        assert cmd == Command.NONE, f"Frame {i}: expected NONE during debounce"

    # On the Nth frame, should return the command
    cmd = logic.update(faces)
    assert cmd == Command.LEFT, f"Expected LEFT after debounce, got {cmd}"


def test_left_command():
    """Target left of AF face → LEFT command (after debounce)."""
    logic = CommandLogic()

    # Target has DETECTED state (gray AF inside), active AF is elsewhere
    faces = [
        _make_face(300, is_target=True, similarity=0.8, af_state=AFState.DETECTED),
        _make_face(700, is_target=False, af_state=AFState.TRACKED),
    ]

    # Pass debounce period
    for _ in range(config.INITIAL_CONFIRM_FRAMES):
        cmd = logic.update(faces)

    assert cmd == Command.LEFT


def test_right_command():
    """Target right of AF face → RIGHT command (after debounce)."""
    logic = CommandLogic()

    # Target has DETECTED state (gray AF inside), active AF is elsewhere
    faces = [
        _make_face(700, is_target=True, similarity=0.8, af_state=AFState.DETECTED),
        _make_face(300, is_target=False, af_state=AFState.TRACKED),
    ]

    for _ in range(config.INITIAL_CONFIRM_FRAMES):
        cmd = logic.update(faces)

    assert cmd == Command.RIGHT


def test_fn1_when_vertically_stacked():
    """Vertically stacked faces → FN1 command."""
    logic = CommandLogic()

    # Target has DETECTED state (gray AF inside), active AF is vertically offset
    faces = [
        _make_face(500, y_center=200, is_target=True, similarity=0.8,
                   af_state=AFState.DETECTED),
        _make_face(500, y_center=700, is_target=False,
                   af_state=AFState.TRACKED),
    ]

    for _ in range(config.INITIAL_CONFIRM_FRAMES):
        cmd = logic.update(faces)

    assert cmd == Command.FN1


def test_no_command_when_no_af_face():
    """No AF face visible → no command (nothing to steer away from)."""
    logic = CommandLogic()

    # Target has DETECTED state, but no face has TRACKED/ACTIVE_FOCUS
    # Without an active AF outside target, there's nothing to steer away from
    faces = [
        _make_face(500, is_target=True, similarity=0.8, af_state=AFState.DETECTED),
        _make_face(700, is_target=False, af_state=AFState.NONE),
    ]

    for _ in range(config.INITIAL_CONFIRM_FRAMES + 1):
        cmd = logic.update(faces)

    assert cmd == Command.NONE, f"Expected NONE (no AF to steer from), got {cmd}"


def test_stuck_detection():
    """After STUCK_THRESHOLD same commands, should try alternative."""
    logic = CommandLogic()

    # Target has DETECTED state (gray AF inside), active AF is elsewhere
    faces = [
        _make_face(300, is_target=True, similarity=0.8, af_state=AFState.DETECTED),
        _make_face(700, is_target=False, af_state=AFState.TRACKED),
    ]

    # Pass debounce (last frame of this loop sends the first LEFT)
    for _ in range(config.INITIAL_CONFIRM_FRAMES):
        logic.update(faces)

    # Send LEFT commands — the stuck flip triggers on the STUCK_THRESHOLD-th
    # additional command (since the debounce exit already set last_sent=LEFT)
    commands = []
    for _ in range(config.STUCK_THRESHOLD):
        cmd = logic.update(faces)
        commands.append(cmd)

    # The last command should be the flipped direction
    assert commands[-1] == Command.RIGHT, \
        f"Expected RIGHT after stuck, got {commands[-1]}"


def test_reset_on_target_lost():
    """Losing target resets debounce state."""
    logic = CommandLogic()

    # Target has DETECTED state (gray AF inside), active AF is elsewhere
    faces_with_target = [
        _make_face(300, is_target=True, similarity=0.8, af_state=AFState.DETECTED),
        _make_face(700, is_target=False, af_state=AFState.TRACKED),
    ]

    # Partial debounce
    logic.update(faces_with_target)

    # Target disappears
    faces_no_target = [
        _make_face(700, is_target=False, af_state=AFState.TRACKED),
    ]
    logic.update(faces_no_target)

    # Target reappears — debounce should restart
    for i in range(config.INITIAL_CONFIRM_FRAMES - 1):
        cmd = logic.update(faces_with_target)
        assert cmd == Command.NONE, f"Frame {i}: debounce should have reset"


def test_reset_on_success():
    """When target gets AF, state resets."""
    logic = CommandLogic()

    # Target has DETECTED state (gray AF inside), active AF is elsewhere
    faces_need_steer = [
        _make_face(300, is_target=True, similarity=0.8, af_state=AFState.DETECTED),
        _make_face(700, is_target=False, af_state=AFState.TRACKED),
    ]

    # Pass debounce and send commands
    for _ in range(config.INITIAL_CONFIRM_FRAMES + 2):
        logic.update(faces_need_steer)

    # Target now has AF
    faces_success = [
        _make_face(300, is_target=True, similarity=0.8, af_state=AFState.TRACKED),
    ]
    cmd = logic.update(faces_success)
    assert cmd == Command.NONE

    # New steering should require debounce again
    for i in range(config.INITIAL_CONFIRM_FRAMES - 1):
        cmd = logic.update(faces_need_steer)
        assert cmd == Command.NONE, f"Frame {i}: debounce should have reset after success"


def test_af_overlapping_target_no_command():
    """AF on different detection but overlapping target → no command.

    This handles cases where AF is on an eye region while target is the full face.
    """
    logic = CommandLogic()

    # Target is full face (large box)
    # AF is on overlapping region (e.g., eye detection within face)
    faces = [
        _make_face(500, is_target=True, similarity=0.8, width=300, height=300,
                   af_state=AFState.NONE),
        _make_face(480, is_target=False, width=100, height=100,
                   af_state=AFState.TRACKED),  # Overlaps with target
    ]

    # Should return NONE because AF overlaps with target
    for _ in range(config.INITIAL_CONFIRM_FRAMES + 1):
        cmd = logic.update(faces)

    assert cmd == Command.NONE, f"Expected NONE (AF overlaps target), got {cmd}"


def test_no_detected_inside_target_no_command():
    """No gray AF point inside target → no command (nothing to switch to)."""
    logic = CommandLogic()

    # Target has no AF, and there's no DETECTED point inside target
    faces = [
        _make_face(300, is_target=True, similarity=0.8, af_state=AFState.NONE),
        _make_face(700, is_target=False, af_state=AFState.TRACKED),
    ]

    # Should return NONE because there's no DETECTED AF inside target
    for _ in range(config.INITIAL_CONFIRM_FRAMES + 1):
        cmd = logic.update(faces)

    assert cmd == Command.NONE, f"Expected NONE (no DETECTED inside target), got {cmd}"


if __name__ == "__main__":
    tests = [
        test_no_target_no_command,
        test_target_already_tracked,
        test_target_already_active_focus,
        test_debounce_requires_consistent_frames,
        test_left_command,
        test_right_command,
        test_fn1_when_vertically_stacked,
        test_no_command_when_no_af_face,
        test_stuck_detection,
        test_reset_on_target_lost,
        test_reset_on_success,
        test_af_overlapping_target_no_command,
        test_no_detected_inside_target_no_command,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR: {test.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
