"""Tests for AF overlay detection."""

import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.overlay_detect import detect_af_state, AFState, _find_af_square
import config


def _make_frame_with_square(color_bgr, box, square_size=40, hollow=True, border=3):
    """Create a test frame with a colored square near the face box.

    Args:
        color_bgr: Color of the square in BGR format.
        box: Face bounding box [x1, y1, x2, y2].
        square_size: Size of the square in pixels.
        hollow: If True, draw only the outline (like real AF indicators).
        border: Border thickness for hollow squares.
    """
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Fill with dark background
    frame[:] = (30, 30, 30)

    # Draw the colored square centered on the face box
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half = square_size // 2

    if hollow:
        # Draw only the outline (top, bottom, left, right borders)
        # Top border
        frame[cy - half:cy - half + border, cx - half:cx + half] = color_bgr
        # Bottom border
        frame[cy + half - border:cy + half, cx - half:cx + half] = color_bgr
        # Left border
        frame[cy - half:cy + half, cx - half:cx - half + border] = color_bgr
        # Right border
        frame[cy - half:cy + half, cx + half - border:cx + half] = color_bgr
    else:
        # Filled square
        frame[cy - half:cy + half, cx - half:cx + half] = color_bgr
    return frame


def test_detect_green_active_focus():
    """Green square should be classified as ACTIVE_FOCUS."""
    box = np.array([500, 300, 700, 500])
    # Pure green in BGR
    frame = _make_frame_with_square((0, 200, 0), box)
    state = detect_af_state(frame, box)
    assert state == AFState.ACTIVE_FOCUS, f"Expected ACTIVE_FOCUS, got {state}"


def test_detect_white_tracked():
    """White square should be classified as TRACKED."""
    box = np.array([500, 300, 700, 500])
    # White in BGR
    frame = _make_frame_with_square((240, 240, 240), box)
    state = detect_af_state(frame, box)
    assert state == AFState.TRACKED, f"Expected TRACKED, got {state}"


def test_detect_gray_detected():
    """Gray square should be classified as DETECTED."""
    box = np.array([500, 300, 700, 500])
    # Mid-gray in BGR
    frame = _make_frame_with_square((140, 140, 140), box)
    state = detect_af_state(frame, box)
    assert state == AFState.DETECTED, f"Expected DETECTED, got {state}"


def test_detect_none():
    """Dark frame with no colored squares should return NONE."""
    box = np.array([500, 300, 700, 500])
    frame = np.full((1080, 1920, 3), 20, dtype=np.uint8)
    state = detect_af_state(frame, box)
    assert state == AFState.NONE, f"Expected NONE, got {state}"


def _draw_hollow_square(frame, y1, y2, x1, x2, color_bgr, border=3):
    """Draw a hollow square outline on the frame."""
    # Top border
    frame[y1:y1 + border, x1:x2] = color_bgr
    # Bottom border
    frame[y2 - border:y2, x1:x2] = color_bgr
    # Left border
    frame[y1:y2, x1:x1 + border] = color_bgr
    # Right border
    frame[y1:y2, x2 - border:x2] = color_bgr


def test_green_takes_priority_over_white():
    """When both green and white squares present, green should win."""
    box = np.array([500, 300, 700, 500])
    frame = np.full((1080, 1920, 3), 30, dtype=np.uint8)

    # White hollow square
    _draw_hollow_square(frame, 380, 420, 550, 590, (240, 240, 240))
    # Green hollow square
    _draw_hollow_square(frame, 380, 420, 620, 660, (0, 200, 0))

    state = detect_af_state(frame, box)
    assert state == AFState.ACTIVE_FOCUS, f"Expected ACTIVE_FOCUS, got {state}"


def test_small_square_ignored():
    """Squares below minimum area should be ignored."""
    box = np.array([500, 300, 700, 500])
    frame = np.full((1080, 1920, 3), 30, dtype=np.uint8)

    # Very small green square (5x5 = 25 pixels, below OVERLAY_MIN_AREA)
    frame[398:403, 598:603] = (0, 200, 0)

    state = detect_af_state(frame, box)
    assert state == AFState.NONE, f"Expected NONE, got {state}"


def test_box_expansion():
    """Overlay should be detected even if slightly outside the face box."""
    box = np.array([500, 300, 700, 500])
    frame = np.full((1080, 1920, 3), 30, dtype=np.uint8)

    # Green hollow square just outside the box but within expansion range
    # Box width = 200, 30% expansion = 60 pixels each side
    # So overlay at x=720 (20px outside box) should still be found
    _draw_hollow_square(frame, 380, 420, 710, 750, (0, 200, 0))

    state = detect_af_state(frame, box)
    assert state == AFState.ACTIVE_FOCUS, f"Expected ACTIVE_FOCUS, got {state}"


def test_empty_roi():
    """Box at image edge should handle gracefully."""
    box = np.array([0, 0, 10, 10])
    frame = np.full((1080, 1920, 3), 30, dtype=np.uint8)
    state = detect_af_state(frame, box)
    assert state == AFState.NONE, f"Expected NONE, got {state}"


def test_filled_square_rejected():
    """Filled squares (not hollow outlines) should be rejected."""
    box = np.array([500, 300, 700, 500])
    # Create a filled green square (hollow=False)
    frame = _make_frame_with_square((0, 200, 0), box, hollow=False)
    state = detect_af_state(frame, box)
    assert state == AFState.NONE, f"Expected NONE (filled rejected), got {state}"


def test_edge_margin_square_rejected():
    """Squares near the frame edge should be rejected."""
    # Place face box near edge so the square falls in margin zone
    # Frame is 1920x1080, 5% margin = 96px from left/right, 54px from top/bottom
    box = np.array([20, 20, 80, 80])  # Very close to top-left corner
    frame = np.full((1080, 1920, 3), 30, dtype=np.uint8)

    # Draw hollow green square centered on face (at ~50, 50 - within 5% margin)
    _draw_hollow_square(frame, 30, 70, 30, 70, (0, 200, 0))

    state = detect_af_state(frame, box)
    assert state == AFState.NONE, f"Expected NONE (edge margin rejected), got {state}"


def test_center_square_accepted():
    """Squares in the center of the frame should be accepted."""
    # Place face in center of frame
    box = np.array([860, 440, 1060, 640])  # Center of 1920x1080
    frame = np.full((1080, 1920, 3), 30, dtype=np.uint8)

    # Draw hollow green square centered on face
    _draw_hollow_square(frame, 520, 560, 940, 980, (0, 200, 0))

    state = detect_af_state(frame, box)
    assert state == AFState.ACTIVE_FOCUS, f"Expected ACTIVE_FOCUS, got {state}"


if __name__ == "__main__":
    tests = [
        test_detect_green_active_focus,
        test_detect_white_tracked,
        test_detect_gray_detected,
        test_detect_none,
        test_green_takes_priority_over_white,
        test_small_square_ignored,
        test_box_expansion,
        test_empty_roi,
        test_filled_square_rejected,
        test_edge_margin_square_rejected,
        test_center_square_accepted,
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
