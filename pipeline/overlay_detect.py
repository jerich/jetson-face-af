"""AF overlay detection via HSV color thresholding and contour analysis."""

from enum import Enum
import numpy as np
import cv2

import config


class AFState(Enum):
    """Autofocus overlay state for a face region."""
    NONE = "none"
    DETECTED = "detected"        # Gray square — face detected but not selected
    TRACKED = "tracked"          # White square — tracked/selected
    ACTIVE_FOCUS = "active_focus"  # Green square — actively focused


def detect_af_state(image: np.ndarray, box: np.ndarray) -> AFState:
    """Classify the AF overlay state for a face region.

    Expands the face bounding box by OVERLAY_BOX_EXPAND, then searches
    for colored rectangular overlays indicating AF state.

    Args:
        image: Full frame (BGR, HxWx3).
        box: Face bounding box [x1, y1, x2, y2].

    Returns:
        AFState classification for this face.
    """
    img_height, img_width = image.shape[:2]

    # Expand bounding box
    x1, y1, x2, y2 = box.astype(int)
    w = x2 - x1
    h = y2 - y1
    expand = config.OVERLAY_BOX_EXPAND
    x1_exp = max(0, int(x1 - w * expand))
    y1_exp = max(0, int(y1 - h * expand))
    x2_exp = min(img_width, int(x2 + w * expand))
    y2_exp = min(img_height, int(y2 + h * expand))

    roi = image[y1_exp:y2_exp, x1_exp:x2_exp]
    if roi.size == 0:
        return AFState.NONE

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Check in priority order: green (active) > white (tracked) > gray (detected)
    if _find_af_square(hsv, config.OVERLAY_GREEN_H_RANGE,
                       config.OVERLAY_GREEN_S_RANGE,
                       config.OVERLAY_GREEN_V_RANGE,
                       roi_offset=(x1_exp, y1_exp),
                       image_size=(img_width, img_height)):
        return AFState.ACTIVE_FOCUS

    if _find_af_square(hsv, config.OVERLAY_WHITE_H_RANGE,
                       config.OVERLAY_WHITE_S_RANGE,
                       config.OVERLAY_WHITE_V_RANGE,
                       roi_offset=(x1_exp, y1_exp),
                       image_size=(img_width, img_height)):
        return AFState.TRACKED

    if _find_af_square(hsv, config.OVERLAY_GRAY_H_RANGE,
                       config.OVERLAY_GRAY_S_RANGE,
                       config.OVERLAY_GRAY_V_RANGE,
                       roi_offset=(x1_exp, y1_exp),
                       image_size=(img_width, img_height)):
        return AFState.DETECTED

    return AFState.NONE


def _is_in_edge_margin(cx: int, cy: int, roi_offset: tuple,
                       image_size: tuple, margin: float) -> bool:
    """Check if a point (in ROI coords) falls within the edge margin zone.

    Args:
        cx, cy: Center position in ROI coordinates.
        roi_offset: (x_offset, y_offset) of ROI in full image.
        image_size: (width, height) of full image.
        margin: Fraction of frame to exclude (e.g., 0.10 for 10%).

    Returns:
        True if the point is in the margin zone (should be rejected).
    """
    # Convert to full image coordinates
    full_x = cx + roi_offset[0]
    full_y = cy + roi_offset[1]

    img_width, img_height = image_size
    margin_x = img_width * margin
    margin_y = img_height * margin

    # Check if in margin zone
    if full_x < margin_x or full_x > (img_width - margin_x):
        return True
    if full_y < margin_y or full_y > (img_height - margin_y):
        return True
    return False


def _is_hollow_square(mask: np.ndarray, x: int, y: int, w: int, h: int,
                      max_fill_ratio: float) -> bool:
    """Check if the detected square is hollow (outline only).

    AF indicators are thin outlines (~2-4px). This function samples pixels
    inside the bounding rect and verifies most DON'T match the threshold color.

    Args:
        mask: Binary mask from color thresholding.
        x, y, w, h: Bounding rect of the candidate contour.
        max_fill_ratio: Maximum allowed fill ratio for interior.

    Returns:
        True if the square is hollow (valid AF indicator).
    """
    # Shrink to interior (exclude the border)
    border = max(2, int(min(w, h) * 0.2))
    interior = mask[y + border:y + h - border, x + border:x + w - border]
    if interior.size == 0:
        return True  # Too small to check, assume valid
    fill_ratio = np.count_nonzero(interior) / interior.size
    return fill_ratio < max_fill_ratio


def _find_af_square(hsv: np.ndarray, h_range: tuple, s_range: tuple,
                    v_range: tuple, roi_offset: tuple = (0, 0),
                    image_size: tuple = None) -> bool:
    """Check if the HSV ROI contains a rectangular AF overlay indicator.

    Args:
        hsv: HSV image of the ROI.
        h_range, s_range, v_range: HSV threshold ranges.
        roi_offset: (x, y) offset of ROI in full image coordinates.
        image_size: (width, height) of full image for edge margin check.

    Returns:
        True if a valid square-ish hollow contour is found within the color range.
    """
    lower = np.array([h_range[0], s_range[0], v_range[0]], dtype=np.uint8)
    upper = np.array([h_range[1], s_range[1], v_range[1]], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < config.OVERLAY_MIN_AREA or area > config.OVERLAY_MAX_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            continue

        # Check aspect ratio (must be square)
        aspect = w / h
        if not (config.OVERLAY_ASPECT_MIN <= aspect <= config.OVERLAY_ASPECT_MAX):
            continue

        # Edge margin exclusion: reject squares near frame edges
        if image_size is not None:
            cx = x + w // 2
            cy = y + h // 2
            if _is_in_edge_margin(cx, cy, roi_offset, image_size,
                                  config.OVERLAY_EDGE_MARGIN):
                continue

        # Hollow square validation: AF indicators are outlines, not filled
        if not _is_hollow_square(mask, x, y, w, h, config.OVERLAY_HOLLOW_RATIO):
            continue

        return True

    return False
