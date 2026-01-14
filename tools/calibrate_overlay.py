"""Interactive HSV threshold calibration tool for AF overlay detection.

Usage:
    python3 tools/calibrate_overlay.py                  # Live camera feed
    python3 tools/calibrate_overlay.py --image frame.png  # Single image

Adjust trackbars to find optimal thresholds for white/green/gray AF squares.
Press 'p' to print current values, 'q' to quit.

Color coding:
    Green outline  = Accepted (square, hollow, in valid region)
    Red outline    = Rejected (wrong aspect ratio)
    Yellow outline = Rejected (in edge margin zone)
    Cyan outline   = Rejected (filled, not hollow)
"""

import argparse
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def is_in_edge_margin(cx, cy, img_width, img_height, margin):
    """Check if a point falls within the edge margin zone."""
    margin_x = img_width * margin
    margin_y = img_height * margin
    if cx < margin_x or cx > (img_width - margin_x):
        return True
    if cy < margin_y or cy > (img_height - margin_y):
        return True
    return False


def is_hollow_square(mask, x, y, w, h, max_fill_ratio):
    """Check if the detected square is hollow (outline only)."""
    border = max(2, int(min(w, h) * 0.2))
    interior = mask[y + border:y + h - border, x + border:x + w - border]
    if interior.size == 0:
        return True, 0.0
    fill_ratio = np.count_nonzero(interior) / interior.size
    return fill_ratio < max_fill_ratio, fill_ratio


def draw_edge_margin_zone(display, img_width, img_height, margin):
    """Draw the edge margin exclusion zone."""
    margin_x = int(img_width * margin)
    margin_y = int(img_height * margin)
    # Draw semi-transparent overlay for margin zones
    overlay = display.copy()
    # Top margin
    cv2.rectangle(overlay, (0, 0), (img_width, margin_y), (128, 128, 128), -1)
    # Bottom margin
    cv2.rectangle(overlay, (0, img_height - margin_y), (img_width, img_height), (128, 128, 128), -1)
    # Left margin (excluding corners already drawn)
    cv2.rectangle(overlay, (0, margin_y), (margin_x, img_height - margin_y), (128, 128, 128), -1)
    # Right margin
    cv2.rectangle(overlay, (img_width - margin_x, margin_y), (img_width, img_height - margin_y), (128, 128, 128), -1)
    # Blend
    cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)


def nothing(x):
    pass


def main():
    parser = argparse.ArgumentParser(description="Calibrate AF overlay HSV thresholds")
    parser.add_argument("--image", help="Path to a saved frame (otherwise uses live camera)")
    parser.add_argument("--device", default=config.CAM_DEV, help="Camera device")
    args = parser.parse_args()

    if args.image:
        source_image = cv2.imread(args.image)
        if source_image is None:
            print(f"Error: Cannot read image {args.image}")
            sys.exit(1)
        cap = None
    else:
        dev = int(args.device) if args.device.isdigit() else args.device
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {args.device}")
            sys.exit(1)
        source_image = None

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)

    # Create trackbars
    cv2.createTrackbar("H Low", "Calibration", 0, 180, nothing)
    cv2.createTrackbar("H High", "Calibration", 180, 180, nothing)
    cv2.createTrackbar("S Low", "Calibration", 0, 255, nothing)
    cv2.createTrackbar("S High", "Calibration", 40, 255, nothing)
    cv2.createTrackbar("V Low", "Calibration", 200, 255, nothing)
    cv2.createTrackbar("V High", "Calibration", 255, 255, nothing)
    cv2.createTrackbar("Min Area", "Calibration", config.OVERLAY_MIN_AREA, 20000, nothing)
    cv2.createTrackbar("Max Area", "Calibration", config.OVERLAY_MAX_AREA, 50000, nothing)

    print("Controls:")
    print("  Trackbars: adjust HSV range and area constraints")
    print("  'p': print current threshold values")
    print("  '1': load white preset")
    print("  '2': load green preset")
    print("  '3': load gray preset")
    print("  'm': toggle edge margin overlay")
    print("  'q': quit")
    print("\nColor coding:")
    print("  Green  = Accepted (square, hollow, valid region)")
    print("  Red    = Rejected (wrong aspect ratio)")
    print("  Yellow = Rejected (in edge margin zone)")
    print("  Cyan   = Rejected (filled, not hollow)")

    show_margin_overlay = True

    while True:
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                continue
        else:
            frame = source_image.copy()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Read trackbar values
        h_low = cv2.getTrackbarPos("H Low", "Calibration")
        h_high = cv2.getTrackbarPos("H High", "Calibration")
        s_low = cv2.getTrackbarPos("S Low", "Calibration")
        s_high = cv2.getTrackbarPos("S High", "Calibration")
        v_low = cv2.getTrackbarPos("V Low", "Calibration")
        v_high = cv2.getTrackbarPos("V High", "Calibration")
        min_area = cv2.getTrackbarPos("Min Area", "Calibration")
        max_area = cv2.getTrackbarPos("Max Area", "Calibration")

        lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
        upper = np.array([h_high, s_high, v_high], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # Find and draw contours
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        display = frame.copy()
        img_height, img_width = frame.shape[:2]

        # Draw edge margin zone if enabled
        if show_margin_overlay:
            draw_edge_margin_zone(display, img_width, img_height,
                                  config.OVERLAY_EDGE_MARGIN)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if h == 0:
                continue
            aspect = w / h
            cx, cy = x + w // 2, y + h // 2

            # Check all rejection criteria
            is_square = config.OVERLAY_ASPECT_MIN <= aspect <= config.OVERLAY_ASPECT_MAX
            in_margin = is_in_edge_margin(cx, cy, img_width, img_height,
                                          config.OVERLAY_EDGE_MARGIN)
            is_hollow, fill_ratio = is_hollow_square(mask_clean, x, y, w, h,
                                                     config.OVERLAY_HOLLOW_RATIO)

            # Determine color based on rejection reason
            # Green = accepted, Red = wrong aspect, Yellow = margin, Cyan = filled
            if not is_square:
                color = (0, 0, 255)  # Red
                status = "RECT"
            elif in_margin:
                color = (0, 255, 255)  # Yellow
                status = "EDGE"
            elif not is_hollow:
                color = (255, 255, 0)  # Cyan
                status = f"FILL={fill_ratio:.0%}"
            else:
                color = (0, 255, 0)  # Green
                status = "OK"

            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display, f"A={int(area)} R={aspect:.2f} {status}",
                        (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1)

        cv2.imshow("Calibration", display)
        cv2.imshow("Mask", mask_clean)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            print(f"\nCurrent thresholds:")
            print(f"  H: ({h_low}, {h_high})")
            print(f"  S: ({s_low}, {s_high})")
            print(f"  V: ({v_low}, {v_high})")
            print(f"  Area: ({min_area}, {max_area})")
        elif key == ord("1"):  # White preset
            cv2.setTrackbarPos("H Low", "Calibration", 0)
            cv2.setTrackbarPos("H High", "Calibration", 180)
            cv2.setTrackbarPos("S Low", "Calibration", 0)
            cv2.setTrackbarPos("S High", "Calibration", 40)
            cv2.setTrackbarPos("V Low", "Calibration", 200)
            cv2.setTrackbarPos("V High", "Calibration", 255)
        elif key == ord("2"):  # Green preset
            cv2.setTrackbarPos("H Low", "Calibration", 35)
            cv2.setTrackbarPos("H High", "Calibration", 85)
            cv2.setTrackbarPos("S Low", "Calibration", 100)
            cv2.setTrackbarPos("S High", "Calibration", 255)
            cv2.setTrackbarPos("V Low", "Calibration", 100)
            cv2.setTrackbarPos("V High", "Calibration", 255)
        elif key == ord("3"):  # Gray preset
            cv2.setTrackbarPos("H Low", "Calibration", 0)
            cv2.setTrackbarPos("H High", "Calibration", 180)
            cv2.setTrackbarPos("S Low", "Calibration", 0)
            cv2.setTrackbarPos("S High", "Calibration", 40)
            cv2.setTrackbarPos("V Low", "Calibration", 100)
            cv2.setTrackbarPos("V High", "Calibration", 180)
        elif key == ord("m"):  # Toggle margin overlay
            show_margin_overlay = not show_margin_overlay
            print(f"Edge margin overlay: {'ON' if show_margin_overlay else 'OFF'}")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
