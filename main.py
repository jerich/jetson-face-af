"""Jetson Face AF Steering — main entry point."""

import argparse
import logging
import time
import sys

import cv2
import numpy as np

import config
from pipeline.capture import FrameGrabber
from pipeline.face_detect import FaceDetector
from pipeline.face_recognize import FaceRecognizer
from pipeline.overlay_detect import detect_af_state, AFState
from pipeline.command_logic import CommandLogic, FaceInfo, Command
from camera_control.nikon_controller import NikonAFController


def parse_args():
    parser = argparse.ArgumentParser(description="Jetson Face AF Steering")
    parser.add_argument("--debug", action="store_true",
                        help="Show annotated video display")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log commands without sending to camera")
    parser.add_argument("--no-nikon", action="store_true",
                        help="Disable Nikon controller entirely")
    parser.add_argument("--device", default=None,
                        help="Camera device (overrides CAM_DEV env)")
    parser.add_argument("--embeddings", default=None,
                        help="Path to target embeddings .npy file")
    return parser.parse_args()


def draw_debug(frame: np.ndarray, faces: list[FaceInfo],
               command: Command, fps: float) -> np.ndarray:
    """Annotate frame with face boxes, AF state, and command info."""
    display = frame.copy()

    for face in faces:
        x1, y1, x2, y2 = face.box.astype(int)

        # Color based on target/AF state
        if face.is_target:
            color = (0, 255, 0)  # Green for target
        elif face.af_state in (AFState.TRACKED, AFState.ACTIVE_FOCUS):
            color = (255, 255, 0)  # Cyan for AF face
        else:
            color = (128, 128, 128)  # Gray for others

        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        # Label
        label_parts = []
        if face.is_target:
            label_parts.append(f"TARGET({face.similarity:.2f})")
        label_parts.append(face.af_state.value)

        label = " | ".join(label_parts)
        cv2.putText(display, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Command and FPS overlay
    info = f"FPS: {fps:.1f} | CMD: {command.value}"
    cv2.putText(display, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return display


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("main")

    # Initialize components
    logger.info("Initializing face detector...")
    detector = FaceDetector()

    logger.info("Initializing face recognizer...")
    recognizer = FaceRecognizer(
        target_embeddings_path=args.embeddings or config.TARGET_EMBEDDINGS_PATH
    )

    logic = CommandLogic()

    # Start camera capture
    device = args.device or config.CAM_DEV
    grabber = FrameGrabber(device=device)
    grabber.start()
    logger.info("Camera capture started")

    # Start Nikon controller
    nikon = None
    if not args.no_nikon:
        nikon = NikonAFController(dry_run=args.dry_run)
        try:
            nikon.start()
        except Exception as e:
            logger.error(f"Nikon controller failed to start: {e}")
            logger.info("Continuing without camera control")
            nikon = None

    if args.debug:
        cv2.namedWindow("Face AF", cv2.WINDOW_NORMAL)

    logger.info("Pipeline running. Press 'q' to quit.")

    frame_count = 0
    fps_start = time.time()
    fps = 0.0

    try:
        while True:
            frame = grabber.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Face detection
            boxes, scores, landmarks = detector.detect(frame)

            # Build face info list
            faces = []
            for i in range(len(boxes)):
                # Face recognition
                embedding = recognizer.get_embedding(frame, landmarks[i])
                is_target, similarity = recognizer.is_target(embedding)

                # AF overlay detection
                af_state = detect_af_state(frame, boxes[i])

                faces.append(FaceInfo(
                    box=boxes[i],
                    is_target=is_target,
                    similarity=similarity,
                    af_state=af_state,
                ))

            # Command logic
            command = logic.update(faces)

            # Send command to Nikon
            if nikon is not None and nikon.connected:
                nikon.send_command(command)

            if command != Command.NONE:
                logger.info(f"Command: {command.value}")

            # FPS calculation
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            # Debug display
            if args.debug:
                display = draw_debug(frame, faces, command, fps)
                cv2.imshow("Face AF", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        logger.info("Shutting down...")
        grabber.stop()
        if nikon is not None:
            nikon.stop()
        if args.debug:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
