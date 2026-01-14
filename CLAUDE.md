# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time face-recognition-based autofocus steering system for NVIDIA Jetson + Nikon Z8. The main pipeline (`main.py`) identifies a target person via face recognition on the camera's HDMI output, detects AF overlay state, and sends multi-selector commands via USB (MC-N10 emulation) to steer autofocus to the target.

Legacy demo scripts (`yolo_cam.py`, `classify_cam.py`) remain for reference.

## Running

```bash
# One-time setup: download models, build TRT engines, install deps
bash ./setup_host.sh

# Enroll target face (provide photos in data/target_images/)
python3 training/enroll_face.py --images data/target_images/

# Run the AF steering pipeline
python3 main.py                    # Full pipeline with Nikon control
python3 main.py --debug            # With video display overlay
python3 main.py --debug --no-nikon # Vision-only (no USB control)
python3 main.py --debug --dry-run  # Log commands without sending

# Run tests
python3 tests/test_overlay_detect.py
python3 tests/test_command_logic.py
```

## Environment

- **Target hardware:** NVIDIA Jetson Orin (JetPack 6.x / L4T r36.4.0)
- **Python deps:** TensorRT, PyCUDA, OpenCV, NumPy, onnx
- **External package:** `~/nikon-usb-control/nikon-usb-control/` (installed as editable)
- **Constraints:** `numpy<2`, `opencv-python<4.9` for Jetson compatibility

## Architecture

```
pipeline/capture.py          → Threaded V4L2 frame grabber (1920x1080 @ 30fps)
pipeline/face_detect.py      → SCRFD 2.5G face detection via TensorRT
pipeline/face_recognize.py   → MobileFaceNet embeddings via TensorRT
pipeline/overlay_detect.py   → HSV color + contour AF overlay classification
pipeline/command_logic.py    → Decision engine with debouncing + stuck detection
camera_control/nikon_controller.py → Threaded MC-N10 USB command sender
utils/trt_inference.py       → Generic ONNX→TRT builder + inference runner
utils/face_align.py          → 5-point landmark affine alignment (112x112)
config.py                    → All tunable constants
main.py                      → Entry point, orchestrates all components
```

Pipeline flow:
```
[Capture Thread]           [Main Loop]                [Nikon Thread]
  V4L2 read → buffer    →  SCRFD detect faces     →   heartbeat 500ms
                            MobileFaceNet embed         poll command queue
                            overlay_detect state        send left/right/fn1
                            command_logic decide
                            post to queue ────────────→
```

## Key Configuration (`config.py`)

- `CAM_DEV`: Camera device (env var, default `/dev/video0`)
- `CAPTURE_WIDTH/HEIGHT`: 1920x1080
- `SCRFD_INPUT_SIZE`: 640, `SCRFD_CONF_THRESHOLD`: 0.5
- `RECOGNITION_THRESHOLD`: 0.4 (cosine similarity for target match)
- `INITIAL_CONFIRM_FRAMES`: 3 (debounce before first command)
- `STUCK_THRESHOLD`: 5 (same-direction commands before trying alternative)
- `OVERLAY_*`: HSV ranges for white/green/gray AF squares
- `HEARTBEAT_INTERVAL_SEC`: 0.5

## ML Models

| Model | File | Purpose | Input | Output |
|-------|------|---------|-------|--------|
| SCRFD 2.5G (kps) | `models/scrfd_2.5g_bnkps.{onnx,trt}` | Face detection | 640x640 | Boxes + 5-pt landmarks |
| MobileFaceNet | `models/w600k_mbf.{onnx,trt}` | Face embeddings | 112x112 aligned | 512-dim vector |

## Tools

- `tools/calibrate_overlay.py` — Interactive HSV threshold tuner with trackbars
- `training/enroll_face.py` — Generate target embeddings from photos
