"""Configuration constants for the Jetson Face AF Steering pipeline."""

import os

# --- Camera Capture ---
CAM_DEV = os.environ.get("CAM_DEV", "/dev/video0")
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
CAPTURE_FPS = 30

# --- Face Detection (SCRFD) ---
SCRFD_MODEL_PATH = "models/det_500m.onnx"
SCRFD_ENGINE_PATH = "models/det_500m.trt"
SCRFD_INPUT_SIZE = 640
SCRFD_CONF_THRESHOLD = 0.5
SCRFD_NMS_THRESHOLD = 0.4

# --- Face Recognition (MobileFaceNet) ---
RECOGNITION_MODEL_PATH = "models/w600k_mbf.onnx"
RECOGNITION_ENGINE_PATH = "models/w600k_mbf.trt"
RECOGNITION_INPUT_SIZE = 112
RECOGNITION_THRESHOLD = 0.4  # cosine similarity threshold
TARGET_EMBEDDINGS_PATH = "data/target_embeddings.npy"

# --- AF Overlay Detection (HSV thresholds) ---
# White AF square: high value, low saturation
OVERLAY_WHITE_H_RANGE = (0, 180)
OVERLAY_WHITE_S_RANGE = (0, 40)
OVERLAY_WHITE_V_RANGE = (200, 255)

# Green AF square: green hue, high saturation
OVERLAY_GREEN_H_RANGE = (35, 85)
OVERLAY_GREEN_S_RANGE = (100, 255)
OVERLAY_GREEN_V_RANGE = (100, 255)

# Gray AF square: low saturation, mid value
OVERLAY_GRAY_H_RANGE = (0, 180)
OVERLAY_GRAY_S_RANGE = (0, 40)
OVERLAY_GRAY_V_RANGE = (100, 180)

# Contour constraints for AF squares (always square, never rectangular)
OVERLAY_MIN_AREA = 400
OVERLAY_MAX_AREA = 10000
OVERLAY_ASPECT_MIN = 0.90  # minimum width/height ratio (near-square)
OVERLAY_ASPECT_MAX = 1.10  # maximum width/height ratio (near-square)

# Edge margin exclusion (ignore UI elements near frame edges)
OVERLAY_EDGE_MARGIN = 0.05  # Ignore squares in outer 5% of frame

# Hollow square detection (AF indicators are outlines, not filled shapes)
OVERLAY_HOLLOW_RATIO = 0.30  # Interior must be < 30% filled to be considered hollow

# Bounding box expansion for overlay search region
OVERLAY_BOX_EXPAND = 0.3  # 30% expansion

# --- Command Logic ---
INITIAL_CONFIRM_FRAMES = 3  # consistent frames before first command
STUCK_THRESHOLD = 5  # commands in same direction before trying alternative

# --- Nikon Controller ---
HEARTBEAT_INTERVAL_SEC = 0.5
