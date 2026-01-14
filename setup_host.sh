#!/bin/bash
set -e

echo "=== Jetson Face AF — Host Setup ==="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- Install Python dependencies ---
echo "[1/4] Installing Python dependencies..."
pip3 install --user -r requirements.txt

# --- Check nikon-usb-control ---
echo "[2/4] Checking nikon-usb-control..."
if [ -f "$HOME/nikon-usb-control/nikon-usb-control/Nikon_mc_n10.py" ]; then
    echo "  Found Nikon_mc_n10.py (imported via sys.path at runtime)"
else
    echo "WARNING: ~/nikon-usb-control/nikon-usb-control/Nikon_mc_n10.py not found."
    echo "  Camera control will not work without it."
fi

# --- Download models ---
echo "[3/4] Downloading ONNX models..."
mkdir -p models

PACK_URL="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"

if [ ! -f models/det_500m.onnx ] || [ ! -f models/w600k_mbf.onnx ]; then
    echo "  Downloading InsightFace buffalo_sc model pack..."
    wget -q --show-progress -O /tmp/buffalo_sc.zip "$PACK_URL"
    echo "  Extracting models..."
    unzip -o -j /tmp/buffalo_sc.zip "*.onnx" -d models/
    rm -f /tmp/buffalo_sc.zip
    echo "  Extracted: $(ls models/*.onnx 2>/dev/null)"
else
    echo "  Models already exist, skipping."
fi

# --- Verify and build TensorRT engines ---
echo "[4/4] Verifying models and building TensorRT engines..."
python3 -c "
import onnx
import config

# Verify ONNX models
for name, path in [('SCRFD', config.SCRFD_MODEL_PATH), ('MobileFaceNet', config.RECOGNITION_MODEL_PATH)]:
    print(f'  Checking {name}: {path}')
    model = onnx.load(path)
    inputs = [i.name for i in model.graph.input]
    outputs = [o.name for o in model.graph.output]
    print(f'    Inputs: {inputs}')
    print(f'    Outputs: {outputs}')
    if not outputs:
        print(f'    WARNING: No outputs defined in ONNX graph!')

print()
from utils.trt_inference import build_engine

print('  Building SCRFD engine...')
s = config.SCRFD_INPUT_SIZE
build_engine(config.SCRFD_MODEL_PATH, config.SCRFD_ENGINE_PATH, fp16=True,
             input_shape=(1, 3, s, s))
print('  SCRFD engine ready.')

print('  Building MobileFaceNet engine...')
s = config.RECOGNITION_INPUT_SIZE
build_engine(config.RECOGNITION_MODEL_PATH, config.RECOGNITION_ENGINE_PATH, fp16=True,
             input_shape=(1, 3, s, s))
print('  MobileFaceNet engine ready.')
"

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Copy target face photos to data/target_images/"
echo "  2. Run: python3 training/enroll_face.py"
echo "  3. Run: python3 main.py --debug --no-nikon"
