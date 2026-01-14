#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSTRAINTS="${SCRIPT_DIR}/constraints.txt"

# force public PyPI (avoids the NGC mirror that sometimes fails)
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_EXTRA_INDEX_URL=https://pypi.ngc.nvidia.com

python3 -m pip install --upgrade pip --no-cache-dir

# start clean (ignore errors if not installed yet)
python3 -m pip uninstall -y ultralytics opencv-python numpy || true

# install with constraints
python3 -m pip install --no-cache-dir -c "${CONSTRAINTS}" numpy
python3 -m pip install --no-cache-dir -c "${CONSTRAINTS}" opencv-python ultralytics

python3 - <<'PY'
import numpy, cv2, ultralytics, torch
print("numpy:", numpy.__version__)
print("opencv:", cv2.__version__)
print("ultralytics:", ultralytics.__version__)
print("cuda?", torch.cuda.is_available())
PY
