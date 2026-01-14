#!/usr/bin/env bash
set -euo pipefail

# ---------- config ----------
IMG="dustynv/l4t-pytorch:r36.4.0"   # JetPack 6.x PyTorch image
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # mount this repo
CAM_DEV="${CAM_DEV:-/dev/video0}"   # override: CAM_DEV=/dev/video1 ./run_pytorch.sh
# ----------------------------

# allow the container root to open X windows
xhost +local:root >/dev/null 2>&1 || true

# collect available video nodes (map a few common ones)
MAP_DEVICES=()
for n in 0 1 2 3; do
  if [[ -e "/dev/video${n}" ]]; then
    MAP_DEVICES+=(--device "/dev/video${n}:/dev/video${n}")
  fi
done

# helpful flags:
# --network host : simpler networking
# --ipc host     : better perf for DL libs
# --ulimit memlock: allow pinned memory
docker run -it --rm --runtime nvidia \
  --network host --ipc=host --ulimit memlock=-1 \
  "${MAP_DEVICES[@]}" \
  -e DISPLAY="${DISPLAY}" -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "${WORKSPACE}:/workspace" \
  -e CAM_DEV="${CAM_DEV}" \
  ${IMG} /bin/bash
