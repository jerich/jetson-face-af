# Jetson Face AF

Real-time face-recognition autofocus steering for Nikon Z8 on NVIDIA Jetson.

This project identifies a target person via face recognition on a camera's HDMI output, detects the autofocus overlay state, and sends USB commands to steer the camera's autofocus point to the target. It uses TensorRT-accelerated neural networks for face detection and recognition, running at 30fps on NVIDIA Jetson Orin hardware with a Nikon Z8 camera.

## Overview

Nikon's face-recognition autofocus works great. But, it doesn't know me, it doesn't know my family. What if I could guide it to lock onto a specific face I was interested in?

The use case I had in mind creating this was shooting my kid on a stage at a year-end school performance. There's many kids on stage together, moving around, and I'm constantly hitting the Left/Right arrow keys to steer the Nikon AF system back to the face I want.

So, most importantly, I don't want to compromise Nikon's shooting ergonomics, and especially don't want to be shooting from a laptop. I want the viewfinder up to my eye and my hands on the buttons. That's where the MC-N10 remote grip comes in. It's an excellent accessory and I recommend that any Nikon Z-system user that shoots a lot of video to pick one up. I noticed it uses the USB-C connection to the camera in MTP/PTP mode, and wondered if I could use that for my system while keeping the user controls on the camera active. This project is the end result of that experimentation.

After using [Ataradov USB sniffer](https://github.com/ataradov/usb-sniffer) to grab the USB packets sent between the Camera and the MC-N10, I used ChatGPT to start building this, then finished it off with Claude Code.

The current version uses pretraining to detect a target face. I trained it on folder of 100+ images in a variety of settings and conditions and the face recognition works quite well. Analysis of the Nikon AF UI elements seems to be working, and the USB control can send the correct steering commands to the camera. It needs more testing, but it works as a proof of concept.

What would make it better? If Nikon provided the live view output over USB or wifi without disabling the on-camera controls, and additionally provided an incoming control to specify an (x,y) focus point, the same as a touch event on the camera's rear screen; that would be faster and more accurate than trying to send Left/Right commands to steer the AF system.

Actually, wifi would be best, because that would allow all of this to take place in an iPhone app. It coud grab frames from live view, and iOS already learns the faces of people that you take pictures of most often, so the face recognition is already taken care of on the phone. Then it's just a matter of sending the focus point back to the camera and wrapping it up with a tidy UI.

An autofocus helper loop like this could be a major differentiator for any of the major camera companies. The intent is not to replace the camera’s autofocus system, but to expose it to external software, AI or otherwise, to interact with the camera as a user does, operating alongside the user in real time. Interchageable lens cameras are long life products; they're never going to be on a yearly release cycle and their internal processing will always lag behind the latest iPhone or something like the Nvidia Jetson. Opening up a processing loop around an already excellent AF system would enable faster innovation and make existing cameras better year after year.

Imagine a future version with an LLM running alongside the camera: "Stay focused on the player in jersey 77" or "Focus on the woman in the green dress" or even "Focus on James during his solo." I can picture an app with AF recipes, just like Nikon's picture control recipes in the Nikon Image Space app right now. Or even extend it to shutter control: "Start one second of burst shooting when the squirrel grabs the peanut." It can be done with the Nikon hardware that's out there today; it just needs a small change to the camera mode and a new API.

## Features

- **Face detection** using SCRFD 2.5G with 5-point landmarks
- **Face recognition** using MobileFaceNet embeddings (512-dim)
- **AF overlay detection** via HSV color analysis (tracked/active/detected states)
- **USB command steering** via MC-N10 remote emulation
- **Configurable thresholds** for recognition sensitivity and debouncing
- **Debug overlay mode** for visualization and tuning

## Requirements

- NVIDIA Jetson Orin (JetPack 6.x / L4T r36.4.0)
- Nikon Z-series mirrorless camera compatible with the MC-N10 remote grip (I used a Z8)
- HDMI capture card (1080p30 output to `/dev/video0`, I used an Elgato CamLink4K)
- Python 3.10+
- [nikon-usb-control](https://github.com/jerich/nikon-usb-control) package installed at `~/nikon-usb-control/`

## Quick Start

```bash
# Clone and setup
git clone https://github.com/your-username/jetson-face-af.git
cd jetson-face-af
bash ./setup_host.sh

# Copy target face photos and enroll
cp /path/to/photos/*.jpg data/target_images/
python3 training/enroll_face.py --images data/target_images/

# Run with debug overlay
python3 main.py --debug
```

## Usage

```bash
python3 main.py                    # Full pipeline with Nikon control
python3 main.py --debug            # With video display overlay
python3 main.py --debug --no-nikon # Vision-only (no USB control)
python3 main.py --debug --dry-run  # Log commands without sending
```

| Flag | Description |
|------|-------------|
| `--debug` | Show live video with detection overlays |
| `--no-nikon` | Disable USB camera control (vision pipeline only) |
| `--dry-run` | Log steering commands without sending to camera |

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RECOGNITION_THRESHOLD` | 0.4 | Cosine similarity threshold for target match (lower = stricter) |
| `INITIAL_CONFIRM_FRAMES` | 3 | Consecutive frames before sending first command (debounce) |
| `STUCK_THRESHOLD` | 5 | Commands in same direction before trying alternative |
| `SCRFD_CONF_THRESHOLD` | 0.5 | Face detection confidence threshold |
| `OVERLAY_*` | varies | HSV ranges for AF square detection |

Adjust `RECOGNITION_THRESHOLD` if you get false positives (raise it) or the target isn't recognized (lower it). Use `tools/calibrate_overlay.py` to tune HSV thresholds for your specific camera's AF overlay colors.

## Project Structure

```
main.py                      # Entry point
config.py                    # All tunable constants
pipeline/
  capture.py                 # Threaded V4L2 frame grabber
  face_detect.py             # SCRFD face detection (TensorRT)
  face_recognize.py          # MobileFaceNet embeddings (TensorRT)
  overlay_detect.py          # HSV-based AF overlay classification
  command_logic.py           # Steering decision engine
camera_control/
  nikon_controller.py        # Threaded MC-N10 USB command sender
utils/
  trt_inference.py           # ONNX to TensorRT builder + runner
  face_align.py              # 5-point landmark alignment
training/
  enroll_face.py             # Generate target embeddings
tools/
  calibrate_overlay.py       # Interactive HSV threshold tuner
models/                      # ONNX and TensorRT model files
data/
  target_images/             # Photos of target person
  target_embeddings.npy      # Generated by enroll_face.py
```

## Tests

```bash
python3 tests/test_overlay_detect.py
python3 tests/test_command_logic.py
```
