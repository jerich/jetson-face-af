import os, time, cv2
from ultralytics import YOLO

CAM_DEV = os.environ.get("CAM_DEV", "/dev/video0")
VIEW_W, VIEW_H = 1920, 1080    # on-screen window size (scales output)
REQ_W, REQ_H  = 1280, 720      # ask the driver for this; may be ignored by UVC

print(f"[info] opening {CAM_DEV}")
cap = cv2.VideoCapture(CAM_DEV)
if not cap.isOpened():
    raise SystemExit(f"[error] cannot open {CAM_DEV}")

# ask for a smaller capture size (fallback if driver ignores)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  REQ_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQ_H)
cap.set(cv2.CAP_PROP_FPS, 30)

model = YOLO("yolo11n.pt")     # tiny model; try yolo11s.pt if you want more accuracy
t0, n = time.time(), 0

while True:
    ok, frame = cap.read()
    if not ok:
        print("[warn] read failed"); break

    # run detector (auto-uses CUDA in this Jetson container)
    res = model.predict(source=frame, imgsz=640, conf=0.5, verbose=False)
    out = res[0].plot()

    # scale the preview (independent of capture size)
    out_disp = cv2.resize(out, (VIEW_W, VIEW_H), interpolation=cv2.INTER_AREA)

    n += 1
    if n % 10 == 0:
        fps = n / (time.time() - t0)
        cv2.putText(out_disp, f"{fps:.1f} FPS", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

    cv2.imshow("YOLO11 (q to quit)", out_disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
