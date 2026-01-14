import os, cv2, torch
import torchvision.transforms as T
import torchvision.models as models

CAM_DEV = os.environ.get("CAM_DEV", "/dev/video0")
VIEW_W, VIEW_H = 1920, 1080
REQ_W, REQ_H = 1280, 720

# load pretrained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval().cuda()

# ImageNet labels
import urllib.request
labels = urllib.request.urlopen(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
).read().decode().splitlines()

# Preprocessing
transform = T.Compose([
    T.ToPILImage(),
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

print(f"[info] opening {CAM_DEV}")
cap = cv2.VideoCapture(CAM_DEV)
if not cap.isOpened():
    raise SystemExit(f"[error] cannot open {CAM_DEV}")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQ_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQ_H)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[warn] read failed"); break

    # preprocess
    img = transform(frame).unsqueeze(0).cuda()

    # inference
    with torch.no_grad():
        out = model(img)
        pred = out.argmax(1).item()
        label = labels[pred]

    # scale and display
    out_disp = cv2.resize(frame, (VIEW_W, VIEW_H), interpolation=cv2.INTER_AREA)
    cv2.putText(out_disp, label, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("ResNet18 Classification (q to quit)", out_disp)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
