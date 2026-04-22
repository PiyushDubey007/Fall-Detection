# test_images.py
from ultralytics import YOLO
import cv2
import os

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH = r"D:/major project 1/runs/detect/fall_detection/v16/weights/best.pt"
IMAGE_PATH = r"D:/ad2/patient fall detection.v8i.yolov11/test/images/fall-01-cam1-rgb-152_png_jpg.rf.d67c63175a0fddf460061a27aceae27b.jpg"
OUTPUT_DIR = "test_results"
CONF       = 0.4
IOU        = 0.5
# ───────────────────────────────────────────────────────────────────────────────

FALL_CLASSES = {"fallen", "falling"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

frame = cv2.imread(IMAGE_PATH)

if frame is None:
    print(f"❌ Could not read image: {IMAGE_PATH}")
    exit()

results  = model(frame, conf=CONF, iou=IOU)[0]
boxes    = results.boxes

status = "SAFE"
color  = (0, 200, 0)

if boxes is not None and len(boxes):
    for box in boxes:
        cls_name = model.names[int(box.cls)]
        conf_val = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls_name in FALL_CLASSES:
            status = "FALL DETECTED"
            color  = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{cls_name} {conf_val:.2f}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Status banner
cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
cv2.putText(frame, status, (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

# Save output
fname    = os.path.basename(IMAGE_PATH)
out_path = os.path.join(OUTPUT_DIR, fname)
cv2.imwrite(out_path, frame)
print(f"[{status}] {fname}")
print(f"✅ Saved to → {out_path}")

# Popup — press any key to close
cv2.imshow("Fall Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()