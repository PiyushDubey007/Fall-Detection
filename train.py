# train.py
from ultralytics import YOLO
import yaml
import os

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_YAML   = "D:/ad3/patient fall detection.v11i.yolov11/data.yaml"        # path to your dataset YAML
MODEL       = "yolo11n.pt"   # base model (n/s/m/l/x)
PROJECT     = "fall_detection"
RUN_NAME    = "v16"
EPOCHS      = 100
IMG_SIZE    = 640
BATCH       = 16                 # reduce to 8 if VRAM is tight
WORKERS     = 4
DEVICE      = 0                  # GPU index, or "cpu"
# ───────────────────────────────────────────────────────────────────────────────

def train():
    model = YOLO(MODEL)

    results = model.train(
        data      = DATA_YAML,
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH,
        workers   = WORKERS,
        device    = DEVICE,
        project   = PROJECT,
        name      = RUN_NAME,

        # Augmentation
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        flipud    = 0.3,
        fliplr    = 0.5,
        mosaic    = 1.0,
        mixup     = 0.1,

        # Regularisation
        dropout   = 0.1,
        weight_decay = 0.0005,

        # Optimizer
        optimizer = "AdamW",
        lr0       = 0.001,
        lrf       = 0.01,
        warmup_epochs = 3,

        # Save
        save          = True,
        save_period   = 10,       # checkpoint every N epochs
        exist_ok      = True,
        plots         = True,
    )

    print(f"\n✅ Training complete → {PROJECT}/{RUN_NAME}/weights/best.pt")
    return results


if __name__ == "__main__":
    train()