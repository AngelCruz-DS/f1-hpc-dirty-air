# -*- coding: utf-8 -*-
import os
import glob
import random
import shutil
from ultralytics import YOLO

IMAGES_DIR = "data/frames/images"
BASE_DIR = "data/frames"

for split in ["train", "val"]:
    os.makedirs(os.path.join(BASE_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, split, "labels"), exist_ok=True)

all_images = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
random.seed(42)
random.shuffle(all_images)

split_idx = int(len(all_images) * 0.8)
splits = {"train": all_images[:split_idx], "val": all_images[split_idx:]}

model = YOLO("yolov8n.pt")
VEHICLE_CLASSES = {2, 4, 7}

labeled_count = 0
for split, img_list in splits.items():
    for img_path in img_list:
        fname = os.path.basename(img_path)
        dst_img = os.path.join(BASE_DIR, split, "images", fname)
        dst_lbl = os.path.join(BASE_DIR, split, "labels", fname.replace(".jpg", ".txt"))
        shutil.copy(img_path, dst_img)

        results = model.predict(img_path, verbose=False, conf=0.25)
        lines = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in VEHICLE_CLASSES:
                    x, y, w, h = box.xywhn[0].tolist()
                    if y < 0.82:
                        lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        if lines:
            labeled_count += 1

yaml_content = """path: /tmp/$USER/f1_dataset
train: train/images
val: val/images

nc: 1
names: ['f1_car']
"""
with open(os.path.join(BASE_DIR, "dataset.yaml"), "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"Dataset dividido (80/20) y auto-etiquetado: {labeled_count}/{len(all_images)} imagenes con detecciones semilla.")
