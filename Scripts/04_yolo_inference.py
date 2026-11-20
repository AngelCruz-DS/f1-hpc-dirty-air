import sys
import os
import cv2
import pandas as pd
from ultralytics import YOLO

video_chunk = sys.argv[1] if len(sys.argv) > 1 else "data/raw_video/silverstone_lec_onboard.mp4"
output_parquet = sys.argv[2] if len(sys.argv) > 2 else "data/telemetry/yolo_detections.parquet"
model_path = sys.argv[3] if len(sys.argv) > 3 else "yolov8n.pt"

FOCAL_LENGTH_PX = 1100.0
CAR_REAL_WIDTH_M = 2.0

model = YOLO(model_path)
cap = cv2.VideoCapture(video_chunk)
fps = cap.get(cv2.CAP_PROP_FPS) or 50.0

records = []
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, verbose=False, conf=0.45)
    min_distance_m = None
    best_conf = 0.0

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w_px = x2 - x1
            if w_px > 5:
                dist_m = (FOCAL_LENGTH_PX * CAR_REAL_WIDTH_M) / w_px
                if min_distance_m is None or dist_m < min_distance_m:
                    min_distance_m = round(dist_m, 2)
                    best_conf = round(float(box.conf[0]), 3)

    records.append({
        "frame": frame_idx,
        "video_time_sec": round(frame_idx / fps, 3),
        "rival_distance_m": min_distance_m,
        "confidence": best_conf,
        "dirty_air_zone": bool(min_distance_m is not None and 5.0 <= min_distance_m <= 35.0)
    })
    frame_idx += 1

cap.release()
pd.DataFrame(records).to_parquet(output_parquet, index=False)
print(f"Inferencia completada ({frame_idx} frames) -> {output_parquet}")
