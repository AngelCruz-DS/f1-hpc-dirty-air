import cv2
import os

OUTPUT_DIR = "data/frames/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

videos = [
    ("data/raw_video/china_sprint_ham_lec.mp4", 1.0, "china"),
    ("data/raw_video/leclerc_onboard_collection.mp4", 4.0, "col")
]

total_saved = 0
for video_path, sec_step, prefix in videos:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(fps * sec_step))
    count, saved = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            filename = os.path.join(OUTPUT_DIR, f"{prefix}_{saved:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1
            total_saved += 1
        count += 1
    cap.release()
    print(f"{prefix}: {saved} fotogramas extraídos.")

print(f"✅ Total de fotogramas listos en {OUTPUT_DIR}: {total_saved}")
