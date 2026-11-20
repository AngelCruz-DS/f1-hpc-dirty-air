import subprocess
import os

INPUT_VIDEO = "data/raw_video/silverstone_lec_onboard.mp4"
CHUNKS_DIR = "data/raw_video/chunks"
SEGMENT_SECONDS = 300

os.makedirs(CHUNKS_DIR, exist_ok=True)

cmd = [
    "ffmpeg", "-i", INPUT_VIDEO,
    "-c", "copy", "-map", "0",
    "-segment_time", str(SEGMENT_SECONDS),
    "-f", "segment", "-reset_timestamps", "1",
    os.path.join(CHUNKS_DIR, "chunk_%03d.mp4")
]

print("Particionando video para distribución en nodos GPU...")
subprocess.run(cmd, check=True)
print(f"Segmentos generados en {CHUNKS_DIR}")
