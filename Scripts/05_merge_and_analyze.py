import pandas as pd

TEL_PATH = "data/telemetry/2026_Silverstone_LEC_telemetry.parquet"
YOLO_PATH = "data/telemetry/yolo_detections.parquet"
OUTPUT_MERGED = "data/telemetry/2026_Silverstone_LEC_multimodal.parquet"

df_tel = pd.read_parquet(TEL_PATH)
df_yolo = pd.read_parquet(YOLO_PATH)

df_tel["tel_sec"] = df_tel["SessionTime"].dt.total_seconds()
t0_offset = df_tel["tel_sec"].min()
df_yolo["tel_sec"] = df_yolo["video_time_sec"] + t0_offset

df_merged = pd.merge_asof(
    df_yolo.sort_values("tel_sec"),
    df_tel.sort_values("tel_sec"),
    on="tel_sec",
    direction="nearest"
)

df_merged.to_parquet(OUTPUT_MERGED, index=False)
print(f"Dataset multimodal fusionado guardado en: {OUTPUT_MERGED}")
