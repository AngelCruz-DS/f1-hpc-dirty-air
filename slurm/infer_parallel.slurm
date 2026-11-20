#!/bin/bash
#SBATCH --job-name=f1_yolo_infer
#SBATCH --partition=gpu
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=/tmp/f1_infer_%A_%a.out

CHUNK_ID=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
mkdir -p /tmp/$USER/work

cp ~/f1-hpc-dirty-air/data/raw_video/chunks/chunk_${CHUNK_ID}.mp4 /tmp/$USER/work/
cp ~/f1-hpc-dirty-air/models/f1_yolov8_best.pt /tmp/$USER/work/
cp ~/f1-hpc-dirty-air/Scripts/04_yolo_inference.py /tmp/$USER/work/

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

python /tmp/$USER/work/04_yolo_inference.py \
    /tmp/$USER/work/chunk_${CHUNK_ID}.mp4 \
    /tmp/$USER/work/detections_${CHUNK_ID}.parquet \
    /tmp/$USER/work/f1_yolov8_best.pt

cp /tmp/$USER/work/detections_${CHUNK_ID}.parquet ~/f1-hpc-dirty-air/data/telemetry/
rm -rf /tmp/$USER/work
