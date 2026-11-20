#!/bin/bash
#SBATCH --job-name=f1_yolo_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --output=/tmp/f1_train_%j.out
#SBATCH --error=/tmp/f1_train_%j.err

mkdir -p /tmp/$USER/f1_dataset
cp -r ~/f1-hpc-dirty-air/data/frames/* /tmp/$USER/f1_dataset/

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

yolo detect train data=/tmp/$USER/f1_dataset/dataset.yaml model=yolov8n.pt epochs=50 imgsz=640 device=0,1,2,3 project=/tmp/$USER/runs name=f1_tcam

cp /tmp/$USER/runs/f1_tcam/weights/best.pt ~/f1-hpc-dirty-air/models/f1_yolov8_best.pt
rm -rf /tmp/$USER/f1_dataset /tmp/$USER/runs
