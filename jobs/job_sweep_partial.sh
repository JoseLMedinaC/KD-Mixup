#!/bin/bash
#SBATCH --job-name=kd_mix
#SBATCH --output=logs/kd_mix_%A_%a.log
#SBATCH --error=logs/kd_mix_%A_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=99:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source ~/projects/KD_proj/envs/py312/bin/activate
cd ~/projects/MixUP/KD-Mixup/

TEACHER="resnet152v2"
STUDENT="mobilenetv2"
ALPHA=0.5
PARTIAL=1
# Ejecutar secuencialmente, no en paralelo
python -m scripts.train_partial_mixup \
    --teacher $TEACHER \
    --student $STUDENT \
    --alpha $ALPHA \
    --partial $PARTIAL