#!/bin/bash
#SBATCH --job-name=gpu_kd_mix
#SBATCH --output=logs/gpu_mixup_%j.log
#SBATCH --error=logs/gpu_mixup_%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=99:00:00
#SBATCH --partition=gpu             # Cambia a 'gpu' (antes era main) exclude=falcon[1-6]
#SBATCH --gres=gpu:1


# Activar el environment
source ~/projects/KD_proj/envs/py312/bin/activate
# Navegar al directorio
cd ~/projects/MixUP/KD-Mixup/
python -m scripts.train_normal --model resnet18

