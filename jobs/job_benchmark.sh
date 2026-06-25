#!/bin/bash
#SBATCH --job-name=gpu_kd_mix
#SBATCH --output=logs/benchmark_%j.log
#SBATCH --error=logs/benchmark_%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=99:00:00
#SBATCH --partition=gpu             # Cambia a 'gpu' (antes era main) :a100-80g 
#SBATCH --gres=gpu:1
#SBATCH --exclude=falcon[1-6]

# Activar el environment
source ~/projects/KD_proj/envs/py312/bin/activate
# Navegar al directorio
cd ~/projects/MixUP/KD-Mixup/
# Ejecutar el script
python -m scripts.benchmark --student mobilenetv2 --teacher resnet152v2 --temperature 2 --method mixup --alpha 1 --partial 1