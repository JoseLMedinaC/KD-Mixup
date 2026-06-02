#!/bin/bash
#SBATCH --job-name=gpu_kd_mix
#SBATCH --output=logs/gpu_mixup_%j.log
#SBATCH --error=logs/gpu_mixup_%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=99:00:00
#SBATCH --partition=gpu             # Cambia a 'gpu' (antes era main)
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --exclude=falcon[1-6]

# Activar el environment
source ~/projects/KD_proj/envs/py312/bin/activate
# Navegar al directorio
cd ~/projects/MixUP/KD-Mixup/
# Ejecutar el script
#python ./scripts/multiple_teacher.py
#python -m scripts.student_train_classic --teacher convnextlarge
#python -m scripts.student_train_classic --teacher vitbase
#python -m scripts.student_train_classic --teacher resnet152v2
# Correr todas las temperaturas para convnexttiny
python -m scripts.student_train_mixup --teacher vitbase --temperature 2 --student resnet50
#python -m scripts.student_train_mixup --teacher convnexttiny --temperature 2 --student resnet50
