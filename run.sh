#!/bin/bash -lT
#SBATCH -J TASC
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -n 1
#SBATCH -p gpu_7day
#SBATCH --gpus 4
#SBATCH --mem-per-gpu 24G
#SBATCH -w dscog023
#SBATCH -t 48:00:00

conda activate erie

python train_many.py