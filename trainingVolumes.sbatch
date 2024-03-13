#!/bin/bash
#SBATCH --cpus-per-gpu=20
#SBATCH --time=1200
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1

srun python trainingVolumes.py