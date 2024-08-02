#!/bin/bash
#SBATCH --cpus-per-gpu=10
#SBATCH --time=1200
#SBATCH --mem=150G
#SBATCH --gres=gpu:a100:1

srun python trainingVolumes.py