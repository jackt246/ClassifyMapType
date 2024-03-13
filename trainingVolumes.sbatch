#!/bin/bash
#SBATCH --cpus=20
#SBATCH --time=1200
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1

srun python trainingVolumes.py