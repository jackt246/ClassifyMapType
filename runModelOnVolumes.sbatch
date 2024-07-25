#!/bin/bash
#SBATCH --cpus-per-task=20
#SBATCH --time=1200
#SBATCH --mem=250G

srun python runModelOnVolumes.py