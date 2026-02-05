#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -t 06:00:00
#SBATCH -A lt200353
#SBATCH -J TEST

# 1. Clear previous environment clutter
module purge
module load Mamba
module load cuda  # Ensures basic driver access

# 2. Activate Environment
conda activate myenv
echo "Environment Activated: $CONDA_DEFAULT_ENV"

# 5. Run the 
# srun python ASformer_cas_train.py
srun python eval_asformer.py
