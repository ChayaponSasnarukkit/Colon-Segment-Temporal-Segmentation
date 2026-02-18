#!/bin/bash
#SBATCH -p gpu-devel                  # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 16
#SBATCH --ntasks-per-node=1   # 4 tasks
#SBATCH --gpus-per-node=1     # 4 GPUs total on the node
#SBATCH -t 2:00:00                     # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt200353               # Specify project name
#SBATCH -J TEST                         # Specify job name

module load cuda 
module load Mamba/23.11.0-0             # Load the module that you want to use

conda activate myenv                    # Activate your environment
echo $CUDA_HOME
which python

export HF_HUB_CACHE="/home/csasnaru/.cache/huggingface"
export HF_HOME="/home/csasnaru/.cache/huggingface"
export HF_DATASETS_CACHE="/home/csasnaru/.cache/huggingface"

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

srun python extract_dinov3.py 
