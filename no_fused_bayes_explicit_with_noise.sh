#!/bin/bash
#SBATCH -p gpu                  # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 16
#SBATCH --ntasks-per-node=1   # 4 tasks
#SBATCH --gpus-per-node=1     # 4 GPUs total on the node
#SBATCH -t 48:00:00                     # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt200353               # Specify project name
#SBATCH -J TEST                         # Specify job name

module load cuda 
module load Mamba             # Load the module that you want to use

conda activate myenv                    # Activate your environment
echo $CUDA_HOME
which python

export HF_HUB_CACHE="/home/csasnaru/.cache/huggingface"
export HF_HOME="/home/csasnaru/.cache/huggingface"
export HF_DATASETS_CACHE="/home/csasnaru/.cache/huggingface"

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 1. Define the path to your PyTorch libraries
export TORCH_LIB=/lustrefs/disk/home/csasnaru/.conda/envs/myenv/lib/python3.9/site-packages/torch/lib

# 2. Force the system to look here first (Library Path)
export LD_LIBRARY_PATH=$TORCH_LIB:/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

# 3. THE ATOMIC FIX: Preload the missing libraries into memory immediately
# This prevents Python from ever failing to find them, regardless of import order.
# Preload ONLY the core C++ and CUDA dependencies.
# We explicitly exclude libtorch_python.so to avoid the symbol error.
export LD_PRELOAD=$TORCH_LIB/libc10.so:$TORCH_LIB/libtorch.so:$TORCH_LIB/libtorch_cuda.so
# 4. Run the training (Pass all variables)
srun --export=ALL,LD_LIBRARY_PATH=$LD_LIBRARY_PATH,LD_PRELOAD=$LD_PRELOAD python simple_bayes_explicit_with_noise.py


# 1. Force the physical path to the libraries (use /lustrefs, NOT /home)
#export LD_LIBRARY_PATH=/lustrefs/disk/home/csasnaru/.conda/envs/myenv/lib/python3.9/site-packages/torch/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

# 2. Debugging: Check if the file is actually visible on the compute node
#srun --export=ALL bash -c "ls -l /lustrefs/disk/home/csasnaru/.conda/envs/myenv/lib/python3.9/site-packages/torch/lib/libc10.so && echo 'Library Found' || echo 'LIBRARY MISSING'"
#srun --export=ALL,LD_LIBRARY_PATH=$LD_LIBRARY_PATH python test.py
# 3. Run the training with explicit environment export
#srun --export=ALL,LD_LIBRARY_PATH=$LD_LIBRARY_PATH python cas_train.py
