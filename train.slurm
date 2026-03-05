#!/bin/bash
#SBATCH --job-name=mistral_cipher
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=l4
#SBATCH --cpus-per-task=32    # Keeps dataloaders fed across 4 GPUs
#SBATCH --mem=64G            # Crucial for FSDP CPU offloading/gathering
#SBATCH --gres=gpu:l4:4

set -e

# Update to your actual Mistral project directory
PROJECT_DIR=/ceph/project/SW10-CausalLM/Mistral
cd $PROJECT_DIR

mkdir -p logs
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "Job started on $(hostname) at $(date)" | tee logs/train_live_${SLURM_JOB_ID}.log

# Force exactly the allocated GPUs to be visible to avoid NCCL deadlock oversubscription
export CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print $1","$2","$3","$4}')

nvidia-smi | tee -a logs/train_live_${SLURM_JOB_ID}.log

export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=8

# NCCL Optimizations for FSDP on L4s
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_P2P_DISABLE=1 # Uncomment ONLY if you experience PCIe P2P hangs on your specific cluster

# Ensure environment is synced
uv pip install -e .

# Mistral relies heavily on FA2 for sliding window attention.
# We retain your excellent direct wheel compilation strategy.
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# Ensure master_port is randomized slightly to prevent port collision on shared nodes
MASTER_PORT=$((10000 + $RANDOM % 20000))

# Launch with torchrun. Hugging Face Trainer automatically detects the distributed 
# environment spawned by torchrun and applies the FSDP arguments defined in train.py
uv run torchrun \
    --nproc_per_node=4 \
    --master_port=$MASTER_PORT \
    src/train.py 2>&1 | tee -a logs/train_live_${SLURM_JOB_ID}.log

echo "Job finished at $(date)" | tee -a logs/train_live_${SLURM_JOB_ID}.log