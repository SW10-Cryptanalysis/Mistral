#!/bin/bash
#SBATCH --job-name=mistral_cipher_1gpu
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=l4
#SBATCH --cpus-per-task=8      # Reduced since you are only feeding 1 GPU now
#SBATCH --mem=32G            # Reduced memory, but keep high if your dataloader needs it
#SBATCH --gres=gpu:l4:1      # CHANGED: Request exactly 1 L4 GPU

set -e

# Update to your actual Mistral project directory
PROJECT_DIR=/ceph/project/SW10-CausalLM/Mistral
cd $PROJECT_DIR

# ==============================================================================
# DYNAMIC AUTO-CHAINING LOGIC
# ==============================================================================
# 1. Break the chain if training successfully finished in a previous run
if [ -d "outputs/final_model" ]; then
    echo "Success: 'outputs/final_model' found. Training is completely finished."
    echo "Breaking the chain. No further jobs will be submitted."
    exit 0
fi

# 2. Submit the NEXT job to the queue, dependent on THIS job finishing or timing out.
# $SLURM_JOB_ID is the current job. $0 refers to this exact script file.
echo "Dynamically queuing the next 12-hour continuation job..."
sbatch --dependency=afterany:$SLURM_JOB_ID $0
# ==============================================================================

mkdir -p logs
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "Job started on $(hostname) at $(date)" | tee logs/train_live_${SLURM_JOB_ID}.log

# CHANGED: Simplified. Slurm will handle isolating the 1 GPU it gave you.
# We just ensure it grabs the first available ID if multiple are somehow exposed.
export CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print $1}')

nvidia-smi | tee -a logs/train_live_${SLURM_JOB_ID}.log

export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=8

# NCCL Optimizations for FSDP on L4s
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Ensure environment is synced
uv pip install -e .

# Mistral relies heavily on FA2 for sliding window attention.
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# Ensure master_port is randomized slightly to prevent port collision on shared nodes
MASTER_PORT=$((10000 + $RANDOM % 20000))

# CHANGED: nproc_per_node set to 1 for single GPU training
uv run torchrun \
    --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    src/train.py 2>&1 | tee -a logs/train_live_${SLURM_JOB_ID}.log

echo "Job finished at $(date)" | tee -a logs/train_live_${SLURM_JOB_ID}.log