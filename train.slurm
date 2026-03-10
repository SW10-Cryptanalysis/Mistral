#!/bin/bash
#SBATCH --job-name=mistral_cipher_1gpu
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=l4
#SBATCH --cpus-per-task=8      # Keep at 32 when moving back to 4 GPUs
#SBATCH --mem=32G            # Keep at 64G when moving back to 4 GPUs
#SBATCH --gres=gpu:l4:1      # Change to gpu:l4:4 when ready

set -e

# Update to your actual Mistral project directory
PROJECT_DIR=/ceph/project/SW10-CausalLM/Mistral
cd $PROJECT_DIR

# ==============================================================================
# DYNAMIC AUTO-CHAINING LOGIC
# ==============================================================================
if [ -d "outputs/final_model" ]; then
    echo "Success: 'outputs/final_model' found. Training is completely finished."
    echo "Breaking the chain. No further jobs will be submitted."
    exit 0
fi

echo "Dynamically queuing the next 12-hour continuation job..."
sbatch --dependency=afterany:$SLURM_JOB_ID $0
# ==============================================================================

mkdir -p logs
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "Job started on $(hostname) at $(date)" | tee logs/train_live_${SLURM_JOB_ID}.log

export CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print $1}')

nvidia-smi | tee -a logs/train_live_${SLURM_JOB_ID}.log

export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=8

# NCCL Optimizations
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

uv pip install -e .
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

MASTER_PORT=$((10000 + $RANDOM % 20000))

# I have strictly removed all trailing whitespace after the backslashes below.
# When copying this, ensure no spaces are accidentally added after the '\'.
uv run torchrun \
    --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    src/train.py 2>&1 | tee -a logs/train_live_${SLURM_JOB_ID}.log

echo "Job finished at $(date)" | tee -a logs/train_live_${SLURM_JOB_ID}.log