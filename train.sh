#!/bin/bash
set -e

# 1. Navigate to your mounted workspace
cd /work

# 2. Clone the repository and specific branch if it doesn't exist yet
if [ ! -d "Mistral" ]; then
    echo "Cloning repository..."
    git clone -b UCloud https://github.com/SW10-Cryptanalysis/Mistral.git
fi

cd Mistral
mkdir -p logs
export HF_HUB_ENABLE_HF_TRANSFER=1

# 3. Dynamically count available GPUs
# This allows the script to adapt whether you request 1, 2, or 4 H100s
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Training Job started on $(hostname) at $(date) with $NUM_GPUS GPU(s)" | tee logs/train_live.log

# Safely set CUDA_VISIBLE_DEVICES based on the detected count
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
nvidia-smi | tee -a logs/train_live.log

# Ensure uv is installed 
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

export OMP_NUM_THREADS=16  # Increased for H100's stronger CPUs to feed the dataloader

# 4. H100 NVLink & NCCL Optimizations
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
if [ "$NUM_GPUS" -gt 1 ]; then
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
fi

# Install project dependencies
uv pip install --system -e .

# 5. Flash Attention 2 Installation
echo "Skipping Installing Flash Attention 2 (compiling for Hopper architecture)..."
# uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

MASTER_PORT=$((10000 + $RANDOM % 20000))

# 6. Launch Training
echo "Launching torchrun with $NUM_GPUS processes..."
uv run torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    -m src.train "$@" 2>&1 | tee -a logs/train_live.log

echo "Training Job finished at $(date)" | tee -a logs/train_live.log