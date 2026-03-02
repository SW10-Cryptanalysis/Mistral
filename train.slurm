#!/bin/bash
#SBATCH --job-name=mistral_cipher
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=l4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:l4:4

set -e

# 1. Environment Setup
export HF_HUB_ENABLE_HF_TRANSFER=1
export PATH="$HOME/.local/bin:$PATH"

# 2. Synchronize Dependencies
# We use 'uv sync' which reads your pyproject.toml. 
# It will automatically install the Flash Attention wheel from the URL 
# if you added it to your [project.dependencies] as discussed.
echo "Checking and syncing dependencies..."
uv sync

# 3. Validation Check
# This runs a quick check to make sure the Flash Attention wheel is 
# actually visible to the Python interpreter before starting the heavy job.
uv run python -c "import flash_attn; print(f'Flash Attention {flash_attn.__version__} is ready.')"

# 4. Distributed Training
# --nproc_per_node=4 because you requested 4 L4 GPUs.
echo "Starting Distributed Training on 4 GPUs..."
uv run torchrun --nproc_per_node=4 src/train.py

echo "Job finished at $(date)"