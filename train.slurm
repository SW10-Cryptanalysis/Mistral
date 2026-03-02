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

# Load your environment here (e.g., conda activate cipher_env)
export HF_HUB_ENABLE_HF_TRANSFER=1

# Distribute across 4 GPUs
torchrun --nproc_per_node=4 src/train.py