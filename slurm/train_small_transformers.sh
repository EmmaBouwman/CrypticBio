#!/bin/bash
#SBATCH --job-name=train_tiny_model
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --mail-user="s2548526@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --partition=gpu:2080_ti:3
#SBATCH --gres=gpu:1
#SBATCH --ntasks=6
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3

mkdir -p logs

module purge

module load ALICE/default
module load slurm
module load CUDA/12.3.2
module load uv

echo "## Available CUDA devices: $CUDA_VISIBLE_DEVICES"

echo "## Checking status of CUDA device with nvidia-smi"
nvidia-smi

# Syncing the uv environment
echo "Starting to sync uv"
uv sync
echo "Synced uv"

echo "Starting job for vit_tiny_patch16_224"

# Run the script with your requested parameters
uv run scripts/train_transformer.py \
    --batch_size 16 \
    --epochs 100 \
    --model_name "vit_small_patch16_224" \
    --save_name "best_bird_sat_small.pth" \
    --num_workers 4