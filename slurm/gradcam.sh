#!/bin/bash
#SBATCH --job-name=gradcam
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --mail-user="s2548526@vuw.leidenuniv.nl"
#SBATCH --mail-type="END"
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16

mkdir -p logs

module purge

module load ALICE/default
module load slurm
module load CUDA/12.3.2
module load uv

nvidia-smi

# Syncing the uv environment
echo "Starting to sync uv"
uv sync

echo "Starting job for vit_tiny_patch16_224"

# Run the script with your requested parameters
uv run scripts/gradcam.py \
  --model_path ./models/base_transformer_model_animal.pth \
  --model_type 1 \
  --model_name vit_base_patch16_224 \
  --output_dir gradcam