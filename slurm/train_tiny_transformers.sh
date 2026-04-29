#!/bin/bash
#SBATCH --job-name=train_tiny_model
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --mail-user="s2548526@vuw.leidenuniv.nl"
#SBATCH --mail-type="END"
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu-2080ti-11g 
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8

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
uv run scripts/train_transformer.py \
    --batch_size 64 \
    --num_workers 8 \
    --epochs 100 \
    --model_name "vit_tiny_patch16_224" \
    --save_name "best_animal_sat_tiny_resized.pth" \
    --lr_head 2e-4 \
    --lr_backbone 2e-6 \
    --transform_size 224