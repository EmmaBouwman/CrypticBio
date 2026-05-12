#!/bin/bash
#SBATCH --job-name=train_tiny_model
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --mail-user="s2548526@vuw.leidenuniv.nl"
#SBATCH --mail-type="END"
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
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
uv run scripts/train_test.py \
    --batch_size 64 \
    --num_workers 8 \
    --epochs 200 \
    --lr_head 1e-4 \
    --lr_backbone 1e-5 \
    --weight_decay 0.01 \
    --transform_size 224 \
    --random_seed 69 \
    --model_name "vit_small_patch16_224" \
    --save_name "best_animal_sat_small_resized_69.pth" \
    --model_type 3
    