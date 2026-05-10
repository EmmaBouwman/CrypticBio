#!/bin/bash
#SBATCH --job-name=early_fusion_model
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --mail-user="s2832674@vuw.leidenuniv.nl"
#SBATCH --mail-type="END"
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu-2080ti-11g
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8

module purge

# load modules (assuming you start from the default environment)
module load ALICE/default
module load uv
module load slurm
module load CUDA/12.3.2

# Syncing the uv environment
echo "Starting to sync uv"
uv sync

# Creates log directory
mkdir -p logs

# Run the python script
echo "Starting to train early fusion model"
uv run scripts/train_test.py \
    --batch_size 64 \
    --num_workers 8 \
    --epochs 100 \
    --model_name "resnet50" \
    --save_name "best_early_fusion_resized.pth" \
    --lr_head 1e-4 \
    --lr_backbone 1e-5 \
    --transform_size 224 \
    --model_type 4