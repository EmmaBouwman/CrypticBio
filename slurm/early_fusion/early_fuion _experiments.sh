#!/bin/bash
#SBATCH --job-name=early_fusion_model_experiments
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
module load ALICE/default
module load uv
module load CUDA/12.3.2

mkdir -p logs

# Variabelen met standaardwaarden
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-64}
DROPOUT=${DROPOUT:-0.3}
MODEL_NAME=${MODEL_NAME:-"resnet50"}
SAVE_NAME=${SAVE_NAME:-"best_early_fusion.pth"}
LR_HEAD=${LR_HEAD:-1e-4}
LR_BACKBONE=${LR_BACKBONE:-1e-5}

echo "Starting to sync uv"
uv sync

echo "Job: $SLURM_JOB_ID | Model: $MODEL_NAME | Dropout: $DROPOUT | Epochs: $EPOCHS"

uv run scripts/train_test.py \
    --model_type 4 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --dropout $DROPOUT \
    --model_name $MODEL_NAME \
    --save_name $SAVE_NAME \
    --lr_head $LR_HEAD \
    --lr_backbone $LR_BACKBONE \
    --transform_size 224 \
    --num_workers 8