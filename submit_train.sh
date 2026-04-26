#!/bin/bash
#SBATCH --job-name=vit_sweep
#SBATCH --partition=L40s_students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/sweep_%A_%a.log
#SBATCH --array=0-2                # 0=tiny, 1=small, 2=base

mkdir -p logs

# Define our model list
MODELS=("vit_tiny_patch16_224" "vit_small_patch16_224" "vit_base_patch16_224")
NAMES=("tiny" "small" "base")

# Pick the current model based on the array index
CURRENT_MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
CURRENT_NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}

echo "Starting job for $CURRENT_MODEL (Index: $SLURM_ARRAY_TASK_ID)"

# Run the script with your requested parameters
uv run scripts/train_transformer.py \
    --batch_size 16 \
    --epochs 25 \
    --model_name "$CURRENT_MODEL" \
    --save_name "best_bird_sat_${CURRENT_NAME}.pth" \
    --num_workers 4