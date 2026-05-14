#!/bin/bash
#SBATCH --job-name=train_tiny_model_sat_69
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

module purge
module load ALICE/default
module load slurm
module load CUDA/12.3.2

mkdir -p logs

source .venv/bin/activate

echo "Job: $SLURM_JOB_ID | Model: vit_tiny_satellite | Random seed: 69"

# Run the script with your requested parameters
python scripts/train_test.py \
    --batch_size 64 \
    --num_workers 8 \
    --epochs 200 \
    --lr_head 1e-4 \
    --lr_backbone 1e-5 \
    --weight_decay 0.01 \
    --transform_size 224 \
    --random_seed 69 \
    --model_name "vit_tiny_patch16_224" \
    --save_name "best_sat_tiny_resized.pth" \
    --model_type 2