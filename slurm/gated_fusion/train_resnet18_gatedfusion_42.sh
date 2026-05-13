#!/bin/bash
#SBATCH --job-name=gated_fusion_model_resnet18_42
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --mail-user="s2832674@vuw.leidenuniv.nl"
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

echo "Job: $SLURM_JOB_ID | Model: resnet18 | Random seed: 42"

# Run the python script
echo "Starting to train gated fusion model"
python scripts/train_test.py \
    --batch_size 32 \
    --num_workers 8 \
    --epochs 200 \
    --lr_head 1e-4 \
    --lr_backbone 1e-5 \
    --weight_decay 0.01 \
    --transform_size 224 \
    --random_seed 42 \
    --model_name "resnet18" \
    --save_name "best_gated_fusion_resized_resnet18_42.pth" \
    --model_type 6