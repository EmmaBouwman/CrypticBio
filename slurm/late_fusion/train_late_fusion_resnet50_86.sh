#!/bin/bash
<<<<<<<< HEAD:slurm/late_fusion/train_late_fusion_resnet50_86.sh
#SBATCH --job-name=late_fusion_model_resnet50_86
========
#SBATCH --job-name=late_fusion_model_86_50
>>>>>>>> ee834e6 (created new late fusion slurm files with different random seeds):slurm/late_fusion/train_late_fusion_86_50.sh
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --mail-user="s2800020@vuw.leidenuniv.nl"
#SBATCH --mail-type="END"
#SBATCH --mem=128G
<<<<<<<< HEAD:slurm/late_fusion/train_late_fusion_resnet50_86.sh
#SBATCH --time=7-00:00:00
========
#SBATCH --time=3-00:00:00
>>>>>>>> ee834e6 (created new late fusion slurm files with different random seeds):slurm/late_fusion/train_late_fusion_86_50.sh
#SBATCH --partition=gpu-2080ti-11g
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8

<<<<<<<< HEAD:slurm/late_fusion/train_late_fusion_resnet50_86.sh

========
>>>>>>>> ee834e6 (created new late fusion slurm files with different random seeds):slurm/late_fusion/train_late_fusion_86_50.sh
module purge
module load ALICE/default
module load slurm
module load CUDA/12.3.2

mkdir -p logs

source .venv/bin/activate

<<<<<<<< HEAD:slurm/late_fusion/train_late_fusion_resnet50_86.sh
echo "Job: $SLURM_JOB_ID | Model: resnet50 | Random seed: 86"

# Run the python script
echo "Starting to train late fusion model"
python scripts/train_test.py \
    --batch_size 32 \
========
echo "Job: $SLURM_JOB_ID | Model: fusion50 | Random seed: 86"
python scripts/train_test.py \
    --batch_size 64 \
>>>>>>>> ee834e6 (created new late fusion slurm files with different random seeds):slurm/late_fusion/train_late_fusion_86_50.sh
    --num_workers 8 \
    --epochs 200 \
    --lr_head 1e-4 \
    --lr_backbone 1e-5 \
    --weight_decay 0.01 \
    --transform_size 224 \
<<<<<<<< HEAD:slurm/late_fusion/train_late_fusion_resnet50_86.sh
    --random_seed 86 \
    --model_name "resnet50" \
    --save_name "best_late_fusion_resized_resnet50_86.pth" \
    --model_type 5 
========
    --model_name "resnet50" \
    --save_name "best_late_fusion_86_50.pth" \
    --model_type 5 \
    --random_seed 86
>>>>>>>> ee834e6 (created new late fusion slurm files with different random seeds):slurm/late_fusion/train_late_fusion_86_50.sh
