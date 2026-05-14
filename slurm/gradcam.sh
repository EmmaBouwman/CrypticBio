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

module purge
module load ALICE/default
module load slurm
module load CUDA/12.3.2

mkdir -p logs

source .venv/bin/activate

echo "Job: $SLURM_JOB_ID | Model: vit_tiny | Random seed: 42"


# Run the script with your requested parameters
python scripts/gradcam.py \
  --model_path ./best_animal_sat_tiny_resized_42.pth \
  --model_type 3 \
  --model_name vit_tiny_patch16_224 \
  --output_dir gradcam