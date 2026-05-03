#!/bin/bash
#SBATCH --job-name=run_late_fusion_model
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --mail-user="s2800020@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu-2080ti-11g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=6
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8

module purge 
module load ALICE/default
module load slurm
module load CUDA/12.3.2
module load uv

echo "Starting to sync uv"
uv sync
echo "Synced uv"

# Run the python script
echo "Starting to run the model"
uv run src/train.py
echo "Succesfully ran the model"