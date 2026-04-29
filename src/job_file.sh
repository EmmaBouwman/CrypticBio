#!/bin/bash
#SBATCH --job-name=run_late_fusion_model
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --mail-user="s2800020@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=6
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3

module load ALICE/default
module load uv

echo "Starting to sync uv"
uv sync
echo "Synced uv"

# Run the python script
echo "Starting to run the model"
uv run src/train.py
echo "Succesfully ran the model"