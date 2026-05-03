#!/bin/bash
#SBATCH --job-name=test_early_fusion_model
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --mail-user="s2832674@vuw.leidenuniv.nl"
#SBATCH --mail-type="END"
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --partition=cpu-short
#SBATCH --ntasks=6
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3

# load modules (assuming you start from the default environment)
module load ALICE/default
module load uv

# Syncing the uv environment
echo "Starting to sync uv"
uv sync
echo "Synced uv"

# Creates log directory
mkdir -p logs

# Run the python script
echo "Starting to train early fusion model"
uv run ./src/train.py
echo "Succesfully tested train.py"