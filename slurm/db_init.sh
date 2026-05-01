#!/bin/bash
#SBATCH --job-name=create_db
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --mail-user="s2548526@vuw.leidenuniv.nl"
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

# Run the python script
echo "Starting to download the database"
uv run ./scripts/create_db.py
echo "Succesfully downloaded the database"