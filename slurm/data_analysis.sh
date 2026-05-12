#!/bin/bash
#SBATCH --job-name=data_analysis
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --mail-user="s2800020@vuw.leidenuniv.nl"
#SBATCH --mail-type=END,FAIL

#SBATCH --partition=cpu-short
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

mkdir -p logs

module purge
module load ALICE/default
module load uv

echo "Starting environment sync..."
uv sync || exit 1

echo "Starting data analysis job..."

uv run python src/data_analysis_network.py