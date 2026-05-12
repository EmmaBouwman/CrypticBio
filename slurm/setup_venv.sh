#!/bin/bash
#SBATCH --job-name=setup_venv
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --mail-user=s2832674@vuw.leidenuniv.nl
#SBATCH --mail-type=END
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --partition=cpu-short
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2

module purge
module load ALICE/default

mkdir -p logs

echo "Removing old venv..."
rm -rf .venv

echo "Creating venv..."
python -m venv .venv
source .venv/bin/activate

echo "Installing packages..."
pip install -e .
pip install "Pillow<10.0.0"

echo "Done!"