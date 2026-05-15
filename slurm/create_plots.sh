#!/bin/bash
#SBATCH --job-name=plot_from_out
#SBATCH --output=logs/plot_job_%x_%j.out
#SBATCH --mail-user="s2832674@vuw.leidenuniv.nl"
#SBATCH --mail-type="END"
#SBATCH --mem=4G
#SBATCH --time=0:10:00
#SBATCH --partition=cpu-short
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

module purge
module load ALICE/default

mkdir -p plots

source .venv/bin/activate

echo "Starting plotting job (Job ID: $SLURM_JOB_ID, host: $(hostname))"

python scripts/plot_from_out.py logs/early_fusion_final/*.out
python scripts/plot_from_out.py logs/late_fusion_final/*.out
python scripts/plot_from_out.py logs/gated_fusion_final/*.out
python scripts/plot_from_out.py logs/transformer_final/*.out

echo "Done!"