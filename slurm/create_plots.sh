#SBATCH --job-name=plot_from_out
#SBATCH --output=logs/plot_job_%x_%j.out
#SBATCH --mail-user="s.......@vuw.leidenuniv.nl"
#SBATCH --mail-type="END"
#SBATCH --mem=4G
#SBATCH --time=0:05:00
#SBATCH --partition=cpu-short
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

set -e

module purge
module load ALICE/default
module load uv

echo "Starting plotting job (Job ID: $SLURM_JOB_ID, host: $(hostname))"
uv sync
uv run scripts/plot_from_out.py logs/*.out
echo "Done!"