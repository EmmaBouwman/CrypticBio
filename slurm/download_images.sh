#!/bin/bash
#SBATCH --job-name=image_download
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --mail-user="s2548526@vuw.leidenuniv.nl"
#SBATCH --mail-type="END"
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --partition=cpu-zen4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Check if arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: sbatch $0 <path_to_csv> <cluster_name>"
    exit 1
fi

CSV_FILE=$1
CLUSTER_NAME=$2

# Load modules
module load ALICE/default
module load uv

# Syncing the uv environment
echo "Starting to sync uv..."
uv sync
echo "Synced uv."

# Run the python script
echo "Starting image download for: $CLUSTER_NAME from $CSV_FILE"
uv run scripts/image_download.py "$CSV_FILE" "$CLUSTER_NAME"

echo "Process completed for $CLUSTER_NAME."