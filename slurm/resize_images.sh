#!/bin/bash
#SBATCH --job-name=resize_images
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user="s2548526@vuw.leidenuniv.nl"
#SBATCH --mail-type="END"
#SBATCH --mem=16G                     # Resizing 16 images at once doesn't need huge RAM
#SBATCH --time=04:00:00               # 1 hour is plenty for ~70k-140k images with 16 cores
#SBATCH --partition=cpu-short         # Use CPU partition to save GPU credits
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            # Matches your max_workers=16 in Python

# 1. Environment Setup
module purge
module load ALICE/default
module load uv

# 2. Synchronize your environment (ensures torchvision/PIL are ready)
echo "## Syncing environment..."
uv sync

# 3. Create log directory if not exists
mkdir -p logs

# 4. Run the script
echo "## Starting resizing of satelite images"
uv run scripts/resize_images.py \
    --size 224 \
    --source "/zfsstore/courses/2025-2026/4343MMEBX/Group3/images_sh"

echo "## Starting resizing of animal images"
uv run scripts/resize_images.py \
    --size 224 \
    --source "/zfsstore/courses/2025-2026/4343MMEBX/Group3/images_cb"

echo "## Job finished at $(date)"