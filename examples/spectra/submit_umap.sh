#!/bin/bash
#SBATCH --job-name=umap_manifold
#SBATCH --output=logs/umap_%A_%a.out
#SBATCH --error=logs/umap_%A_%a.err
#SBATCH --array=0-1            # FIXED: 2 spenders * 1 split = 2 tasks (0 and 1)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      
#SBATCH --mem=32G              
#SBATCH --time=04:00:00

# 1. Parameter Arrays
SPENDERS=("spender_I" "spender_II")
SPLITS=("orig")

# 2. Index Math
# Total splits is 1, so:
# Task 0: 0 / 1 = 0 (spender_I), 0 % 1 = 0 (orig)
# Task 1: 1 / 1 = 1 (spender_II), 1 % 1 = 0 (orig)
NUM_SPLITS=${#SPLITS[@]}
SPENDER_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SPLITS))
SPLIT_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SPLITS))

CURRENT_SPENDER=${SPENDERS[$SPENDER_IDX]}
CURRENT_SPLIT=${SPLITS[$SPLIT_IDX]}

# 3. Execution
# Ensure logs directory exists
mkdir -p logs

echo "Running UMAP bench for $CURRENT_SPENDER on $CURRENT_SPLIT"

# Make sure the python filename matches your actual script name
python umap_benchmark.py \
    --spender "$CURRENT_SPENDER" \
    --split "$CURRENT_SPLIT"