#!/bin/bash
#SBATCH --job-name=resid_orig
#SBATCH --output=logs/resid_%A_%a.out
#SBATCH --error=logs/resid_%A_%a.err
#SBATCH --array=0-1                  # FIXED: 2 spenders * 1 split = 2 tasks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            
#SBATCH --mem=32G                    
#SBATCH --time=04:00:00              

# 1. Parameter Arrays
SPENDERS=("spender_I" "spender_II")
SPLITS=("orig")                      # Restricted to orig split only

# 2. Index Math
# With only 1 split, Task 0 = Spender I, Task 1 = Spender II
NUM_SPLITS=${#SPLITS[@]}
SPENDER_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SPLITS))
SPLIT_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SPLITS))

CURRENT_SPENDER=${SPENDERS[$SPENDER_IDX]}
CURRENT_SPLIT=${SPLITS[$SPLIT_IDX]}

# 3. Execution Setup
mkdir -p logs
echo "Processing: $CURRENT_SPENDER | Split: $CURRENT_SPLIT"

# 4. Run the Python worker
# Make sure the filename matches your saved residualization script
python resid_benchmark.py \
    --spender "$CURRENT_SPENDER" \
    --split "$CURRENT_SPLIT"