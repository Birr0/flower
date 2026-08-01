#!/bin/bash
#SBATCH --job-name=sdss_models
#SBATCH --output=./logs/job_%a.out
#SBATCH --array=0-14  # 5 embed types * 3 seeds = 15 jobs (indices 0 through 14)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=short

# 1. Define your parameter arrays
EMBED_TYPES=("z" "cond" "cond+z" "uncond" "orig")
SEEDS=(42 43 44)

# 2. Calculate the indices using bash arithmetic
# SLURM_ARRAY_TASK_ID goes from 0 to 14.
# Modulo (%) gives us 0,1,2,3,4 repeating (for embedding types)
# Division (/) gives us 0,0,0,0,0 then 1,1,1,1,1 then 2,2,2,2,2 (for seeds)
EMBED_IDX=$((SLURM_ARRAY_TASK_ID % 5))
SEED_IDX=$((SLURM_ARRAY_TASK_ID / 5))

# 3. Extract the specific parameters for this particular node
CURRENT_EMBED=${EMBED_TYPES[$EMBED_IDX]}
CURRENT_SEED=${SEEDS[$SEED_IDX]}

# Set your feature here (if you want to parameterize this too later, you'd add another multiplier to the array)
FEATURE="LGM_FIB_P50"

# 4. Log the parameters so you can check the output files later
echo "=================================================="
echo "Starting Node Task ID: $SLURM_ARRAY_TASK_ID"
echo "Feature: $FEATURE | Embed: $CURRENT_EMBED | Seed: $CURRENT_SEED"
echo "=================================================="

source $DATA/venvs/wwdc_spectra/bin/activate

# 5. Execute the Python script
python train_logm.py \
    --feature "$FEATURE" \
    --embed_type "$CURRENT_EMBED" \
    --seed "$CURRENT_SEED"