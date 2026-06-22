#!/bin/bash
#SBATCH --job-name=tsne_parallel
#SBATCH --output=./logs/tsne_%a.out
#SBATCH --error=./logs/tsne_%a.err
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00

source ./fp/to/venv # add venv fp here

# Map the array index to your keys
KEYS=("vae" "uncond" "cond")
CURRENT_KEY=${KEYS[$SLURM_ARRAY_TASK_ID]}

echo "Starting t-SNE job for: $CURRENT_KEY"

# Run the python script
python tsne_embed.py --key $CURRENT_KEY