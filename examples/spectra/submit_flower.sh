#!/bin/bash

# Configuration
SPLITS=("cond" "orig" "uncond")
SPENDERS=("spender_I" "spender_II")

for SPENDER in "${SPENDERS[@]}"
do
    for SPLIT in "${SPLITS[@]}"
    do
        echo "Launching: $SPENDER with split $SPLIT"
        
        # Use '&' to run in background (parallel)
        # Or remove '&' to run them one after another
        python flower_benchmark.py --spender "$SPENDER" --split "$SPLIT" &
    done
done

# Wait for all background processes to finish
wait
echo "All tasks finished."