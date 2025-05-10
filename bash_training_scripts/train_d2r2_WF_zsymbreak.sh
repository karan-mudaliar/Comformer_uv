#!/bin/bash
# Train Comformer on D2R2 WF properties with z-symmetry breaking

# Ensure we're in the project root directory
cd "$(dirname "$0")/.." || exit 1

# Create output directory
mkdir -p output/D2R2_WF_zsymbreak

# Run the training script
python -m comformer.scripts.train_D2R2_WF_zsymbreak