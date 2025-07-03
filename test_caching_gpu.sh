#!/bin/bash

# Load GPU utility functions
source "$(dirname "$0")/bash_training_scripts/robinlabscripts/gpu_utils.sh"

# Allocate GPU dynamically
ALLOCATED_GPU=$(allocate_gpu)
if [ $? -ne 0 ]; then
    echo "$ALLOCATED_GPU"
    exit 1
fi

# Set GPU environment
export_gpu_env "$ALLOCATED_GPU"

# Simulate SLURM-like job ID using timestamp
JOB_ID=$(date +%Y%m%d_%H%M%S)

# Output/error logs
OUT_FILE="test_caching_${JOB_ID}.txt"
ERR_FILE="test_caching_err_${JOB_ID}.txt"

# Start logging
{
echo "===== Graph Caching Training Test Started: $(date) ====="
echo "Hostname: $(hostname)"
echo "Working Directory: $PWD"
echo "User: $USER"
echo "Job ID: $JOB_ID"

# Debug information
echo "Current directory: $PWD"
echo "Python version and path:"
which python
python --version

# Load conda (no module system)
source /home/kmudaliar/miniconda3/etc/profile.d/conda.sh
conda activate comformer_uv

# Switch to the correct branch
git checkout feature/training-splits
echo "✅ Switched to feature/training-splits branch"

# More debug information
echo "PYTHONPATH: $PYTHONPATH"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Content of data directory:"
ls -l /home/kmudaliar/data/

# Set data path for Robin lab
export ROBINLAB_DATA_PATH="/home/kmudaliar/data"
echo "Data path: $ROBINLAB_DATA_PATH"

# Check data file exists
if [ -f "$ROBINLAB_DATA_PATH/combined_elemental.csv" ]; then
    echo "✅ Data file found: $ROBINLAB_DATA_PATH/combined_elemental.csv"
    echo "Data file size:"
    wc -l "$ROBINLAB_DATA_PATH/combined_elemental.csv"
else
    echo "❌ Data file not found: $ROBINLAB_DATA_PATH/combined_elemental.csv"
    exit 1
fi

# Run the caching training test
echo ""
echo "🚀 Running graph caching training test..."
echo "This will train iComformer on 1000 rows for 1 epoch, twice"
echo "Second run should be faster due to cached graphs"
echo ""

python test_caching_training.py

echo "===== Graph Caching Training Test Finished: $(date) ====="
} > "$OUT_FILE" 2> "$ERR_FILE"

echo "📊 Test completed. Check logs:"
echo "   Output: $OUT_FILE" 
echo "   Errors: $ERR_FILE"