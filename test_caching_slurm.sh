#!/bin/bash

#SBATCH --job-name=test_caching_training
#SBATCH --output=test_caching_training_%j.txt
#SBATCH --error=test_caching_training_err_%j.txt
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

echo "===== Graph Caching Training Test Started: $(date) ====="
echo "Hostname: $(hostname)"
echo "Working Directory: $PWD"
echo "User: $USER"
echo "Job ID: $SLURM_JOB_ID"

# Check GPU
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Load conda environment
source /home/mudaliar.k/miniconda3/etc/profile.d/conda.sh
conda activate comformer_uv

# Switch to the correct branch
git checkout feature/training-splits
echo "✅ Switched to feature/training-splits branch"

# Set data path
export ROBINLAB_DATA_PATH="/home/mudaliar.k/data"
echo "Data path: $ROBINLAB_DATA_PATH"

# Check data file exists
if [ -f "$ROBINLAB_DATA_PATH/combined_elemental.csv" ]; then
    echo "✅ Data file found: $ROBINLAB_DATA_PATH/combined_elemental.csv"
    wc -l "$ROBINLAB_DATA_PATH/combined_elemental.csv"
else
    echo "❌ Data file not found: $ROBINLAB_DATA_PATH/combined_elemental.csv"
    exit 1
fi

# Run the caching training test
echo "🚀 Running graph caching training test..."
echo "This will train iComformer on 1000 rows for 1 epoch, twice"
echo "Second run should be faster due to cached graphs"

python test_caching_training.py

exit_code=$?

echo "===== Graph Caching Training Test Finished: $(date) ====="
echo "Exit code: $exit_code"

if [ $exit_code -eq 0 ]; then
    echo "✅ Test passed successfully!"
    echo "Check the timing results above to see caching speedup"
else
    echo "❌ Test failed with exit code $exit_code"
fi

exit $exit_code