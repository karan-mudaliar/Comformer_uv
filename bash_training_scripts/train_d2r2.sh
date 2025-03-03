#!/bin/bash

#SBATCH --partition=d2r2               # Updated partition
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks (processes)
#SBATCH --cpus-per-task=12             # Keeping CPU count unchanged
#SBATCH --gres=gpu:l40s:1              # Updated GPU resource request
#SBATCH --mem=16G                      # Memory request (same as before)
#SBATCH --time=72:00:00                # Time limit remains unchanged
#SBATCH -o output_%j.txt                    # Standard output file
#SBATCH -e error_%j.txt                     # Standard error file
#SBATCH --mail-user=$mudaliar.k@northeastern.edu  # Email
#SBATCH --mail-type=ALL                     # Type of email notifications

module load anaconda3/2024.06
module load cuda/12.1

source activate comformer

cd /home/mudaliar.k/github/comformer_uv

python comformer/scripts/train_D2R2.py