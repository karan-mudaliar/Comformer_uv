#!/bin/bash

#SBATCH --partition=d2r2               
#SBATCH --nodes=1                      
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=12             
#SBATCH --gres=gpu:l40s:1              
#SBATCH --mem=16G                      
#SBATCH --time=72:00:00                
#SBATCH -o output_%j.txt                    
#SBATCH -e error_%j.txt                     
#SBATCH --mail-user=$mudaliar.k@northeastern.edu  
#SBATCH --mail-type=ALL                     

# Handle command line arguments
if [ "$#" -ne 1 ]; then
    echo "Error: You must provide exactly one property to train"
    echo "Usage: $0 <property>"
    echo "Available properties: WF_bottom, WF_top, cleavage_energy"
    exit 1
fi

PROPERTY=$1

# Debug information
echo "Current directory: $PWD"
echo "Python version and path:"
which python
python --version
echo "Training model for property: $PROPERTY"

module load anaconda3/2024.06
module load cuda/12.1

# Try this instead of source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate comformer

cd /home/mudaliar.k/github/comformer_uv

# More debug information
echo "PYTHONPATH: $PYTHONPATH"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Content of data directory:"
ls -l data/

# Run with debug output
python -u comformer/scripts/train_D2R2_single_prop.py "$PROPERTY"