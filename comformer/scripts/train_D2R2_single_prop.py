import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
comformer_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(comformer_dir)

from comformer.train_props import train_prop_model 

# Define the available properties
properties = [
    "WF_bottom",
    "WF_top",
    "cleavage_energy",
]

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a Comformer model for a single property')
parser.add_argument('property', type=str, help=f'Property to train, one of: {", ".join(properties)}')
args = parser.parse_args()

prop = args.property

# Validate property name
if prop not in properties:
    print(f"Error: Property must be one of {properties}")
    print(f"You specified: {prop}")
    sys.exit(1)

# Create output directory based on property name
output_dir = f"output/D2R2_{prop}"
os.makedirs(output_dir, exist_ok=True)

print(f"Training single property model for: {prop}")
print(f"Output directory: {output_dir}")

# Train the model for the specified property
train_prop_model(
    dataset="D2R2_surface_data",
    prop=prop,              # Single property to train
    name="iComformer",
    pyg_input=True,
    n_epochs=350,
    max_neighbors=25,
    cutoff=4.0,
    batch_size=64,
    use_lattice=True,
    use_angle=True,
    save_dataloader=True,
    output_dir=output_dir,
    output_features=1,      # Single output feature
    data_path="data/DFT_data.csv")