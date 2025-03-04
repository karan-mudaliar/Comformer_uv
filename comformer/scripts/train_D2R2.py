import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
comformer_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(comformer_dir)

from comformer.train_props import train_prop_model 

props = [
    "WF_bottom",
    "WF_top",
    "cleavage_energy",
]

# Create output directory first
output_dir = "output/D2R2_multi3"
os.makedirs(output_dir, exist_ok=True)

# Explicitly set model config to use output_features=3
model_config = {
    "name": "iComformer",
    "output_features": 3  # Critical: final output dimension set to 3 for multi-property prediction
}

train_prop_model(
    dataset="D2R2_surface_data",
    prop="all",           # Use the combined field.
    name="iComformer",
    pyg_input=True,
    n_epochs=1,
    max_neighbors=25,
    cutoff=4.0,
    batch_size=64,
    use_lattice=True,
    use_angle=True,
    save_dataloader=True,
    output_dir=output_dir,
    model=model_config,   # Pass explicit model config
    data_path="data/DFT_data.csv"  # Add explicit data path
)
