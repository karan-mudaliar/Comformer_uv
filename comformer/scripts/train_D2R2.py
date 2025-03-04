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
    output_dir="output/D2R2_multi3",
    output_features=3,    # Critical: final output dimension set to 3.
    data_path="data/DFT_data.csv"  # Add explicit data path
)
