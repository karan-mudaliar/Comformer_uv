import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
comformer_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(comformer_dir)

from comformer.train_props import train_prop_model 

# Create output directory
output_dir = "output/eComformer_cleavage_energy_elemental_splits_zsym_cartesian"
os.makedirs(output_dir, exist_ok=True)

train_prop_model(
    dataset="D2R2_surface_data",
    prop="cleavage_energy",
    name="eComformer",
    pyg_input=True,
    n_epochs=150,
    max_neighbors=25,
    cutoff=4.0,
    batch_size=32,
    use_lattice=True,
    use_angle=True,
    break_z_symmetry=True,  # eComformer WITH symmetry breaking
    z_symmetry_method="cartesian",  # NEW: Use cartesian z-coordinates instead of relative
    save_dataloader=False,
    output_dir=output_dir,
    output_features=1,    # Single output for cleavage energy
    data_path=os.environ.get("ROBINLAB_DATA_PATH", "/home/mudaliar.k/data") + "/combined_elemental.csv"
)