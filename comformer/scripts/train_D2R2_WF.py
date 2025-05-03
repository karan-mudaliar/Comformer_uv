import os
import sys
import logging
import structlog

# Set up structured logging
logger = structlog.get_logger()
logger.info("Starting D2R2 WF properties training")

current_dir = os.path.dirname(os.path.abspath(__file__))
comformer_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(comformer_dir)

from comformer.train_props import train_prop_model 

# Create output directory first - simple fix to avoid directory not found error
output_dir = "output/D2R2_WF_only"
os.makedirs(output_dir, exist_ok=True)

train_prop_model(
    dataset="D2R2_surface_data",
    prop="WF",           # Use the WF field that combines WF_bottom and WF_top
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
    output_features=2,   # Critical: final output dimension set to 2 for WF_bottom and WF_top
    data_path="data/DFT_data.csv"  # Add explicit data path
)