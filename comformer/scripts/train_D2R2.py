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

train_prop_model(learning_rate=0.001, name="iComformer", prop=props[0], pyg_input=True, n_epochs=500, max_neighbors=25, cutoff=4.0, batch_size=64, use_lattice=True, output_dir="yourdir", use_angle=False, save_dataloader=True)
