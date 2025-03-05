"""Implementation based on the template of ALIGNN."""

import imp
import random
from pathlib import Path
from typing import Optional
# from typing import Dict, List, Optional, Set, Tuple
import os
import torch
import ast
import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms,  pmg_to_atoms
import structlog
from pymatgen.core import Structure
from comformer.graphs import PygGraph, PygStructureDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from jarvis.db.jsonutils import dumpjson
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
# from sklearn.pipeline import Pipeline
import pickle as pk
from sklearn.preprocessing import StandardScaler
# use pandas progress_apply
tqdm.pandas()
logger = structlog.get_logger()


def load_dataset(
    name: str = "D2R2_surface_data", 
    data_path: str = "data/DFT_data.csv",
    target=None,
    limit: Optional[int] = None,
    classification_threshold: Optional[float] = None,
):
    logger.info(f"reading data from path {data_path}")
    logger.info(f"limit parameter value: {limit}")
    df = pd.read_csv(data_path, on_bad_lines="skip")
    if limit is not None:
        df = df[:limit]

    df["jid"] = df["mpid"].astype(str) + df["miller"].astype(str) + df["term"].astype(str)
    
    # For multi-property training: if target=="all" and dataset is D2R2_surface_data,
    # combine WF_bottom, WF_top, and cleavage_energy into one list.
    if target == "all":
        logger.info("Combining WF_bottom, WF_top, and cleavage_energy into 'all' field for D2R2_surface_data")
        df["all"] = df.apply(
            lambda x: [x["WF_bottom"], x["WF_top"], x["cleavage_energy"]],
            axis=1
        )
    else:
        logger.info(f"The target property is {target}")

    if "slab" in df.columns:
        df = df.rename(columns={"slab": "atoms"})
    
    logger.info(f"There are {len(df)} rows in this df")
    return df


def mean_absolute_deviation(data, axis=None):
    """Get Mean absolute deviation."""
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)


def load_pyg_graphs(
    df: pd.DataFrame,
    name: str = "dft_3d",
    neighbor_strategy: str = "k-nearest",
    cutoff: float = 8,
    max_neighbors: int = 12,
    cachedir: Optional[Path] = None,
    use_canonize: bool = False,
    use_lattice: bool = False,
    use_angle: bool = False,
):
    """Construct crystal graphs.

    Load only atomic number node features
    and bond displacement vector edge features.

    Resulting graphs have scheme e.g.
    ```
    Graph(num_nodes=12, num_edges=156,
          ndata_schemes={'atom_features': Scheme(shape=(1,)}
          edata_schemes={'r': Scheme(shape=(3,)})
    ```
    """
    def atoms_to_graph(atoms):
        """Convert structure dict to DGLGraph."""
        # structure = Atoms.from_dict(atoms)
        structure = pmg_to_atoms(Structure.from_dict(eval(atoms)))
        return PygGraph.atom_dgl_multigraph(
            structure,
            neighbor_strategy=neighbor_strategy,
            cutoff=cutoff,
            atom_features="atomic_number",
            max_neighbors=max_neighbors,
            compute_line_graph=False,
            use_canonize=use_canonize,
            use_lattice=use_lattice,
            use_angle=use_angle,
        )
    logger.info("Applying transform to our code")
    graphs = df["atoms"].apply(atoms_to_graph).values 
    # graphs = df["atoms"].paral_apply(atoms_to_graph).values

    return graphs


def get_id_train_val_test(
    total_size=None,
    split_seed=123,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    n_train=None,
    n_test=None,
    n_val=None,
    keep_data_order=False,
):
    """Get train, val, test IDs."""
    if (
        train_ratio is None
        and val_ratio is not None
        and test_ratio is not None
    ):
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print("Using rest of the dataset except the test and val sets.")
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    # indices = list(range(total_size))
    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)
    ids = list(np.arange(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)
    if n_train + n_val + n_test > total_size:
        raise ValueError(
            "Check total number of samples.",
            n_train + n_val + n_test,
            ">",
            total_size,
        )

    id_train = ids[:n_train]
    id_val = ids[-(n_val + n_test) : -n_test]
    id_test = ids[-n_test:]
    return id_train, id_val, id_test


def get_pyg_dataset(
    dataset=[],
    id_tag="jid",
    target="",
    neighbor_strategy="",
    atom_features="",
    use_canonize="",
    name="",
    line_graph="",
    cutoff=8.0,
    max_neighbors=12,
    classification=False,
    output_dir=".",
    tmp_name="dataset",
    use_lattice=False,
    use_angle=False,
    data_from='Jarvis',
    use_save=False,
    mean_train=None,
    std_train=None,
    now=False, # for test
):
    """Get pyg Dataset."""
    df = pd.DataFrame(dataset)
    
    # Modify to handle multi-output case
    if target == "all":
        vals = np.array([x for x in df[target].values])
        output_features = 3  # Number of properties we're predicting
    else:
        vals = df[target].values
        output_features = 1
    
    output_dir = "./saved_data/" + tmp_name + "test_graph_angle.pkl" # for fast test use
    print("data range", np.max(vals), np.min(vals))
    print(output_dir)
    print('graphs not saved')
    graphs = load_pyg_graphs(
        df,
        name=name,
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        use_lattice=use_lattice,
        use_angle=use_angle,
    )
    if mean_train == None:
        mean_train = np.mean(vals)
        std_train = np.std(vals)
        data = PygStructureDataset(
            df,
            graphs,
            target=target,
            atom_features=atom_features,
            line_graph=line_graph,
            id_tag=id_tag,
            classification=classification,
            neighbor_strategy=neighbor_strategy,
            mean_train=mean_train,
            std_train=std_train,
        )
    else:
        data = PygStructureDataset(
            df,
            graphs,
            target=target,
            atom_features=atom_features,
            line_graph=line_graph,
            id_tag=id_tag,
            classification=classification,
            neighbor_strategy=neighbor_strategy,
            mean_train=mean_train,
            std_train=std_train,
        )
    return data, mean_train, std_train


def get_train_val_loaders(
    dataset: str = "dft_3d",
    dataset_array=[],
    target: str = "formation_energy_peratom",
    atom_features: str = "cgcnn",
    neighbor_strategy: str = "k-nearest",
    n_train=None,
    n_val=None,
    n_test=None,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    batch_size: int = 5,
    standardize: bool = False,
    line_graph: bool = True,
    split_seed: int = 123,
    workers: int = 0,
    pin_memory: bool = True,
    save_dataloader: bool = False,
    filename: str = "sample",
    id_tag: str = "jid",
    use_canonize: bool = False,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    classification_threshold: Optional[float] = None,
    target_multiplication_factor: Optional[float] = None,
    standard_scalar_and_pca=False,
    keep_data_order=False,
    output_features=1,
    output_dir=None,
    matrix_input=False,
    pyg_input=False,
    use_lattice=False,
    use_angle=False,
    use_save=True,
    mp_id_list=None,
    data_path=None,
):
    """Help function to set up JARVIS train and val dataloaders."""
    # data loading
    mean_train=None
    std_train=None
    assert (matrix_input and pyg_input) == False
    
    train_sample = filename + "_train.data"
    val_sample = filename + "_val.data"
    test_sample = filename + "_test.data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if (
        os.path.exists(train_sample)
        and os.path.exists(val_sample)
        and os.path.exists(test_sample)
        and save_dataloader
    ):
        print("Loading from saved file...")
        print("Make sure all the DataLoader params are same.")
        print("This module is made for debugging only.")
        train_loader = torch.load(train_sample)
        val_loader = torch.load(val_sample)
        test_loader = torch.load(test_sample)
        if train_loader.pin_memory != pin_memory:
            train_loader.pin_memory = pin_memory
        if test_loader.pin_memory != pin_memory:
            test_loader.pin_memory = pin_memory
        if val_loader.pin_memory != pin_memory:
            val_loader.pin_memory = pin_memory
        if train_loader.num_workers != workers:
            train_loader.num_workers = workers
        if test_loader.num_workers != workers:
            test_loader.num_workers = workers
        if val_loader.num_workers != workers:
            val_loader.num_workers = workers
        print("train", len(train_loader.dataset))
        print("val", len(val_loader.dataset))
        print("test", len(test_loader.dataset))
        return (
            train_loader,
            val_loader,
            test_loader,
            train_loader.dataset.prepare_batch,
        )
    else:
        if not dataset_array:
            # First load the dataset
            df = load_dataset(name=dataset, data_path=data_path)
            
            # Make sure the 'all' field is created if target is 'all'
            if target == "all" and "all" not in df.columns:
                print("Creating 'all' field from WF_bottom, WF_top, and cleavage_energy columns")
                required_cols = ["WF_bottom", "WF_top", "cleavage_energy"]
                if all(col in df.columns for col in required_cols):
                    df["all"] = df.apply(
                        lambda x: [x["WF_bottom"], x["WF_top"], x["cleavage_energy"]],
                        axis=1
                    )
                else:
                    missing = [col for col in required_cols if col not in df.columns]
                    raise ValueError(f"Cannot create 'all' field. Missing columns: {missing}")
            
            d = df.to_dict(orient="records")
        else:
            d = dataset_array
            # for ii, i in enumerate(pc_y):
            #    d[ii][target] = pc_y[ii].tolist()

        dat = []
        if classification_threshold is not None:
            print(
                "Using ",
                classification_threshold,
                " for classifying ",
                target,
                " data.",
            )
            print("Converting target data into 1 and 0.")
        all_targets = []

        # TODO:make an all key in qm9_dgl
        if dataset == "qm9_dgl" and target == "all":
            print("Making all qm9_dgl")
            tmp = []
            for ii in d:
                ii["all"] = [
                    ii["mu"],
                    ii["alpha"],
                    ii["homo"],
                    ii["lumo"],
                    ii["gap"],
                    ii["r2"],
                    ii["zpve"],
                    ii["U0"],
                    ii["U"],
                    ii["H"],
                    ii["G"],
                    ii["Cv"],
                ]
                tmp.append(ii)
            print("Made all qm9_dgl")
            d = tmp
        # logger.info(d)
        for i in d:
            # If target is 'all' but not present in the data, create it on the fly
            if target == "all" and target not in i:
                required_cols = ["WF_bottom", "WF_top", "cleavage_energy"]
                if all(col in i for col in required_cols):
                    i[target] = [i["WF_bottom"], i["WF_top"], i["cleavage_energy"]]
                    print(f"Creating 'all' field on the fly: {i[target]}")
                else:
                    missing = [col for col in required_cols if col not in i]
                    print(f"Warning: Cannot create 'all' field. Missing columns: {missing}. Available: {list(i.keys())}")
                    continue  # Skip this item
            
            try:
                if isinstance(i[target], list):  # multioutput target
                    all_targets.append(torch.tensor(i[target]))
                    dat.append(i)
                elif (
                    i[target] is not None
                    and i[target] != "na"
                    and not math.isnan(i[target])
                ):
                    if target_multiplication_factor is not None:
                        i[target] = i[target] * target_multiplication_factor
                    if classification_threshold is not None:
                        if i[target] <= classification_threshold:
                            i[target] = 0
                        elif i[target] > classification_threshold:
                            i[target] = 1
                        else:
                            raise ValueError(
                                "Check classification data type.",
                                i[target],
                                type(i[target]),
                            )
                    dat.append(i)
                    all_targets.append(i[target])
            except KeyError:
                print(f"Warning: Target '{target}' not found in data. Keys: {list(i.keys())}")
                continue  # Skip this item
    
    if mp_id_list is not None:
        if mp_id_list == 'bulk':
            print('using mp bulk dataset')
            with open('/data/keqiangyan/bulk_shear/bulk_megnet_train.pkl', 'rb') as f:
                dataset_train = pk.load(f)
            with open('/data/keqiangyan/bulk_shear/bulk_megnet_val.pkl', 'rb') as f:
                dataset_val = pk.load(f)
            with open('/data/keqiangyan/bulk_shear/bulk_megnet_test.pkl', 'rb') as f:
                dataset_test = pk.load(f)
        
        if mp_id_list == 'shear':
            print('using mp shear dataset')
            with open('/data/keqiangyan/bulk_shear/shear_megnet_train.pkl', 'rb') as f:
                dataset_train = pk.load(f)
            with open('/data/keqiangyan/bulk_shear/shear_megnet_val.pkl', 'rb') as f:
                dataset_val = pk.load(f)
            with open('/data/keqiangyan/bulk_shear/shear_megnet_test.pkl', 'rb') as f:
                dataset_test = pk.load(f)

    else:
        id_train, id_val, id_test = get_id_train_val_test(
            total_size=len(dat),
            split_seed=split_seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            n_train=n_train,
            n_test=n_test,
            n_val=n_val,
            keep_data_order=keep_data_order,
        )
        ids_train_val_test = {}
        ids_train_val_test["id_train"] = [dat[i][id_tag] for i in id_train]
        ids_train_val_test["id_val"] = [dat[i][id_tag] for i in id_val]
        ids_train_val_test["id_test"] = [dat[i][id_tag] for i in id_test]
        dumpjson(
            data=ids_train_val_test,
            filename=os.path.join(output_dir, "ids_train_val_test.json"),
        )
        dataset_train = [dat[x] for x in id_train]
        dataset_val = [dat[x] for x in id_val]
        dataset_test = [dat[x] for x in id_test]
    
    train_data, mean_train, std_train = get_pyg_dataset(
        dataset=dataset_train,
        id_tag=id_tag,
        atom_features=atom_features,
        target=target,
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        name=dataset,
        line_graph=line_graph,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        classification=classification_threshold is not None,
        output_dir=output_dir,
        tmp_name="train_data",
        use_lattice=use_lattice,
        use_angle=use_angle,
        use_save=False,
    )
    val_data,_,_ = get_pyg_dataset(
        dataset=dataset_val,
        id_tag=id_tag,
        atom_features=atom_features,
        target=target,
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        name=dataset,
        line_graph=line_graph,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        classification=classification_threshold is not None,
        output_dir=output_dir,
        tmp_name="val_data",
        use_lattice=use_lattice,
        use_angle=use_angle,
        use_save=False,
        mean_train=mean_train,
        std_train=std_train,
    )
    test_data,_,_ = get_pyg_dataset(
        dataset=dataset_test,
        id_tag=id_tag,
        atom_features=atom_features,
        target=target,
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        name=dataset,
        line_graph=line_graph,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        classification=classification_threshold is not None,
        output_dir=output_dir,
        tmp_name="test_data",
        use_lattice=use_lattice,
        use_angle=use_angle,
        use_save=False,
        mean_train=mean_train,
        std_train=std_train,
    )

    
    collate_fn = train_data.collate
    if line_graph:
        collate_fn = train_data.collate_line_graph

    # use a regular pytorch dataloader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    if save_dataloader:
        torch.save(train_loader, train_sample)
        torch.save(val_loader, val_sample)
        torch.save(test_loader, test_sample)
    
    print("n_train:", len(train_loader.dataset))
    print("n_val:", len(val_loader.dataset))
    print("n_test:", len(test_loader.dataset))
    return (
        train_loader,
        val_loader,
        test_loader,
        train_loader.dataset.prepare_batch,
        mean_train,
        std_train,
    )
    
