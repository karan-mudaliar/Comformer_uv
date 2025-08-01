from functools import partial

# from pathlib import Path
from typing import Any, Dict, Union

import ignite
import torch

from ignite.contrib.handlers import TensorboardLogger
try:
    from ignite.contrib.handlers.stores import EpochOutputStore
except Exception as exp:
    from ignite.handlers.stores import EpochOutputStore

    pass
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers.tensorboard_logger import (
    global_step_from_engine,
)
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.contrib.metrics import ROC_AUC, RocCurve
from ignite.metrics import (
    Accuracy,
    Precision,
    Recall,
    ConfusionMatrix,
)
import pickle as pk
import numpy as np
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Loss, MeanAbsoluteError
from torch import nn
from comformer import models
from comformer.data import get_train_val_loaders
from comformer.config import TrainingConfig
from comformer.models.comformer import iComformer, eComformer

from jarvis.db.jsonutils import dumpjson
import json
import pprint

import os


# torch config
torch.set_default_dtype(torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolynomialLRDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iters, start_lr, end_lr, power=1, last_epoch=-1):
        self.max_iters = max_iters
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.power = power
        self.last_iter = 0  # Custom attribute to keep track of last iteration count
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            (self.start_lr - self.end_lr) * 
            ((1 - self.last_iter / self.max_iters) ** self.power) + self.end_lr 
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        self.last_iter += 1  # Increment the last iteration count
        return super().step(epoch)

def count_parameters(model):
        total_params = 0
        for parameter in model.parameters():
            total_params += parameter.element_size() * parameter.nelement()
        for parameter in model.buffers():
            total_params += parameter.element_size() * parameter.nelement()
        total_params = total_params / 1024 / 1024
        print(f"Total size: {total_params}")
        print("Total trainable parameter number", sum(p.numel() for p in model.parameters() if p.requires_grad))
        return total_params

def activated_output_transform(output):
    """Exponentiate output."""
    y_pred, y = output
    y_pred = torch.exp(y_pred)
    y_pred = y_pred[:, 1]
    return y_pred, y


def make_standard_scalar_and_pca(output):
    """Use standard scalar and PCS for multi-output data."""
    sc = pk.load(open(os.path.join(tmp_output_dir, "sc.pkl"), "rb"))
    y_pred, y = output
    y_pred = torch.tensor(sc.transform(y_pred.cpu().numpy()), device=device)
    y = torch.tensor(sc.transform(y.cpu().numpy()), device=device)
    return y_pred, y


def thresholded_output_transform(output):
    """Round off output."""
    y_pred, y = output
    y_pred = torch.round(torch.exp(y_pred))
    # print ('output',y_pred)
    return y_pred, y


def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def setup_optimizer(params, config: TrainingConfig):
    """Set up optimizer for param groups."""
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    return optimizer


def train_main(
    config: Union[TrainingConfig, Dict[str, Any]],
    model: nn.Module = None,
    train_val_test_loaders=[],
    test_only=False,
    use_save=True,
    mp_id_list=None,
):
    """
    `config` should conform to matformer.conf.TrainingConfig, and
    if passed as a dict with matching keys, pydantic validation is used
    """
    print(config)
    import os
    
    # Ensure config is properly set up
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)
            print('error in converting to training config!')
            # Fall back to using dict directly if conversion fails
    
    # Get the output directory from config (either dict or object)
    output_dir = config.output_dir if hasattr(config, 'output_dir') else config.get('output_dir', '.')
    
    # Make sure the output directory exists (create all parent directories)
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        try:
            # Using exist_ok to handle race conditions
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory {output_dir}: {str(e)}")
            # Fall back to a safe directory if we can't create the requested one
            output_dir = os.path.join(os.getcwd(), "output")
            os.makedirs(output_dir, exist_ok=True)
            print(f"Using fallback output directory: {output_dir}")
        
    # Update config with output_dir if it's a dict
    if isinstance(config, dict):
        config['output_dir'] = output_dir
        
    checkpoint_dir = os.path.join(output_dir)
    deterministic = False
    classification = False
    print("config:")
    
    # Handle both dict and TrainingConfig objects
    tmp = config.dict() if hasattr(config, 'dict') else config.copy()
    f = open(os.path.join(output_dir, "config.json"), "w")
    f.write(json.dumps(tmp, indent=4))
    f.close()
    global tmp_output_dir
    tmp_output_dir = output_dir
    pprint.pprint(tmp)
    # Handle classification threshold check for both dict and object
    classification_threshold = (
        config.classification_threshold 
        if hasattr(config, 'classification_threshold') 
        else config.get('classification_threshold')
    )
    if classification_threshold is not None:
        classification = True
    
    # Handle random seed check for both dict and object
    random_seed = config.random_seed if hasattr(config, 'random_seed') else config.get('random_seed')
    if random_seed is not None:
        deterministic = True
        ignite.utils.manual_seed(random_seed)

    line_graph = True
    if not train_val_test_loaders:
        # use input standardization for all real-valued feature sets
        print(f"DEBUG: In train_main, config.target='{config.target}' (type: {type(config.target)})")
        (
            train_loader,
            val_loader,
            test_loader,
            prepare_batch,
            mean_train,
            std_train,
        ) = get_train_val_loaders(
            dataset=config.dataset,
            target=config.target,
            n_train=config.n_train,
            n_val=config.n_val,
            n_test=config.n_test,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            batch_size=config.batch_size,
            atom_features=config.atom_features,
            neighbor_strategy=config.neighbor_strategy,
            standardize=config.atom_features != "cgcnn",
            line_graph=line_graph,
            id_tag=config.id_tag,
            pin_memory=config.pin_memory,
            workers=config.num_workers,
            save_dataloader=config.save_dataloader,
            use_canonize=config.use_canonize,
            filename=config.filename,
            cutoff=config.cutoff,
            max_neighbors=config.max_neighbors,
            output_features=config.model.output_features,
            classification_threshold=config.classification_threshold,
            target_multiplication_factor=config.target_multiplication_factor,
            standard_scalar_and_pca=config.standard_scalar_and_pca,
            keep_data_order=config.keep_data_order,
            use_predetermined_splits=config.use_predetermined_splits,
            output_dir=config.output_dir,
            matrix_input=config.matrix_input,
            pyg_input=config.pyg_input,
            use_lattice=config.use_lattice,
            use_angle=config.use_angle,
            use_save=use_save,
            mp_id_list=mp_id_list,
            data_path=config.data_path if hasattr(config, 'data_path') else None,
        )
    else:
        train_loader = train_val_test_loaders[0]
        val_loader = train_val_test_loaders[1]
        test_loader = train_val_test_loaders[2]
        prepare_batch = train_val_test_loaders[3]
    prepare_batch = partial(prepare_batch, device=device)
    if classification:
        config.model.classification = True
    # define network, optimizer, scheduler
    _model = {
        "iComformer" : iComformer,
        "eComformer" : eComformer,
    }
    if std_train is None:
        std_train = 1.0
        print('std train is none!')
    print('std train:', std_train)
    if model is None:
        net = _model.get(config.model.name)(config.model)
        print("config:")
        pprint.pprint(config.model.dict())
    else:
        net = model

    net.to(device)
    if config.distributed:
        import torch.distributed as dist
        import os

        def setup(rank, world_size):
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"

            # initialize the process group
            dist.init_process_group("gloo", rank=rank, world_size=world_size)

        def cleanup():
            dist.destroy_process_group()

        setup(2, 2)
        net = torch.nn.parallel.DistributedDataParallel(
            net
        )
    params = group_decay(net)
    optimizer = setup_optimizer(params, config)

    if config.scheduler == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )

    elif config.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            # pct_start=pct_start,
            pct_start=0.3,
        )
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=100000,
            gamma=0.96,
        )
    elif config.scheduler == "polynomial":
        steps_per_epoch = len(train_loader)
        total_iter = steps_per_epoch * config.epochs
        scheduler = PolynomialLRDecay(optimizer, max_iters=total_iter, start_lr=0.0005, end_lr=0.00001, power=1)

    # select configured loss function
    criteria = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
    }
    criterion = criteria[config.criterion]
    # set up training engine and evaluators
    # SCALING DISABLED: To re-enable validation metric scaling, use:
    # metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError() * std_train, "neg_mae": -1.0 * MeanAbsoluteError() * std_train}
    metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError(), "neg_mae": -1.0 * MeanAbsoluteError()}
    trainer = create_supervised_trainer(
        net,
        optimizer,
        criterion,
        prepare_batch=prepare_batch,
        device=device,
        deterministic=deterministic,
    )
    evaluator = create_supervised_evaluator(
        net,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )
    train_evaluator = create_supervised_evaluator(
        net,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )
    if test_only:
        checkpoint_tmp = torch.load('/data/keqiangyan/Matformer/matformer/scripts/jarvis_results_new/ehull_inv_25nei/checkpoint_500.pt')
        to_load = {
            "model": net,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "trainer": trainer,
        }
        Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint_tmp)
        net.eval()
        targets = []
        predictions = []
        import time
        t1 = time.time()
        with torch.no_grad():
            for dat in test_loader:
                g, lg, _, target = dat
                out_data = net([g.to(device), lg.to(device), _.to(device)])
                out_data = out_data.cpu().numpy().tolist()
                target = target.cpu().numpy().flatten().tolist()
                if len(target) == 1:
                    target = target[0]
                targets.append(target)
                predictions.append(out_data)
        t2 = time.time()
        f.close()
        from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
        # SCALING DISABLED: To re-enable denormalization, use:
        # targets = np.array(targets) * std_train + mean_train
        # predictions = np.array(predictions) * std_train + mean_train
        targets = np.array(targets)  # No scaling needed
        predictions = np.array(predictions)  # No scaling needed
        print("Test MAE:", mean_absolute_error(targets, predictions))
        print("Test MAPE:", mean_absolute_percentage_error(targets, predictions))
        print("Total test time:", t2-t1)
        return mean_absolute_error(targets, predictions)

    # ignite event handlers:
    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    # apply learning rate scheduler
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )
    count_parameters(net)

    if config.write_checkpoint:
        # model checkpointing
        to_save = {
            "model": net,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "trainer": trainer,
        }
        handler = Checkpoint(
            to_save,
            DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=2,
            global_step_transform=lambda *_: trainer.state.epoch,
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)
        # evaluate save
        to_save = {"model": net}
        handler = Checkpoint(
            to_save,
            DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=5,
            filename_prefix='best',
            score_name="neg_mae",
            global_step_transform=lambda *_: trainer.state.epoch,
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)
    if config.progress:
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {"loss": x})
        # pbar.attach(evaluator,output_transform=lambda x: {"mae": x})

    history = {
        "train": {m: [] for m in metrics.keys()},
        "validation": {m: [] for m in metrics.keys()},
    }

    if config.store_outputs:
        # in history["EOS"]
        eos = EpochOutputStore()
        eos.attach(evaluator)
        train_eos = EpochOutputStore()
        train_eos.attach(train_evaluator)

    # collect evaluation performance
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        """Print training and validation metrics to console."""
        evaluator.run(val_loader)

        vmetrics = evaluator.state.metrics
        for metric in metrics.keys():
            vm = vmetrics[metric]
            t_metric = metric
            if metric == "roccurve":
                vm = [k.tolist() for k in vm]
            if isinstance(vm, torch.Tensor):
                vm = vm.cpu().numpy().tolist()

            history["validation"][metric].append(vm)

        
        
        epoch_num = len(history["validation"][t_metric])
        if epoch_num % 10 == 0:  # Evaluate training set every 10 epochs instead of 20
            train_evaluator.run(train_loader)
            tmetrics = train_evaluator.state.metrics
            for metric in metrics.keys():
                tm = tmetrics[metric]
                if metric == "roccurve":
                    tm = [k.tolist() for k in tm]
                if isinstance(tm, torch.Tensor):
                    tm = tm.cpu().numpy().tolist()

                history["train"][metric].append(tm)
        else:
            tmetrics = {}
            tmetrics['mae'] = -1

        if config.store_outputs:
            history["EOS"] = eos.data
            history["trainEOS"] = train_eos.data
            dumpjson(
                filename=os.path.join(config.output_dir, "history_val.json"),
                data=history["validation"],
            )
            dumpjson(
                filename=os.path.join(config.output_dir, "history_train.json"),
                data=history["train"],
            )
        if config.progress:
            pbar = ProgressBar()
            if not classification:
                pbar.log_message(f"Val_MAE: {vmetrics['mae']:.4f}")
                pbar.log_message(f"Train_MAE: {tmetrics['mae']:.4f}")
            else:
                pbar.log_message(f"Train ROC AUC: {tmetrics['rocauc']:.4f}")
                pbar.log_message(f"Val ROC AUC: {vmetrics['rocauc']:.4f}")

    if config.n_early_stopping is not None:
        if classification:
            my_metrics = "accuracy"
        else:
            my_metrics = "neg_mae"

        def default_score_fn(engine):
            score = engine.state.metrics[my_metrics]
            return score

        es_handler = EarlyStopping(
            patience=config.n_early_stopping,
            score_function=default_score_fn,
            trainer=trainer,
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, es_handler)
    # optionally log results to tensorboard
    if config.log_tensorboard:

        tb_logger = TensorboardLogger(
            log_dir=os.path.join(config.output_dir, "tb_logs", "test")
        )
        for tag, evaluator in [
            ("training", train_evaluator),
            ("validation", evaluator),
        ]:
            tb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names=["loss", "mae"],
                global_step_transform=global_step_from_engine(trainer),
            )

    trainer.run(train_loader, max_epochs=config.epochs)

    if config.log_tensorboard:
        test_loss = evaluator.state.metrics["loss"]
        tb_logger.writer.add_hparams(config, {"hparam/test_loss": test_loss})
        tb_logger.close()
    if config.write_predictions and classification:
        net.eval()
        f = open(
            os.path.join(config.output_dir, "prediction_results_test_set.csv"),
            "w",
        )
        f.write("id,target,prediction\n")
        targets = []
        predictions = []
        with torch.no_grad():
            ids = test_loader.dataset.ids  # [test_loader.dataset.indices]
            for dat, id in zip(test_loader, ids):
                g, lg, _, target = dat
                out_data = net([g.to(device), lg.to(device), _.to(device)])
                # out_data = torch.exp(out_data.cpu())
                top_p, top_class = torch.topk(torch.exp(out_data), k=1)
                target = int(target.cpu().numpy().flatten().tolist()[0])

                f.write("%s, %d, %d\n" % (id, (target), (top_class)))
                targets.append(target)
                predictions.append(
                    top_class.cpu().numpy().flatten().tolist()[0]
                )
        f.close()
        from sklearn.metrics import roc_auc_score

        print("predictions", predictions)
        print("targets", targets)
        print(
            "Test ROCAUC:",
            roc_auc_score(np.array(targets), np.array(predictions)),
        )

    if (
        config.write_predictions
        and not classification
        and config.model.output_features > 1
    ):
        net.eval()
        mem = []
        with torch.no_grad():
            ids = test_loader.dataset.ids  # [test_loader.dataset.indices]
            for dat, id in zip(test_loader, ids):
                g, lg, _, target = dat
                out_data = net([g.to(device), lg.to(device), _.to(device)])
                out_data = out_data.cpu().numpy().tolist()
                if config.standard_scalar_and_pca:
                    sc = pk.load(open("sc.pkl", "rb"))
                    out_data = list(
                        sc.transform(np.array(out_data).reshape(1, -1))[0]
                    )  # [0][0]
                target = target.cpu().numpy().flatten().tolist()
                info = {}
                info["id"] = id
                info["target"] = target
                info["predictions"] = out_data
                mem.append(info)
        dumpjson(
            filename=os.path.join(
                config.output_dir, "multi_out_predictions.json"
            ),
            data=mem,
        )

        # Calculate and print test MAE and MAPE for multi-output predictions
        from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
        all_targets = np.array([item["target"] for item in mem])
        all_predictions = np.array([item["predictions"] for item in mem])

        # Calculate MAE and MAPE for each output dimension
        n_outputs = all_targets.shape[1]
        for i in range(n_outputs):
            # SCALING DISABLED: To re-enable denormalization, use:
            # targets_i = all_targets[:, i] * std_train + mean_train
            # predictions_i = all_predictions[:, i] * std_train + mean_train
            targets_i = all_targets[:, i]  # No scaling needed
            predictions_i = all_predictions[:, i]  # No scaling needed
            mae_i = mean_absolute_error(targets_i, predictions_i)
            mape_i = mean_absolute_percentage_error(targets_i, predictions_i)
            if n_outputs == 2 and i == 0:
                print(f"Test MAE (WF_bottom): {mae_i:.4f}")
                print(f"Test MAPE (WF_bottom): {mape_i:.4f}")
            elif n_outputs == 2 and i == 1:
                print(f"Test MAE (WF_top): {mae_i:.4f}")
                print(f"Test MAPE (WF_top): {mape_i:.4f}")
            else:
                print(f"Test MAE (output {i}): {mae_i:.4f}")
                print(f"Test MAPE (output {i}): {mape_i:.4f}")

        # Calculate overall MAE and MAPE
        # SCALING DISABLED: To re-enable denormalization, use:
        # overall_mae = mean_absolute_error(all_targets.flatten() * std_train + mean_train, all_predictions.flatten() * std_train + mean_train)
        # overall_mape = mean_absolute_percentage_error(all_targets.flatten() * std_train + mean_train, all_predictions.flatten() * std_train + mean_train)
        overall_mae = mean_absolute_error(all_targets.flatten(), all_predictions.flatten())
        overall_mape = mean_absolute_percentage_error(all_targets.flatten(), all_predictions.flatten())
        print(f"Test MAE (overall): {overall_mae:.4f}")
        print(f"Test MAPE (overall): {overall_mape:.4f}")
    if (
        config.write_predictions
        and not classification
        and config.model.output_features == 1
    ):
        net.eval()
        targets = []
        predictions = []
        ids = []
        import time
        t1 = time.time()
        with torch.no_grad():
            from tqdm import tqdm
            test_ids = test_loader.dataset.ids
            for dat, id in zip(tqdm(test_loader), test_ids):
                g, lg, _, target = dat
                out_data = net([g.to(device), lg.to(device), _.to(device)])
                out_data = out_data.cpu().numpy().tolist()
                target = target.cpu().numpy().flatten().tolist()
                if len(target) == 1:
                    target = target[0]
                targets.append(target)
                predictions.append(out_data)
                ids.append(id)
        t2 = time.time()
        f.close()

        # Save predictions to JSON file
        mem = []
        for id, target, pred in zip(ids, targets, predictions):
            info = {}
            info["id"] = id
            info["target"] = target
            info["predictions"] = pred
            mem.append(info)
        
        dumpjson(
            filename=os.path.join(
                config.output_dir, "single_prop_predictions.json"
            ),
            data=mem,
        )
        
        from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
        # SCALING DISABLED: To re-enable denormalization, use:
        # targets = np.array(targets) * std_train + mean_train
        # predictions = np.array(predictions) * std_train + mean_train
        targets = np.array(targets)  # No scaling needed
        predictions = np.array(predictions)  # No scaling needed
        print("Test MAE:", mean_absolute_error(targets, predictions))
        print("Test MAPE:", mean_absolute_percentage_error(targets, predictions))
        
    return history


