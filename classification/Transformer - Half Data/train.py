import torch
import numpy as np
from os.path import join, exists
import loss_metrics
import shutil

from model import SpeechDecodingModel_clf

import sys

master_path = "../.."
sys.path.append(master_path)

import utils
from utils.model_trainer.trainer import Trainer
from utils.model_trainer.logger import Logger


def main(config):
    # fix random seeds for reproducibility

    torch.manual_seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.SEED)

    logger_file = join("logs", f"train-{config.file_names.local}.log")
    if exists(logger_file):
        with open(logger_file, "r") as f:
            lines = f.readlines()
            if (
                len(lines) > 0
                and ("&" * 80) in lines[-1]
                or any([f"Epoch: {config.EPOCHS-1:04d}" in line for line in lines])
            ):
                return None
    logger = Logger(logger_file)

    dataset = iter(
        utils.data.WindowedData(
            config.feat,
            config.cluster,
            window_size=config.window_size,
            num_folds=config.nfolds,
            output_indices=config.output_indices,
        )
    )
    for _ in range(config.fold + 1):
        train_dataset, val_dataset = next(dataset)

    model = SpeechDecodingModel_clf(
        config.d_model,
        config.num_heads,
        config.dim_feedforward,
        config.num_layers,
        config.num_classes,
        config.window_size,
        config.feat_size,
        config.output_indices,
        config.dropout,
    )
    loss, histogram_weights = loss_metrics.get_loss(train_dataset, config.num_classes)
    model_summary = model.__str__(config.BATCH_SIZE)
    logger.log(model_summary, model=True)
    if config.fold == 0:
        shutil.copyfile(logger_file, config.file_names.file[:-2] + "all_model.txt")
        shutil.copyfile("model.py", config.file_names.file[:-2] + "model.py")
    model.to(config.DEVICE)

    logger.log("Starting", right="=")
    logger.log(f"Train dataset length      : {len(train_dataset):5d}")
    logger.log(f"Validation dataset length : {len(val_dataset):5d}")
    if config.model_task == "classification":
        logger.log(f"Max histogram value       : {np.max(histogram_weights):5.2%}")

    # Define loss function and optimizer
    optimizer = config.optimizer(model.parameters())
    trainer = Trainer(
        model,
        loss,
        config.metrics,
        val_dataset,
        config.model_task,
        train_dataset,
        optimizer,
        config.BATCH_SIZE,
        config.EPOCHS,
        logger,
        device=config.DEVICE,
        output_file_name=config.file_names.file,
    )
    trainer.train()
    logger.close()


def train_main(miniconfig):
    import config

    config0 = config.Config()
    config0.participant = miniconfig["participant"]
    config0.num_classes = miniconfig["num_classes"]
    config0.fold = miniconfig["fold"]
    config0.proc()
    main(config0)


# TODO: make it save model summary to file, and other stuff that might seem relevant
if __name__ == "__main__":
    from multiprocessing import Pool
    from itertools import product

    participants = [f"sub-{i:02d}" for i in (6,)]
    folds = [i for i in range(10)]
    nums_classes = (2, 5, 20)

    miniconfigs = [
        {"participant": participant, "num_classes": num_classes, "fold": fold}
        for participant, num_classes, fold in product(participants, nums_classes, folds)
    ]
    parallel = True
    if parallel:
        with Pool(processes=4) as pool:
            pool.map(train_main, miniconfigs)
    else:
        for miniconfig in miniconfigs:
            train_main(miniconfig)

    # import test

    # test.main()
    # import os

    # os.system("shutdown /s /t 5")
