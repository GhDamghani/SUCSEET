import torch
import numpy as np
from trainer import Trainer
from logger import get_logger
import loss_metrics


from model import SpeechDecodingModel_clf, SpeechDecodingModel_reg

import sys

master_path = "../.."
sys.path.append(master_path)

from vocoders.Griffin_Lim import createAudio
from vocoders.VocGAN import StreamingVocGan

import utils


def main(config):
    # fix random seeds for reproducibility

    torch.manual_seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.SEED)

    logger = get_logger("train.log", "w")

    if config.model_task == "classification":
        train_dataset, val_dataset, whole_dataset = next(
            iter(
                utils.data.WindowedData(
                    config.melSpec,
                    config.cluster,
                    window_size=config.window_size,
                    num_folds=config.nfolds,
                    output_size=1,
                    DCT_coeffs=config.DCT_coeffs,
                    preprocessing=config.preprocessing_list,
                )
            )
        )

        model = SpeechDecodingModel_clf(
            config.d_model,
            config.num_heads,
            config.dim_feedforward,
            config.num_layers,
            config.num_classes,
            config.window_size,
            config.num_eeg_channels,
            config.dropout,
        )
        loss, histogram_weights = loss_metrics.get_loss(
            train_dataset, config.num_classes, logger
        )
    elif config.model_task == "regression":
        train_dataset, val_dataset, whole_dataset = next(
            iter(
                utils.data.WindowedData(
                    config.feat,
                    config.melSpec,
                    window_size=config.window_size,
                    num_folds=config.nfolds,
                )
            )
        )

        model = SpeechDecodingModel_reg(
            config.d_model,
            config.num_heads,
            config.num_layers,
            config.window_size,
            config.num_eeg_channels,
            config.output_size,
            config.dropout,
        )
        loss = loss_metrics.mse_loss
    logger(model.__str__(config.BATCH_SIZE), model=True)
    model.to(config.DEVICE)

    logger("Starting", right="=")
    logger(f"Train dataset length      : {len(train_dataset):5d}")
    logger(f"Validation dataset length : {len(val_dataset):5d}")
    if config.model_task == "classification":
        logger(f"Max histogram value       : {np.max(histogram_weights):5.2%}")

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
        output_path=config.output_path,
    )
    trainer.train()


if __name__ == "__main__":
    import config

    config.config_proc(config.Config)

    config = config.Config

    main(config)
