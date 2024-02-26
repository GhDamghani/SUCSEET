import torch
import data
import numpy as np
from multiprocessing import freeze_support
from trainer import Trainer
from os.path import join
from logger import get_logger


from model_module import SpeechDecodingModel, get_LDA_accuracy


def main(config):
    # fix random seeds for reproducibility

    torch.manual_seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.SEED)

    logger = get_logger("train.log", "w")

    resume = False

    train_dataset, val_dataset = data.get_train_val_datasets(
        config.feat,
        config.cluster,
        config.timepoints,
        config.num_eeg_channels,
        config.BATCH_SIZE,
        config.EPOCHS,
        config.DEVICE,
        config.VALIDATION_RATIO,
        config.P_SAMPLE,
    )

    model = SpeechDecodingModel(
        config.d_model,
        config.num_heads,
        config.dim_feedforward,
        config.num_layers,
        config.num_classes,
        config.timepoints,
        config.num_eeg_channels,
        config.dropout_prenet,
        config.dropout_encoder,
        config.dropout_clf,
    )
    if not resume:
        logger(model.__str__(config.BATCH_SIZE), model=True)
    model.to(config.DEVICE)

    LDA_acc = get_LDA_accuracy(train_dataset, val_dataset)
    logger("Starting", right="=")
    logger(f"LDA accuracy              : {LDA_acc:5.2%}")
    logger(f"Train dataset length      : {len(train_dataset):5d}")
    logger(f"Validation dataset length : {len(val_dataset):5d}")

    histogram_weights = config.histogram_weights / np.sum(config.histogram_weights)
    logger(f"Max histogram weight      : {np.max(histogram_weights):5.2%}")

    # criterion = config.criterion(config.kmeans_folder, config.num_classes)
    criterion = config.criterion()

    # Define loss function and optimizer
    optimizer = config.optimizer(model.parameters())
    scheduler = config.lr_scheduler(optimizer)
    trainer = Trainer(
        model,
        criterion,
        optimizer,
        scheduler,
        train_dataset,
        val_dataset,
        config.BATCH_SIZE,
        config.EPOCHS,
        logger,
        config.AUTOSAVE_SECONDS,
        resume,
        config.CHECKPOINTFILE,
        config.LOG_TIMES,
    )
    trainer.train()


if __name__ == "__main__":
    freeze_support()
    import config

    main(config)
