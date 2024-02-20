import argparse
import torch
import numpy as np
from data_loader.data_loaders import get_train_val_datasets
import model.loss as module_loss
from model.model import SpeechDecodingModel, get_LDA_accuracy
from trainer import Trainer
from utils import prepare_device, read_json
from logger import get_logger
from os.path import join
from multiprocessing import freeze_support


# fix random seeds for reproducibility
SEED = 1379
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):

    config = read_json(config.config)
    config = argparse.Namespace(**config)

    resume = bool(config.checkpointfile)
    filemode = "w" if not resume else "a"
    logger = get_logger(config.logging_file, filemode)

    kmeans_folder = config.kmeans_folder
    path_input = join(kmeans_folder, "features")
    participant = "sub-06"

    feat = np.load(join(path_input, f"{participant}_feat.npy")).astype(np.float32)
    cluster = np.load(join(path_input, f"{participant}_melSpec_cluster.npy"))

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config.n_gpu)

    # setup data_loader instances
    train_dataset, val_dataset = get_train_val_datasets(
        feat,
        cluster,
        config.timepoints,
        config.num_eeg_channels,
        config.batch_size,
        config.epochs,
        device,
        config.validation_ratio,
        config.p_sample,
    )

    # build model architecture, then print to console
    model = SpeechDecodingModel(
        config.encoder_prenet_out_d,
        config.d_model,
        config.num_heads,
        config.num_layers,
        config.dim_feedforward,
        config.num_classes,
        config.dropout_prenet,
        config.dropout_encoder,
        config.enc_hidden_dim,
        config.timepoints,
        config.num_eeg_channels,
    )
    if not resume:
        logger(model, model=True)
    model = model.to(device)

    LDA_acc = get_LDA_accuracy(train_dataset, val_dataset)
    logger("Starting", right="=")
    logger(f"LDA accuracy              : {LDA_acc:5.2%}")
    logger(f"Train dataset length      : {len(train_dataset):5d}")
    logger(f"Validation dataset length : {len(val_dataset):5d}")
    histogram_weights = np.load(join(kmeans_folder, "histogram.npy"))
    histogram_weights = histogram_weights / np.sum(histogram_weights)
    logger(f"Max histogram weight      : {np.max(histogram_weights):5.2%}")

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = module_loss.lossfn(kmeans_folder, config.num_classes)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.optimizer["lr"],
        weight_decay=config.optimizer["weight_decay"],
        amsgrad=config.optimizer["amsgrad"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, config.lr_scheduler["step_size"], config.lr_scheduler["gamma"]
    )

    trainer = Trainer(
        model,
        criterion,
        optimizer,
        scheduler,
        train_dataset,
        val_dataset,
        config.batch_size,
        config.epochs,
        config.patience,
        logger,
        config.autosave_seconds,
        resume,
        config.checkpointfile,
    )
    trainer.train()


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="PyTorch Model Training")
    parser.add_argument(
        "-c",
        "--config",
        default="config.json",
        type=str,
        help="config file path (default: None)",
    )
    config = parser.parse_args()
    main(config)
