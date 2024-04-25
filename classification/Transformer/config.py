from functools import partial
from torch import optim, device, cuda
import scipy
import numpy as np
from os.path import join, exists
from os import makedirs

import loss_metrics

import sys

master_path = "../.."
sys.path.append(master_path)


def config_proc(config):
    config.feat = np.load(join(config.path_input, f"{config.participant}_feat.npy"))
    config.num_eeg_channels = config.feat.shape[1]
    config.metrics = {"total": loss_metrics.total}

    if config.model_task == "classification":
        config.cluster = np.load(
            join(
                master_path,
                "results",
                "clustering",
                config.clustering_method,
                f"{config.participant}_spec_{config.vocoder}_cluster_{config.num_classes}_kfold_{config.nfolds}.npy",
            )
        )
        config.melSpec = np.load(
            join(config.path_input, f"{config.participant}_spec.npy")
        )
        config.num_eeg_channels = (
            config.melSpec.shape[1] if config.DCT_coeffs is None else config.DCT_coeffs
        )
        config.melSpec_centers = np.load(
            join(
                master_path,
                "results",
                "clustering",
                config.clustering_method,
                f"{config.participant}_spec_{config.vocoder}_cluster_{config.num_classes}_kfold_{config.nfolds}_centers.npy",
            )
        )
        config.metrics["corrects"] = loss_metrics.corrects
    elif config.model_task == "regression":
        config.melSpec = np.load(
            join(config.path_input, f"{config.participant}_spec.npy")
        )
        config.output_path = join(
            "..",
            "..",
            "regression",
            config.system_type,
            f"{config.window_size}",
            config.clf_name,
        )
        config.output_size = config.melSpec.shape[1]

    makedirs(config.output_path, exist_ok=True)


class Config:
    model_task = "classification"
    nfolds = 10
    dataset_type = "Word"
    vocoder = "VocGAN"  # "Griffin_Lim"
    participant = "sub-06"  # "p07_ses1_sentences"
    path_input = join(master_path, "dataset", dataset_type, vocoder)
    SEED = 1379456
    DEVICE = device("cuda" if cuda.is_available() else "cpu")

    if model_task == "classification":
        clustering_method = "kmeans"
        pca_components = None
        DCT_coeffs = 40
        preprocessing_list = ["normalize"]
        if DCT_coeffs is not None:
            preprocessing_list.append("DCT")
        clf_name = "Transformer_Encoder"

        num_classes = 5
        window_size = 40
        d_model = 256
        dim_feedforward = 1024
        num_heads = 16
        num_layers = 10
        dropout = 0.1
        optimizer = partial(optim.Adam, lr=1e-5, weight_decay=1e-2, amsgrad=True)
        BATCH_SIZE = 16
        EPOCHS = 50
    elif model_task == "regression":
        pca_components = None
        clf_name = "Transformer_Encoder"

        window_size = 40
        d_model = 125
        num_heads = 5
        num_layers = 6
        dropout = 0.25
        optimizer = partial(optim.Adam, lr=1e-5, weight_decay=1e-2, amsgrad=True)
        BATCH_SIZE = 16
        EPOCHS = 20

    LOG_TIMES = 10

    output_path = join(
        master_path, "results", "classification", f"{window_size}", clf_name
    )
