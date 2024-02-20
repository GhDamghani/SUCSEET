from functools import partial
from torch import optim, device, cuda

import numpy as np
from os.path import join

from model_loss import criterion


optimizer = partial(optim.Adam, lr=1e-4, weight_decay=1e-2, amsgrad=True)
lr_scheduler = partial(optim.lr_scheduler.ReduceLROnPlateau, factor=0.5, patience=5)

kmeans_folder = "../Kmeans of sounds"
path_input = join(kmeans_folder, "features")
participant = "sub-06"
feat = np.load(join(path_input, f"{participant}_feat.npy")).astype(np.float32)
cluster = np.load(join(path_input, f"{participant}_melSpec_cluster.npy"))
cluster_centers = np.load(
    join(path_input, f"{participant}_melSpec_cluster_centers.npy")
)

SEED = 1379456
DEVICE = device("cuda" if cuda.is_available() else "cpu")

d_model = 16
dim_feedforward = 128
num_heads = 4
num_layers = 1
timepoints = 96
num_classes = 20
num_eeg_channels = 127

dropout_prenet = 0.0
dropout_encoder = 0.0
dropout_clf = 0.0

VALIDATION_RATIO = 0.2
P_SAMPLE = 1.0

BATCH_SIZE = 16
EPOCHS = 100
LOG_TIMES = 10

AUTOSAVE_SECONDS = 600
CHECKPOINTFILE = None
EARLY_STOPPING_PATIENCE = 10
